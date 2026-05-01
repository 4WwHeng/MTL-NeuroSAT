import os
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.data_preprocessing import read_data
from torch.utils.data import Dataset, DataLoader, random_split
from core.inference import NN_inference
from sat_lit import decode_kmeans_dist
from utils.utils import get_close_assignment, get_checkpoint_path, get_data_path, get_log_path, get_model_name
import optuna
from optuna.pruners import SuccessiveHalvingPruner

# Training for Assignment Decoding Model (ADM), including getting labels embeddings from NeuroSAT.

DATA_PATH = get_data_path()
CHECKPOINT_PATH = get_checkpoint_path()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = get_log_path()

# get labelled literal embeddings from NeuroSAT and save to h5 file for ADM training
def save_instance(h5_filename, instance_id, embeddings, vote, labels):
    h5_filename = os.path.join(DATA_PATH, h5_filename)
    with h5py.File(h5_filename, 'a') as f:
        grp = f.create_group(f"instance_{instance_id}")
        grp.create_dataset("embeddings", data=embeddings.astype('float32'))
        grp.create_dataset("labels", data=labels.astype('int8'))
        grp.attrs["vote"] = vote


def save_labelled_lit_embedding(model_name, test_data="SR(U(10,40))", save_name=""):
    votes, lit_emb, clause_emb, var_votes, latency = NN_inference(model_name, test_data)
    
    data_sat = read_data(f"{test_data}_SAT", is_training=False, fixed_label=1)
    data_unsat = read_data(f"{test_data}_UNSAT", is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if not is_sat_ground_truth:
            continue

        vote = votes[i]
        # all sat instances were used as training data

        L_h = lit_emb[i]
        C_h = clause_emb[i]

        k_means_candidate, var_dist, direct_solved = decode_kmeans_dist(L_h, clauses, n_vars)
        if not direct_solved:
            truth_assignment = get_close_assignment(k_means_candidate, clauses, n_vars)
        else:
            truth_assignment = k_means_candidate
        if truth_assignment is None:
            continue
        lit_truth_assignment = np.concatenate([truth_assignment, 1 - truth_assignment])        
        
        save_instance(f"lit_emb_{save_name}.h5", i, L_h, vote, lit_truth_assignment)

        if (i+1) % 100 == 0:
            print(f"Processed {i+1}...")


#  Assignment Decoding Model + Training
lit_loss_fn = nn.CrossEntropyLoss()
class LiteralDataset(Dataset):
    def __init__(self, h5_file):
        self.all_features = []
        self.all_targets = []

        h5_file = os.path.join(DATA_PATH, h5_file)
        with h5py.File(h5_file, 'r') as f:
            for inst_id in f.keys():
                emb = f[inst_id]['embeddings'][:] # type: ignore
                vote = f[inst_id].attrs['vote'] # type: ignore
                labels = f[inst_id]['labels'][:] # type: ignore
                
                # Pre-process features (65D) by concatenating the vote to each literal embedding
                features = np.hstack((emb, np.full((emb.shape[0], 1), vote))) # type: ignore
                
                self.all_features.append(features)
                self.all_targets.append(labels)
        
        self.X = torch.from_numpy(np.vstack(self.all_features)).float()
        self.y = torch.from_numpy(np.concatenate(self.all_targets)).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AssignmentDecodingModel(nn.Module):
    def __init__(self, dimension=128, dropout=0.2):
        super(AssignmentDecodingModel, self).__init__()
        # 64 embeddings + 1 Vote 
        self.norm = nn.LayerNorm(65)
        
        self.network = nn.Sequential(
            nn.Linear(65, dimension),
            nn.ReLU(),
            nn.Dropout(dropout), # Prevents overfitting
            nn.Linear(dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Output: [0:False, 1:True]
        )
        
    def forward(self, x):
        x = self.norm(x)
        return self.network(x)


def train_epoch_adm(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0
    total_samples = 0 
    
    for i, (features, target) in enumerate(dataloader):
        features, target = features.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        
        outputs = model(features) 
        loss = lit_loss_fn(outputs, target) 
        
        loss.backward()
        optimizer.step()
                
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (i + 1) % 100 == 0: 
            print(f"--- Epoch {epoch + 1} | Batch {i + 1}/{len(dataloader)} | Loss: {loss.item():.6f}")

    return total_loss / total_samples


def save_checkpoint(epoch, model, optimizer, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, filename))


def test_adm(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, target in val_loader:
            features, target = features.to(DEVICE), target.to(DEVICE)
            
            outputs = model(features)
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy


def log_metrics(model_name, epoch, train_loss, val_acc):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_filename = os.path.join(LOG_PATH, f"{model_name}_training_log.txt")
    is_new_file = not os.path.exists(log_filename)
    
    with open(log_filename, 'a') as f:
        if is_new_file:
            f.write(f"--- Training Log for Model: {model_name} ---\n")
        
        log_line = (
            f"[{timestamp}] Epoch: {epoch:04d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Acc: {val_acc:.6f}\n"
        )
        f.write(log_line)


# training loop (with cross validation)
def train_adm_model():
    max_num_epochs = 300
    print(f"Starting Clause Tier Predictor Training on {DEVICE}...")

    model = AssignmentDecodingModel()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    max_acc = 0 
    current_epoch = 0
    save_interval = 25
    patience_counter = 0

    full_dataset = LiteralDataset("lit_emb_train.h5")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1024, shuffle=False)
    
    MODEL_NAME = "AssignmentDecodingModel"
    model.to(DEVICE) 
    
    for epoch in range(current_epoch, max_num_epochs):
        avg_loss = train_epoch_adm(model, train_loader, optimizer, epoch)
        print(f"\n--- Epoch {epoch + 1}: Training Loss = {avg_loss:.4f} ---")

        curr_acc = test_adm(model, val_loader)
        print(f"Validation Accuracy: {curr_acc:.4f}\n")

        log_metrics(MODEL_NAME, epoch + 1, avg_loss, curr_acc)
        
        save_this_epoch = False
        saved_name = None
        
        if epoch % save_interval == 0:
            saved_name = f"{MODEL_NAME}_{epoch+1}.pth"
            save_checkpoint(epoch, model, optimizer, avg_loss, saved_name)
            save_this_epoch = True
            print(f"Saved periodic checkpoint: {saved_name}")
        
        if curr_acc > max_acc:
            max_acc = curr_acc
            patience_counter = 0
    
            if not save_this_epoch: 
                saved_name = f"{MODEL_NAME}_best.pth"
                save_checkpoint(epoch, model, optimizer, avg_loss, saved_name)
                print(f"Saved BEST model: {saved_name}")
        else:
            patience_counter += 1

        if patience_counter >= 10 and avg_loss < 0.6:
            print("Early stopping triggered.")
            break

    name = f"{MODEL_NAME}_{epoch}_final.pth"
    save_checkpoint(epoch, model, optimizer, avg_loss, name)


# ADM auto hyperparameter tuning
def objective(trial, data_loader, val_loader):
    # Optuna will explore these values
    dimension = trial.suggest_int("dimension", 64, 256)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = AssignmentDecodingModel(dimension, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_num_epochs = 400
    best_acc = 0
    patience_counter = 0
    MODEL_NAME = f"AssignmentDecodingModel-D{dimension}-Dr{dropout}-L{lr}"
    model.to(DEVICE) 

    features, targets = next(iter(data_loader)) # Get one batch
    features, targets = features.to(DEVICE), targets.to(DEVICE)
    
    # with torch.no_grad():
    #     initial_output = model(features)
    #     initial_loss = loss_fn(initial_output, targets)
    #     print(f"Trial {trial.number} Initial Loss (Zero Training): {initial_loss.item():.4f}")

    save_interval = 25
    for epoch in range(max_num_epochs):
        avg_loss = train_epoch_adm(model, data_loader, optimizer, epoch)
        print(f"\n--- Epoch {epoch + 1}: Training Loss = {avg_loss:.4f} ---")

        curr_acc = test_adm(model, val_loader)
        print(f"Validation Accuracy: {curr_acc:.4f}\n")


        log_metrics(MODEL_NAME, epoch + 1, avg_loss, curr_acc)

        if epoch % save_interval == 0:
            saved_name = f"{MODEL_NAME}_{epoch+1}.pth"
            save_checkpoint(epoch, model, optimizer, avg_loss, saved_name)
        
        if curr_acc > best_acc:
            best_acc = curr_acc
            patience_counter = 0
    
            saved_name = f"{MODEL_NAME}_best.pth"
            save_checkpoint(epoch, model, optimizer, avg_loss, saved_name)
            print(f"Saved BEST model: {saved_name}")
        else:
            patience_counter += 1

        # Optuna pruning (all trials)
        trial.report(curr_acc, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch}.")
            raise optuna.exceptions.TrialPruned()

        if patience_counter >= 10 and avg_loss < 0.6:
            print("Early stopping triggered.")
            break
    
    return best_acc


# auto hyperparameter tuning for ADM
def auto_hyperparameter_tuning_adm():
    data_loader = DataLoader(LiteralDataset("lit_emb_train.h5"), batch_size=1024, shuffle=True)
    val_loader = DataLoader(LiteralDataset("lit_emb_val.h5"), batch_size=1024, shuffle=False)

    pruner = SuccessiveHalvingPruner(
        min_resource=5,          # start pruning after 5 epochs
        reduction_factor=3        # halves trials each round
    )
    
    study = optuna.create_study(direction="maximize", pruner=pruner)

    def wrapped_objective(trial):
        return objective(trial, data_loader, val_loader)

    study.optimize(wrapped_objective, n_trials=10)
    
    print("Best trial:")
    print(study.best_params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    auto_hyperparameter_tuning_adm()

