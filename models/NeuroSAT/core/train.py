import os
import torch
import shutil
from torch.utils.data import DataLoader, ConcatDataset
from data_loader import SATDataset
from Neurosat import NeuroSAT
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from utils.utils import get_checkpoint_path, get_load_path

# automated training function with early stopping and best model saving, as well as an Optuna-based hyper-parameter tuning function with pruning.

CHECKPOINT_PATH = get_checkpoint_path()
LOAD_PATH = get_load_path()

# training + hyper-parameter tuning

def build_model(train, val):
    # 1. Hyperparameters & Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "NeuroSAT_SR40"
    
    opts = {
        'd_model': 64,          # Embedding dimension (D)
        'T': 26,                # Message passing iterations
        'learning_rate': 2e-5,
        'clip_val': 0.65,       # Gradient clipping
        'seed': 42,
        'device': DEVICE
    }

    BATCH_SIZE = 128
    MAX_EPOCHS = 400
    SAVE_INTERVAL = 50
    PATIENCE_LIMIT = 10

    # 2. Data Preparation
    print("Loading datasets...")
    train_data = SATDataset(data_file=train, is_training=True)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_sat_data = SATDataset(data_file=f"{val}_SAT", is_training=False, fixed_label=1)
    test_unsat_data = SATDataset(data_file=f"{val}_UNSAT", is_training=False, fixed_label=0)
    merged_test_data = ConcatDataset([test_sat_data, test_unsat_data])
    test_loader = DataLoader(merged_test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Solver
    print(f"\nInitializing NeuroSAT on {DEVICE}...")
    solver = NeuroSAT(opts)

    # Resume from checkpoint if it exists
    best_model_path = os.path.join(CHECKPOINT_PATH, f"{MODEL_NAME}.pth")
    solver.restore(best_model_path)

    # 4. Training Loop
    max_acc = 0.0
    patience_counter = 0

    print("\nStarting Training...")
    for epoch in range(MAX_EPOCHS):
        # Train for one epoch
        avg_loss, duration = solver.train_epoch(train_loader, epoch)
        print(f"--- Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Time: {duration:.2f}s ---")

        # Evaluate on the eval set
        curr_acc, conf_matrix = solver.test(test_loader)
        print(f"Validation Accuracy: {curr_acc:.4f}")

        # Periodic Saving
        if (epoch + 1) % SAVE_INTERVAL == 0:
            periodic_path = os.path.join(CHECKPOINT_PATH, f"{MODEL_NAME}_epoch_{epoch+1}.pth")
            solver.save(periodic_path, epoch, avg_loss)
            print(f"Saved periodic checkpoint.")

        # Best Model Tracking & Early Stopping
        if curr_acc > max_acc:
            max_acc = curr_acc
            patience_counter = 0
            
            # Save if it's the best accuracy 
            solver.save(best_model_path, epoch, avg_loss)
            print(f"*** New Best Model Saved! (Acc: {max_acc:.4f}) ***")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE_LIMIT and avg_loss < 0.6:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    print(f"\nTraining Complete. Best Validation Accuracy: {max_acc:.4f}")
    dest = shutil.move(best_model_path, LOAD_PATH)
    return f"{MODEL_NAME}.pth"


def objective(trial, train_loader, test_loader, device):
    # 1. Define the hyperparameter search space for this trial
    d_model = trial.suggest_categorical('d_model', [64, 96, 128])
    T = trial.suggest_int('T', 20, 30)
    lr = trial.suggest_float('lr', 1e-5, 5e-5, log=True)

    # 2. Package into the opts dictionary for our NeuroSAT class
    opts = {
        'd_model': d_model,
        'T': T,
        'learning_rate': lr,
        'clip_val': 0.65,
        'device': device,
        'seed': 42
    }

    # 3. Initialize the solver for this trial
    solver = NeuroSAT(opts)
    
    # 4. Trial Configuration
    MAX_EPOCHS = 250
    SAVE_INTERVAL = 100
    PATIENCE_LIMIT = 10 
    
    best_acc = 0.0
    patience_counter = 0
    model_name = f"M-Trial{trial.number}-T{T}-D{d_model}-L{lr:.2e}"

    print(f"\n=== Starting Trial {trial.number}: T={T}, D={d_model}, LR={lr:.2e} ===")

    for epoch in range(MAX_EPOCHS):
        # Train and Test
        avg_loss, _ = solver.train_epoch(train_loader, epoch)
        curr_acc, _ = solver.test(test_loader)

        # Optuna pruning check (reports intermediate value to Optuna)
        trial.report(curr_acc, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned by Optuna at epoch {epoch}.")
            raise optuna.exceptions.TrialPruned()

        # Update best accuracy and manage custom early stopping patience
        if curr_acc > best_acc:
            best_acc = curr_acc
            patience_counter = 0
            
            # Save the best model for this specific trial
            best_ckpt = os.path.join('./NNSAT_Project/Checkpoints', f"{model_name}_BEST.pth")
            solver.save(best_ckpt, epoch, avg_loss)
        else:
            patience_counter += 1

        # Periodic checkpointing 
        if (epoch + 1) % SAVE_INTERVAL == 0:
            periodic_ckpt = os.path.join('./NNSAT_Project/Checkpoints', f"{model_name}_epoch{epoch + 1}.pth")
            solver.save(periodic_ckpt, epoch, avg_loss)

        # Patience-based custom early stopping
        if patience_counter >= PATIENCE_LIMIT and avg_loss < 0.6:
            print(f"Trial {trial.number} stopped at epoch {epoch} due to custom patience.")
            break

    return best_acc


def auto_hyperparameter_tuning(train, val):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    print(f"Setting up Optuna Study on {DEVICE}...")

    # Load datasets 
    print("Loading datasets...")
    train_data = SATDataset(train, is_training=True)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_sat_data = SATDataset(f"{val}_SAT", is_training=False, fixed_label=1)
    test_unsat_data = SATDataset(f"{val}_UNSAT", is_training=False, fixed_label=0)
    merged_test_data = ConcatDataset([test_sat_data, test_unsat_data])
    test_loader = DataLoader(merged_test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Setup the Optuna Pruner
    pruner = SuccessiveHalvingPruner(
        min_resource=20,          # Allow trials to run for at least 20 epochs before judging them
        reduction_factor=3        # Halves the number of trials each round
    )
    
    study = optuna.create_study(direction="maximize", pruner=pruner)

    # Wrap the objective to inject the data loaders
    def wrapped_objective(trial):
        return objective(trial, train_loader, test_loader, DEVICE)

    # Run the optimization
    study.optimize(wrapped_objective, n_trials=10)
    
    print("\n==================================")
    print("Hyperparameter Tuning Complete!")
    print("Best Trial:", study.best_trial.number)
    print("Best Accuracy:", study.best_value)
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("==================================")


def run_NeuroSAT_training(train="SR_Uniform_10-40_Dataset", val="Val_40"):
    return build_model(train, val)
    # auto_hyperparameter_tuning(train, val)