import torch
import time
import numpy as np
import random
import os
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from mtl_model import MtlNeuroSAT
from core.data_loader import training_data_setup
from mtl_trainer import MtlTrainer
from utils.utils import get_log_path

# MTL Training Paradigm with multiple strategies (Naive, Sequential, Staged) and an additional Unsupervised Refinement phase (not integrated because of time constraints but can be easily integrated). Integrated with Optuna for hyperparameter tuning and early stopping based on validation performance. 

LOG_PATH = get_log_path()

class MTLParadigm:
    def __init__(self, opts, train_dataset="SR_Uniform_10-40_Dataset"):
        self.opts = opts
        self.device = opts.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.train_loader = training_data_setup(train_dataset, generate_labels=True)

        model = MtlNeuroSAT(
            d_model=opts.get('d_model', 64), 
            T=opts.get('T', 26)
        )
        self.trainer = MtlTrainer(model, self.device)
        self.init_random_seeds()

    def init_random_seeds(self):
        seed = self.opts.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _get_lrs(self, stage):
        """Extracts stage-specific LRs from opts, with sensible defaults."""
        if stage == 1:
            return {'core': self.opts.get('lr_stage1_core', 5e-5)}
        elif stage == 2:
            return {'adm': self.opts.get('lr_stage2_adm', 2e-5),
                    'ctp': self.opts.get('lr_stage2_ctp', 1e-4)}
        elif stage == 3:
            return {'core': self.opts.get('lr_stage3_core', 2e-5),
                    'adm':  self.opts.get('lr_stage3_adm',  4e-5),
                    'ctp':  self.opts.get('lr_stage3_ctp',  4e-5),
                    'uc':   self.opts.get('lr_stage3_uc',   4e-5)}
        elif stage == 4:
            return {'core': self.opts.get('lr_stage4', 1e-5)}

    def log_metrics(self, model_name, epoch, train_loss, val_acc, adm_acc, ctp_acc):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_filename = os.path.join(LOG_PATH, f"{model_name}_training_log.txt")
        is_new_file = not os.path.exists(log_filename)
        
        with open(log_filename, 'a') as f:
            if is_new_file:
                f.write(f"--- Training Log for Model: {model_name} ---\n")
            
            log_line = (
                f"[{timestamp}] Epoch: {epoch:04d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Acc: {val_acc:.6f} | "
                f"Adm Acc: {adm_acc:.6f} | "
                f"Ctp Acc: {ctp_acc:.6f}\n"
            )
            f.write(log_line)


    # Strategy 1: Train Everything Together
    def train_naive_together(self, test_sat, test_unsat, max_epochs=400, trial=None):
        print("Starting Naive Together Training...")
        if trial:
            model_name = f"MTL_Together_Trial{trial}"
        else:
            model_name = "MTL_Together"
        
        self.trainer.update_stage_optimizer(3, self._get_lrs(3))
        max_metric = 0 
        patience_counter = 0

        for epoch in range(max_epochs):
            avg_loss = self.trainer.train_epoch(self.train_loader, epoch, stage=3) 

            metrics = self.trainer.test(test_sat, test_unsat)
            c_acc, a_acc, ct_acc = metrics["class_acc"], metrics["adm_acc"], metrics["ctp_acc"]
            head_acc = (2 * a_acc * ct_acc) / (a_acc + ct_acc + 1e-8)
            tracking_metric = (head_acc + c_acc) / 2
    
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Class: {c_acc:.4f} | ADM: {a_acc:.4f} | CTP: {ct_acc:.4f}")
            self.log_metrics(model_name, epoch + 1, avg_loss, c_acc, a_acc, ct_acc)

            if tracking_metric > max_metric:
                max_metric = tracking_metric
                patience_counter = 0
                self.trainer.save(epoch, avg_loss, f"{model_name}_best.pth")
            else:
                patience_counter += 1
    
            if patience_counter >= 10 and c_acc > 0.6:
                print(">>> Early stopping")
                break

            if trial is not None:
                trial.report(tracking_metric, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        print("Together Training Finished.")
        return max_metric


    # Strategy 2: Train Sequentially (Core, then Heads)
    def train_sequential(self, test_sat, test_unsat, max_epochs=400, trial=None):
        print("Starting Sequential Training...")
        if trial:
            model_name = f"MTL_Sequential_Trial{trial}"
        else:
            model_name = "MTL_Sequential"
        
        # Phase 1: Core Only
        self.trainer.update_stage_optimizer(1, self._get_lrs(1))
        max_metric = 0
        patience = 0
        current_stage = 1

        for epoch in range(max_epochs):
            avg_loss = self.trainer.train_epoch(self.train_loader, epoch, stage=current_stage)
            metrics = self.trainer.test(test_sat, test_unsat)
            
            c_acc, a_acc, ct_acc = metrics["class_acc"], metrics["adm_acc"], metrics["ctp_acc"]
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Class: {c_acc:.4f} | ADM: {a_acc:.4f} | CTP: {ct_acc:.4f}")
            self.log_metrics(model_name, epoch + 1, avg_loss, c_acc, a_acc, ct_acc)

            # Phase 1 -> Phase 2 Transition
            if current_stage == 1:
                if c_acc > max_metric:
                    max_metric = c_acc
                    patience = 0
                    self.trainer.save(epoch, avg_loss, f"{model_name}_core_best.pth")
                else:
                    patience += 1

                if patience >= 10 and c_acc > 0.6:
                    print(">>> Core Matured. Freezing Core, Training Heads.")
                    current_stage = 2

                    epoch_trained = self.trainer.restore(f"{model_name}_core_best.pth")
                    self.trainer.update_stage_optimizer(2, self._get_lrs(2))
                    max_metric = 0; patience = 0

            # Phase 2 Logic
            elif current_stage == 2:
                head_acc = (2 * a_acc * ct_acc) / (a_acc + ct_acc + 1e-8)
                if head_acc > max_metric:
                    max_metric = head_acc
                    patience = 0
                    self.trainer.save(epoch, avg_loss, f"{model_name}_heads_best.pth")
                else:
                    patience += 1

                if patience >= 10 and head_acc > 0.6:
                    print(">>> Heads Matured. Early Stopping.")
                    break

                if trial is not None:
                    trial.report(head_acc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

        print("Sequential Training Finished.")
        return max_metric


    # Strategy 3: Staged (1 -> 2 -> 3)
    def train_staged_curriculum(self, test_sat, test_unsat, max_epochs=400, trial=None):
        print("Starting Staged Training...")
        if trial:
            model_name = f"MTL_Staged_Trial{trial}"
        else:
            model_name = "MTL_Staged"
        
        # Phase 1: Core Only
        self.trainer.update_stage_optimizer(1, self._get_lrs(1))
        max_metric = 0
        patience = 0
        current_stage = 1
        buffer = 0

        for epoch in range(max_epochs):
            avg_loss = self.trainer.train_epoch(self.train_loader, epoch, stage=current_stage)
            metrics = self.trainer.test(test_sat, test_unsat)
            
            c_acc, a_acc, ct_acc = metrics["class_acc"], metrics["adm_acc"], metrics["ctp_acc"]
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Class: {c_acc:.4f} | ADM: {a_acc:.4f} | CTP: {ct_acc:.4f}")
            self.log_metrics(model_name, epoch + 1, avg_loss, c_acc, a_acc, ct_acc)

            # Phase 1 -> Phase 2 Transition
            if current_stage == 1:
                if c_acc > max_metric:
                    max_metric = c_acc
                    patience = 0
                    self.trainer.save(epoch, avg_loss, f"{model_name}_core_best.pth")
                else:
                    patience += 1

                if c_acc > 0.61:
                    print(">>> Core Reasonably Matured. Freezing Core, Training Heads.")
                    current_stage = 2
                    self.trainer.update_stage_optimizer(2, self._get_lrs(2))
                    max_metric = 0
                    patience = 0

            # Phase 2 Logic
            elif current_stage == 2:
                head_acc = (2 * a_acc * ct_acc) / (a_acc + ct_acc + 1e-8)
                buffer += 1
                if head_acc > max_metric:
                    max_metric = head_acc
                    patience = 0
                    self.trainer.save(epoch, avg_loss, f"{model_name}_heads_best.pth")
                else:
                    patience += 1

                if head_acc > 0.74 and buffer > 5:
                    print(">>> Heads Reasonably Matured. Starting Full End-to-End.")
                    current_stage = 3
                    self.trainer.update_stage_optimizer(3, self._get_lrs(3))
                    max_metric = 0
                    patience = 0

            elif current_stage == 3:
                head_acc = (2 * a_acc * ct_acc) / (a_acc + ct_acc + 1e-8)
                full_acc = (c_acc + head_acc) / 2
                if full_acc >= max_metric:
                    max_metric = full_acc
                    patience = 0
                    self.trainer.save(epoch, avg_loss, f"{model_name}_final_best.pth")
                else:
                    patience += 1

                if patience >= 15 and head_acc > 0.6:
                    print(f"Early stopping triggered at Stage 3. Training Complete.")
                    break

                if trial is not None:
                    trial.report(full_acc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    
        print("Staged Training Finished.")
        return max_metric


    # Strategy 4: Standalone Unsupervised Refinement (Load & Fine-Tune)
    def run_unsupervised_refinement(self, checkpoint_filename, test_sat, test_unsat, max_epochs=100):
        print(f"Loading unrefined model from '{checkpoint_filename}' for Stage 4 Refinement...")
        model_name = f"{checkpoint_filename}_Refined"

        # 1. Load the pre-trained Stage 3 model
        epoch_trained = self.trainer.restore(checkpoint_filename)
        if epoch_trained is None:
            print("Failed to load checkpoint. Aborting refinement.")
            return

        print(f"Model loaded successfully! (Previously trained for {epoch_trained} epochs).")
        print("Switching to Unsupervised Logical Refinement...")

        # 2. Initialize Stage 4 Optimizer (Unfreeze all, very low learning rate)
        self.trainer.update_stage_optimizer(4, self._get_lrs(4))

        max_metric = 0
        patience_counter = 0

        # 3. Stage 4 Training Loop
        for epoch in range(max_epochs):
            avg_loss = self.trainer.train_epoch(self.train_loader, epoch, stage=4)            
            metrics = self.trainer.test(test_sat, test_unsat)
            
            solve_rate = metrics["solve_rate"]
            c_acc = metrics["class_acc"]
            
            print(f"Refinement Epoch {epoch+1} | Loss: {avg_loss:.4f} | Solve Rate: {solve_rate*100:.2f}% | Class Acc: {c_acc:.4f}")

            # Checkpoint based on the logical solve rate
            if solve_rate > max_metric:
                max_metric = solve_rate
                patience_counter = 0
                self.trainer.save(epoch, avg_loss, f"{model_name}_best.pth")
                print(f"  -> New best solve rate! Saved model.")
            else:
                patience_counter += 1

            if patience_counter >= 15:
                print(f"\n>>> Early stopping triggered. Refinement complete. Best Solve Rate: {max_metric*100:.2f}%")
                break
                
        print("Unsupervised Refinement Finished.")
        return max_metric
    

def objective(trial, train, val, mode):
    opts = {
        'lr_stage1_core': trial.suggest_float('lr_stage1_core', 2e-5, 7e-5, log=True),
        'lr_stage2_adm':  trial.suggest_float('lr_stage2_adm',  1e-5, 1e-3, log=True),
        'lr_stage2_ctp':  trial.suggest_float('lr_stage2_ctp',  1e-5, 1e-3, log=True),
        'lr_stage3_core': trial.suggest_float('lr_stage3_core', 1e-5, 5e-5, log=True),
        'lr_stage3_adm':  trial.suggest_float('lr_stage3_adm',  1e-5, 1e-3, log=True),
        'lr_stage3_ctp':  trial.suggest_float('lr_stage3_ctp',  1e-5, 1e-3, log=True),
        'lr_stage3_uc':   trial.suggest_float('lr_stage3_uc',   1e-5, 1e-3, log=True),
        'd_model': 64,
        'T': 26,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'seed': 42
    }

    val_sat = f"{val}_SAT"
    val_unsat = f"{val}_UNSAT"

    experiment = MTLParadigm(opts, train_dataset=train)
    if mode == "sequential":
        metrics = experiment.train_sequential(
            test_sat=val_sat, 
            test_unsat=val_unsat,
            max_epochs=400,
            trial=trial
        )
        return metrics  

    elif mode == "naive":
        metrics = experiment.train_naive_together(
            test_sat=val_sat, 
            test_unsat=val_unsat,
            max_epochs=400,
            trial=trial
        )
        return metrics  

    elif mode == "staged":
        metrics = experiment.train_staged_curriculum(
            test_sat=val_sat, 
            test_unsat=val_unsat,
            max_epochs=400,
            trial=trial
        )
        return metrics  

    else:
        raise Exception


def auto_hyperparameter_tuning(train, val, mode):
    # Setup the Optuna Pruner
    pruner = SuccessiveHalvingPruner(
        min_resource=20,          # Allow trials to run for at least 20 epochs before judging them
        reduction_factor=3        # Halves the number of trials each round
    )
    study = optuna.create_study(direction="maximize", pruner=pruner)

    def wrapped_objective(trial):
        return objective(trial, train, val, mode)
    
    study.optimize(wrapped_objective, n_trials=10)

    print("\n==================================")
    print("Hyperparameter Tuning Complete!")
    print("Best Trial:", study.best_trial.number)
    print("Best Accuracy:", study.best_value)
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("==================================")


def run_mtl_staged_training(train, val):
    """Runs the full staged training paradigm with default hyperparameters. No automated hyperparameter tuning due to time constraints."""
    opts = {
        'd_model': 64,
        'T': 26,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'seed': 42
    }
    experiment = MTLParadigm(opts, train_dataset=train)

