import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch_geometric.utils import scatter
from mtl_model import UncertaintyWeightedLoss
from core.data_loader import testing_data_setup
from refinement import compute_clause_unsat_loss, compute_penalty_diffsat_loss
from utils.utils import get_checkpoint_path, get_load_path

# Main MTL model class. Includes methods for training, checkpointing, and evaluation. The training loop is designed to handle the 3-stage curriculum with dynamic freezing and loss weighting. The inference method extracts all relevant outputs in one pass for efficiency.

CHECKPOINT_PATH = get_checkpoint_path()
LOAD_PATH = get_load_path()

class MtlTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
        # Loss Functions
        self.global_loss_fn = nn.BCEWithLogitsLoss()
        self.mlp_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.uc_loss_fn = UncertaintyWeightedLoss(n_tasks=3).to(device)

        # State
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5, weight_decay=1e-10)

    # Optimizer & Checkpoint Management
    def update_stage_optimizer(self, stage, lrs=None):
        """
        Dynamically freezes layers and adjusts learning rates.
            - Stage 1: Train Core Only
            - Stage 2: Train ADM and CTP, freeze Core
            - Stage 3: Train all with Uncertainty Loss
            - Stage 4: Refinement stage with custom loss
        """
        if lrs is None:
            lrs = {}
        lr_core = lrs.get('core', 5e-5)
        lr_adm  = lrs.get('adm',  5e-5)
        lr_ctp  = lrs.get('ctp',  5e-5)
        lr_uc   = lrs.get('uc',   5e-5)

        if stage == 1:
            for p in self.model.core.parameters(): p.requires_grad = True
            for p in self.model.adm.parameters(): p.requires_grad = False 
            for p in self.model.ctp.parameters(): p.requires_grad = False 
            for p in self.uc_loss_fn.parameters(): p.requires_grad = False 
            self.optimizer = optim.Adam(self.model.core.parameters(), lr=lr_core, weight_decay=1e-10)
            
        elif stage == 2:
            for p in self.model.core.parameters(): p.requires_grad = False
            for p in self.model.adm.parameters(): p.requires_grad = True
            for p in self.model.ctp.parameters(): p.requires_grad = True
            for p in self.uc_loss_fn.parameters(): p.requires_grad = False
            self.optimizer = optim.Adam([
                {'params': self.model.adm.parameters(), 'lr': lr_adm},
                {'params': self.model.ctp.parameters(), 'lr': lr_ctp},
            ], weight_decay=1e-10)
                
        elif stage == 3:
            for p in self.model.core.parameters(): p.requires_grad = True
            for p in self.uc_loss_fn.parameters(): p.requires_grad = True
            self.optimizer = optim.Adam([
                {'params': self.model.core.parameters(), 'lr': lr_core}, 
                {'params': self.model.adm.parameters(), 'lr': lr_adm},
                {'params': self.model.ctp.parameters(), 'lr': lr_ctp},
                {'params': self.uc_loss_fn.parameters(), 'lr': lr_uc}
            ], weight_decay=1e-10)

        elif stage == 4:
            for p in self.model.parameters(): p.requires_grad = True
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-10)
            
    def save(self, epoch, loss, filename):
        """Saves a checkpoint file with separate states for the core, ADM, CTP, and optimizer."""
        checkpoint = {
            'epoch': epoch,
            #'model_state_dict': self.model.state_dict(), # Saves all 3 networks at once
            'model_state_dict': self.model.core.state_dict(),
            'adm_state_dict': self.model.adm.state_dict(),
            'ctp_state_dict': self.model.ctp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), #type: ignore
            'loss': loss,
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, filename))

    def restore(self, filename):
        """Restores the model and optimizer state from a checkpoint file."""
        checkpoint_path = os.path.join(LOAD_PATH, filename)
        if not os.path.exists(checkpoint_path):
            return None
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.core.load_state_dict(checkpoint['model_state_dict'])
        self.model.adm.load_state_dict(checkpoint['adm_state_dict'])
        self.model.ctp.load_state_dict(checkpoint['ctp_state_dict'])
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is None:
            self.update_stage_optimizer(stage=3)
            
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    

    def restore_seperate(self, core, adm, ctp):
        """Restores the model and optimizer state from separate checkpoint files for core, ADM, and CTP."""
        core = torch.load(core, map_location=self.device)
        self.model.core.load_state_dict(core['model_state_dict'])
        adm = torch.load(adm, map_location=self.device)
        self.model.adm.load_state_dict(adm['model_state_dict'])
        ctp = torch.load(ctp, map_location=self.device)
        self.model.ctp.load_state_dict(ctp['model_state_dict'])
        
        if self.optimizer is None:
            self.update_stage_optimizer(stage=3)
            
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return core['epoch']

    # Evaluation (used for validation and early stopping, separate from inference which extracts raw outputs for analysis)
    def evaluate(self, dataloader):
        """Evaluates the model on the provided dataloader, returning classification accuracy, ADM accuracy, CTP accuracy, and direct solve rate."""
        self.model.core.eval()
        self.model.adm.eval()
        self.model.ctp.eval()
    
        cls_correct, cls_total = 0, 0
        adm_correct, adm_total = 0, 0
        ctp_correct, ctp_total = 0, 0
        solved, total_sat = 0, 0
    
        with torch.no_grad():
            for data_batch in dataloader:
                data_batch = data_batch.to(self.device)
                outputs, L_h, C_h, _ = self.model.core(data_batch)
                vote = torch.sigmoid(outputs)
    
                target = data_batch.y
                sat_mask = target.squeeze().bool()
    
                # Classification
                pred_binary = (vote >= 0.5).float()
                cls_correct += (pred_binary == target).sum().item()
                cls_total += data_batch.num_graphs
    
                # ADM (literal prediction)
                n_literals = data_batch.n_vars * 2 # (batch size)
                lit_votes_tensor = torch.repeat_interleave(vote, n_literals, dim=0) # (batch size * n_lits)
                adm_input = torch.cat([L_h, lit_votes_tensor[:,None]], dim=1)
                adm_logits = self.model.adm(adm_input)
                sat_mask_lit = torch.repeat_interleave(target, data_batch.n_vars * 2, dim=0).squeeze().bool()
                if sat_mask_lit.sum() > 0:
                    predicted_lit = torch.argmax(adm_logits, dim=1)
                    adm_correct += (predicted_lit[sat_mask_lit] == data_batch.lit_label.squeeze()[sat_mask_lit]).sum().item()
                    adm_total += sat_mask_lit.sum().item()
    
                # CTP (clause prediction) 
                n_clauses = data_batch.n_clauses
                clause_votes_tensor = torch.repeat_interleave(vote, n_clauses, dim=0)
                ctp_input = torch.cat([C_h, clause_votes_tensor[:,None]], dim=1)
                ctp_logits = self.model.ctp(ctp_input)
                sat_mask_cls = torch.repeat_interleave(target, data_batch.n_clauses, dim=0).squeeze().bool()
                if sat_mask_cls.sum() > 0:
                    predicted_cls = torch.argmax(ctp_logits, dim=1)
                    ctp_correct += (predicted_cls[sat_mask_cls] == data_batch.clause_label.squeeze()[sat_mask_cls]).sum().item()
                    ctp_total += sat_mask_cls.sum().item()
    
                # Direct Solve (only for SAT instances, using ADM predictions)
                # not the most efficient way to do this, but we only run it at training, so it's fine for now. We can optimize later if needed.
                if sat_mask.sum() > 0:
                    probs = torch.softmax(adm_logits, dim=1)[:, 1]
    
                    consistent_preds = torch.zeros_like(probs)
                    offset = 0
                    for n_v in data_batch.n_vars:
                        n_lits = n_v * 2
                        pos_half = probs[offset : offset + n_v]
                        neg_half = probs[offset + n_v : offset + n_lits]
                        var_is_true = (pos_half > neg_half).float()
                        consistent_preds[offset : offset + n_v] = var_is_true
                        consistent_preds[offset + n_v : offset + n_lits] = 1.0 - var_is_true
                        offset += n_lits
    
                    edge_index = data_batch['literal', 'to', 'clause'].edge_index
                    clause_sat = scatter(consistent_preds[edge_index[0]], edge_index[1], dim=0, dim_size=data_batch.n_clauses.sum(), reduce='max')
                    clause_batch_idx = torch.repeat_interleave(
                        torch.arange(data_batch.num_graphs, device=self.device),
                        data_batch.n_clauses
                    )
                    instance_solved = scatter(clause_sat, clause_batch_idx, dim=0, reduce='min')
                    solved += instance_solved[sat_mask].sum().item()
                    total_sat += sat_mask.sum().item()
    
        return {
            "class_acc":  cls_correct / cls_total          if cls_total  > 0 else 0,
            "adm_acc":    adm_correct / adm_total          if adm_total  > 0 else 0,
            "ctp_acc":    ctp_correct / ctp_total          if ctp_total  > 0 else 0,
            "solve_rate": solved      / total_sat          if total_sat  > 0 else 0,
        }
    
    def test(self, sat="40_SAT", unsat="40_UNSAT"):
        test_loader = testing_data_setup(sat, unsat, generate_labels=True)
        return self.evaluate(test_loader)

    # Inference used for analysis and is optimised
    def inference(self, sat="40_SAT", unsat="40_UNSAT"):
        """does not include solving step, just returns raw outputs for analysis"""
        dataloader = testing_data_setup(sat, unsat)
        self.model.core.eval()
        self.model.adm.eval()
        self.model.ctp.eval()
    
        all_graph_votes = []
        all_lit_embs = []
        all_clause_embs = []
        all_ctp_probs = []
        all_adm_probs = []

        total_samples = 0
        total_start = time.perf_counter()

        with torch.no_grad():
            for data_batch in dataloader:
                data_batch = data_batch.to(self.device)
                
                graph_logits, L_h, C_h, _ = self.model.core(data_batch)
                graph_votes = torch.sigmoid(graph_logits)

                n_literals = data_batch.n_vars * 2 # (batch size)
                lit_votes_tensor = torch.repeat_interleave(graph_votes, n_literals, dim=0) # (batch size * n_lits)
                adm_input = torch.cat([L_h, lit_votes_tensor[:,None]], dim=1)
                adm_logits = self.model.adm(adm_input)
                adm_probs = torch.softmax(adm_logits, dim=1)
                
                n_clauses = data_batch.n_clauses
                clause_votes_tensor = torch.repeat_interleave(graph_votes, n_clauses, dim=0)
                ctp_input = torch.cat([C_h, clause_votes_tensor[:,None]], dim=1)
                ctp_logits = self.model.ctp(ctp_input)
                ctp_probs = torch.softmax(ctp_logits, dim=1)
                
                all_graph_votes.append(graph_votes.cpu())
                all_lit_embs.append(L_h.cpu())
                all_clause_embs.append(C_h.cpu())
                all_adm_probs.append(adm_probs.cpu())
                all_ctp_probs.append(ctp_probs.cpu())

                total_samples += data_batch.num_graphs

        total_time = (time.perf_counter() - total_start) * 1000
        latency_ms = total_time / max(1, total_samples)
        
        return {
            "graph_votes":  torch.cat(all_graph_votes),
            "lit_embs":     torch.cat(all_lit_embs),
            "clause_embs":  torch.cat(all_clause_embs),
            "adm_probs":    torch.cat(all_adm_probs),
            "ctp_probs":    torch.cat(all_ctp_probs),
            "latency":      latency_ms
        }


    # Training 
    def train_epoch(self, dataloader, epoch, stage=1):
        """Executes a single training epoch based on the specified curriculum stage."""
        # Mode Management
        if stage == 1:
            self.model.core.train(); self.model.adm.eval(); self.model.ctp.eval()
        elif stage == 2:
            self.model.core.eval(); self.model.adm.train(); self.model.ctp.train()
        else:
            self.model.train() # Unfreezes all sub-modules
            
        if stage == 3: 
            self.uc_loss_fn.train()
        else:
            self.uc_loss_fn.eval()
            
        total_loss = 0
        total_samples = 0 
        
        # Training Loop
        for i, batch_data in enumerate(dataloader):
            current_batch_size = batch_data.num_graphs
            batch_data = batch_data.to(self.device)

            # Core Forward Pass
            outputs, L_h, C_h, _ = self.model.core(batch_data)
            vote = torch.sigmoid(outputs)
            loss_global = self.global_loss_fn(outputs, batch_data.y) 

            # STAGE 1: Train Core Only 
            if stage == 1:
                self.optimizer.zero_grad()
                loss_global.backward()
                torch.nn.utils.clip_grad_norm_(self.model.core.parameters(), 0.65)
                self.optimizer.step()

                total_loss += loss_global.item() * current_batch_size
                total_samples += current_batch_size
                if (i + 1) % 10 == 0: 
                    print(f"--- Epoch {epoch + 1} | Batch {i + 1}/{len(dataloader)} | Stage 1 Loss: {loss_global.item():.6f}")
                continue

            # Generate MLP Head Inputs (d + 1)
            n_literals = batch_data.n_vars * 2 # (batch size)
            lit_votes_tensor = torch.repeat_interleave(vote, n_literals, dim=0) # (batch size * n_lits)
            adm_input = torch.cat([L_h, lit_votes_tensor[:,None]], dim=1)
            adm_logits = self.model.adm(adm_input)
            
            n_clauses = batch_data.n_clauses
            clause_votes_tensor = torch.repeat_interleave(vote, n_clauses, dim=0)
            ctp_input = torch.cat([C_h, clause_votes_tensor[:,None]], dim=1)
            ctp_logits = self.model.ctp(ctp_input)       
            
            # Generate Masks on SAT instance only
            sat_mask_lit = torch.repeat_interleave(batch_data.y, n_literals, dim=0).squeeze()
            sat_mask_cls = torch.repeat_interleave(batch_data.y, n_clauses, dim=0).squeeze()
            num_sat_lits = sat_mask_lit.sum()
            num_sat_clauses = sat_mask_cls.sum()

            # STAGE 4: refinement Loss
            if stage == 4:
                raw_loss_refine, entropy_reg = compute_clause_unsat_loss(
                    adm_logits, batch_data['literal', 'to', 'clause'].edge_index, batch_data.n_vars, n_clauses, device=self.device)
                loss_refine = (raw_loss_refine * sat_mask_cls).sum() / (num_sat_clauses + 1e-8) + entropy_reg
        
                loss = 0.1 * loss_global + 0.9 * loss_refine
                params_to_clip = self.model.parameters()

                self.optimizer.zero_grad()
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(params_to_clip, 0.65)
                self.optimizer.step()
                
                total_loss += loss.item() * current_batch_size 
                total_samples += current_batch_size
                if (i + 1) % 10 == 0: 
                    print(f"--- Epoch {epoch + 1} | Batch {i + 1}/{len(dataloader)} | Stage 4 Loss: {loss.item():.6f}")
                continue

            # STAGES 2, 3: Standard Supervised Loss
            raw_loss_lit = self.mlp_loss_fn(adm_logits, batch_data.lit_label.squeeze())
            raw_loss_cls = self.mlp_loss_fn(ctp_logits, batch_data.clause_label.squeeze())

            loss_lit = (raw_loss_lit * sat_mask_lit).sum() / (num_sat_lits + 1e-8)
            loss_clause = (raw_loss_cls * sat_mask_cls).sum() / (num_sat_clauses + 1e-8)

            if stage == 2:
                loss = (0.5 * loss_lit) + (0.5 * loss_clause)
                params_to_clip = list(self.model.adm.parameters()) + list(self.model.ctp.parameters())
            elif stage == 3:
                loss = self.uc_loss_fn([loss_global, loss_lit, loss_clause])
                params_to_clip = list(self.model.parameters()) + list(self.uc_loss_fn.parameters())        

            self.optimizer.zero_grad()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(params_to_clip, 0.65)
            self.optimizer.step()
            
            total_loss += loss.item() * current_batch_size 
            total_samples += current_batch_size

            if (i + 1) % 10 == 0: 
                print(f"--- Epoch {epoch + 1} | Batch {i + 1}/{len(dataloader)} | Stage {stage} Loss: {loss.item():.6f}")

        return total_loss / total_samples



