import torch
import torch.nn as nn
import os
from torch.utils.data import ConcatDataset
from core.Neurosat import NeuroSATNetwork
from MTL.mtl_trainer import MtlTrainer
from core.data_loader import SATDataset
from utils.utils import get_load_path

# Supervised decoding NeuroSAT inference with separate ADM and CTP heads, using the MTL architecture for maximum modularity and extensibility. The two MLP heads were loaded separately, but inference is done using MTL which assumes a unified training framework.

LOAD_PATH = get_load_path()
LOAD_PATH_CORE = os.path.join(LOAD_PATH, "M-Trial4-T26-D64-L3.27e-05_epoch127_BEST.pth")
LOAD_PATH_ADM = os.path.join(LOAD_PATH, "AssignmentDecodingModel-D207-Dr0.11160159495261959-L0.0008748982734118006_best.pth")
LOAD_PATH_CTP = os.path.join(LOAD_PATH, "ClauseTierPredictor-D220-Dr0.30486349258950407-L0.0010006590443426412_best.pth")

class SepMlpHeadModel(nn.Module):
    def __init__(self, in_dim=65, hid_dim=128, out_dim=2):
        super(SepMlpHeadModel, self).__init__()
        # 64 embeddings + 1 Vote = 65 inputs
        self.norm = nn.LayerNorm(65)
        
        self.network = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents overfitting
            nn.Linear(hid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim) # Output: [False, True]
        )
        
    def forward(self, x):
        x = self.norm(x)
        return self.network(x)
        

class SepMtlNeuroSAT(nn.Module):
    def __init__(self, d_model=64, T=26):
        super(SepMtlNeuroSAT, self).__init__()
        self.core = NeuroSATNetwork(d_model=d_model, T=T)
        self.adm = SepMlpHeadModel(in_dim=d_model+1, hid_dim=207, out_dim=2)
        self.ctp = SepMlpHeadModel(in_dim=d_model+1, hid_dim=220, out_dim=3)
        
    def forward(self, batch_data):
        """Unified forward pass for all three networks."""
        outputs, L_h, C_h, var_votes = self.core(batch_data)
        adm_logits = self.adm(L_h)
        ctp_logits = self.ctp(C_h)
        
        return outputs, adm_logits, ctp_logits
    

def NNs_inference(test_data_sat='Test_40_SAT', test_data_unsat='Test_40_UNSAT', T_val=26):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing inference on {device}...")

    # 1. Create the blank model and the trainer
    model = SepMtlNeuroSAT(d_model=64, T=T_val)
    trainer = MtlTrainer(model, device)

    # 2. Use the built-in restore method
    epoch_trained = trainer.restore_seperate(LOAD_PATH_CORE, LOAD_PATH_ADM, LOAD_PATH_CTP)
    
    if epoch_trained is None:
        print("Inference aborted: Checkpoint not found.")
        raise Exception("Checkpoint not found")

    print(f"Model successfully loaded! (Trained for {epoch_trained} epochs)")
    print(f"Evaluating on {test_data_sat} and {test_data_unsat}...\n")
    test_sat_data = SATDataset(data_file=test_data_sat, is_training=False, fixed_label=1)
    test_unsat_data = SATDataset(data_file=test_data_unsat, is_training=False, fixed_label=0)
    
    merged_test_data = ConcatDataset([test_sat_data, test_unsat_data])

    # 3. Run the inference and get the results
    result = trainer.inference(test_data_sat, test_data_unsat)
    print(f"Extraction complete! Latency: {result['latency']:.4f} ms/sample")

    # Get the exact number of literals and clauses for every graph in the dataset
    n_lits_per_graph = [data.n_vars.item() * 2 for data in merged_test_data] #type: ignore
    n_clauses_per_graph = [data.n_clauses.item() for data in merged_test_data] #type: ignore
    n_vars_per_graph = [data.n_vars.item() for data in merged_test_data] #type: ignore

    # Split the massive tensors instantly
    lit_embs    = result['lit_embs']
    clause_embs = result['clause_embs']
    adm_probs   = result['adm_probs']
    ctp_probs   = result['ctp_probs']
    var_votes   = result['var_votes']
    split_lit_emb    = [e.numpy() for e in torch.split(lit_embs,    n_lits_per_graph)]
    split_clause_emb = [e.numpy() for e in torch.split(clause_embs, n_clauses_per_graph)]
    split_adm_prob   = [p.numpy() for p in torch.split(adm_probs,   n_lits_per_graph)]
    split_ctp_prob   = [p.numpy() for p in torch.split(ctp_probs,   n_clauses_per_graph)]
    split_var_votes  = [vote.numpy() for vote in torch.split(var_votes, n_vars_per_graph)]

    # 4. Return the split results for downstream analysis
    return result["graph_votes"], split_lit_emb, split_clause_emb, split_adm_prob, split_ctp_prob, split_var_votes, result["latency"]