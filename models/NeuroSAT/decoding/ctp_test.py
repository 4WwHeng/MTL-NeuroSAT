import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report
from core.data_loader import SATDataset
from sat_lit import decode_kmeans_dist
from utils.utils import get_close_assignment
from sat_cls import clause_satlit_count
from core.Neurosat import NeuroSAT
from ctp_training import ClauseTierPredictor
from data.data_preprocessing import read_data
from utils.utils import get_load_path

# test the supervised clause brittleness predictor on the test set, and report the classification report for the clause tier labels (0, 1, 2) to see how well it distinguishes between brittle vs medium vs slack clauses. 

LOAD_PATH = get_load_path()
LOAD_PATH_CORE = os.path.join(LOAD_PATH, "M-Trial4-T26-D64-L3.27e-05_epoch127_BEST.pth")
LOAD_PATH_ADM = os.path.join(LOAD_PATH, "AssignmentDecodingModel-D207-Dr0.11160159495261959-L0.0008748982734118006_best.pth")
LOAD_PATH_CTP = os.path.join(LOAD_PATH, "ClauseTierPredictor-D220-Dr0.30486349258950407-L0.0010006590443426412_best.pth")


# Optimized Batched Inference
def ctp_inference(test_data_sat, test_data_unsat, device):
    # Load NeuroSAT
    opts = {'d_model': 64, 'T': 26, 'device': device}
    solver = NeuroSAT(opts)
    solver.restore(LOAD_PATH_CORE)

    # Load CTP
    ctp = ClauseTierPredictor(dimension=220, dropout=0.3).to(device)
    ctp.load_state_dict(torch.load(LOAD_PATH_CTP)['model_state_dict'])
    ctp.eval()

    # Setup Data
    sat_data = SATDataset(data_file=test_data_sat, is_training=False, fixed_label=1)
    unsat_data = SATDataset(data_file=test_data_unsat, is_training=False, fixed_label=0)
    merged_data = ConcatDataset([sat_data, unsat_data]) 
    dataloader = DataLoader(merged_data, batch_size=128, shuffle=False)

    all_graph_votes = []
    all_lit_embs = []
    all_clause_embs = []
    all_ctp_probs = []

    print("Extracting embeddings and running CTP predictions...")
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # 1. NeuroSAT
            graph_logits, L_h, C_h, _ = solver.model(batch, return_embeddings=True)
            graph_votes = torch.sigmoid(graph_logits)
            
            # 2. CTP 
            # Create a list of how many clauses belong to each graph
            n_clauses_per_graph = batch.n_clauses.tolist()
            
            # Expand the single graph vote to match every clause in that graph
            expanded_votes = torch.repeat_interleave(graph_votes, torch.tensor(n_clauses_per_graph, device=device))
            expanded_votes = expanded_votes.unsqueeze(1) # Shape: [Total_Clauses, 1]
            
            # Concat the clause embeddings with their corresponding expanded votes
            mlp_input = torch.cat([C_h, expanded_votes], dim=1) # Shape: [Total_Clauses, 65]
            
            # Run the CTP on the entire batch instantly
            ctp_logits = ctp(mlp_input)
            ctp_probs = torch.softmax(ctp_logits, dim=1)
            
            # 3. Store Outputs
            all_graph_votes.extend(graph_votes.cpu().numpy())
            
            # Split the massive tensors and convert to numpy
            n_lits_per_graph = (batch.n_vars * 2).tolist()
            all_lit_embs.extend([emb.numpy() for emb in torch.split(L_h.cpu(), n_lits_per_graph)])
            all_clause_embs.extend([emb.numpy() for emb in torch.split(C_h.cpu(), n_clauses_per_graph)])
            all_ctp_probs.extend([prob.numpy() for prob in torch.split(ctp_probs.cpu(), n_clauses_per_graph)])
    
    return np.array(all_graph_votes), all_lit_embs, all_clause_embs, all_ctp_probs


# Evaluation & Scoring
def test_batch_ctp_model(test_data_sat="Test_40_SAT", test_data_unsat="Test_40_UNSAT"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run ctp inference
    votes, lit_emb, clause_emb, ctp_probs = ctp_inference(test_data_sat, test_data_unsat, device)
    
    # Load raw data for evaluation
    data_sat = read_data(test_data_sat, is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    all_true_tiers = []
    all_pred_tiers = []
    
    print("\n--- Starting CTP Evaluation ---")
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if not is_sat_ground_truth or votes[i] < 0.5:
            continue

        L_h = lit_emb[i]
        tier_prob = ctp_probs[i]

        k_means_candidate, var_dist, direct_solved = decode_kmeans_dist(L_h, clauses, n_vars)
        
        if not direct_solved:
            truth_assignment = get_close_assignment(k_means_candidate, clauses, n_vars)
        else:
            truth_assignment = k_means_candidate
            
        if truth_assignment is None:
            continue
            
        # ground truth labels
        sat_lit_count = clause_satlit_count(clauses, truth_assignment)
        tiers = np.where(sat_lit_count == 1, 0, np.where(sat_lit_count == 2, 1, 2))

        # predicted labels
        p_tiers = np.argmax(tier_prob, axis=1)

        all_true_tiers.extend(tiers)
        all_pred_tiers.extend(p_tiers)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} ground-truth SAT problems...")

    # Print Final Report
    print("\n" + "="*50)
    print(" CTP TIER PREDICTION REPORT")
    print("="*50)
    
    if len(all_true_tiers) > 0:
        target_names = ['Tier 0 (Brittle)', 'Tier 1 (Medium)', 'Tier 2 (Slack)']
        print(classification_report(all_true_tiers, all_pred_tiers, target_names=target_names, zero_division=0))
    else:
        print("No valid instances were processed to generate a report.")


if __name__ == "__main__":
    test_batch_ctp_model()