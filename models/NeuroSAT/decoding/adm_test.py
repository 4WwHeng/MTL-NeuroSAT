import numpy as np
import torch
import os
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report
from adm_training import AssignmentDecodingModel
from data.data_preprocessing import read_data
from core.Neurosat import NeuroSAT
from core.data_loader import SATDataset
from utils.utils import count_satisfy, get_close_assignment
from utils.utils import get_load_path

# test the supervised assignment decoding model on the test set, and report how many problems it solved directly, and for the ones it failed, report the F1 classification report for the literal labels (true vs false) to see if it at least got some of them right.

LOAD_PATH = get_load_path()
LOAD_PATH_CORE = os.path.join(LOAD_PATH, "M-Trial4-T26-D64-L3.27e-05_epoch127_BEST.pth")
LOAD_PATH_ADM = os.path.join(LOAD_PATH, "AssignmentDecodingModel-D207-Dr0.11160159495261959-L0.0008748982734118006_best.pth")
LOAD_PATH_CTP = os.path.join(LOAD_PATH, "ClauseTierPredictor-D220-Dr0.30486349258950407-L0.0010006590443426412_best.pth")


# Supervised Assignment Decoding Test
def adm_inference(test_data_sat, test_data_unsat, device):
    # Load NeuroSAT
    opts = {'d_model': 64, 'T': 26, 'device': device}
    solver = NeuroSAT(opts)
    solver.restore(LOAD_PATH_CORE)
    
    # Load ADM
    adm = AssignmentDecodingModel(dimension=207, dropout=0.1).to(device)
    adm.load_state_dict(torch.load(LOAD_PATH_ADM)['model_state_dict'])
    adm.eval()

    # Setup Data
    sat_data = SATDataset(data_file=test_data_sat, is_training=False, fixed_label=1)
    unsat_data = SATDataset(data_file=test_data_unsat, is_training=False, fixed_label=0)
    merged_data = ConcatDataset([sat_data, unsat_data])
    dataloader = DataLoader(merged_data, batch_size=128, shuffle=False)

    all_graph_votes = []
    all_lit_embs = []
    all_adm_probs = []

    print("Extracting embeddings and running ADM predictions...")
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # 1. NeuroSAT
            graph_logits, L_h, _, _ = solver.model(batch, return_embeddings=True)
            graph_votes = torch.sigmoid(graph_logits)
            
            # 2. ADM 
            # Create a list of how many literals belong to each graph in the batch
            n_lits_per_graph = (batch.n_vars * 2).tolist()
            
            # Expand the single graph vote to match every literal in that graph
            expanded_votes = torch.repeat_interleave(graph_votes, torch.tensor(n_lits_per_graph, device=device))
            expanded_votes = expanded_votes.unsqueeze(1) # Shape: [Total_Lits, 1]
            
            # Concat the embeddings with their corresponding votes
            mlp_input = torch.cat([L_h, expanded_votes], dim=1) # Shape: [Total_Lits, 65]
            
            # Run the ADM on the entire batch instantly
            adm_logits = adm(mlp_input)
            adm_probs = torch.softmax(adm_logits, dim=1)
            
            # 3. Store Outputs
            all_graph_votes.extend(graph_votes.cpu().numpy())
            
            # Split the massive tensors and convert to numpy
            all_lit_embs.extend([emb.numpy() for emb in torch.split(L_h.cpu(), n_lits_per_graph)])
            all_adm_probs.extend([prob.numpy() for prob in torch.split(adm_probs.cpu(), n_lits_per_graph)])
    
    return np.array(all_graph_votes), all_lit_embs, all_adm_probs


# Evaluation & Scoring
def test_adm_pipeline(test_data_sat="Test_40_SAT", test_data_unsat="Test_40_UNSAT"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run adm inference
    votes, lit_embs, adm_probs = adm_inference(test_data_sat, test_data_unsat, device)
    
    # Load raw data for evaluation
    data_sat = read_data(test_data_sat, is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    adm_solved_count = 0
    pred_sat_count = 0
    
    pred_class = []
    true_class = []
    
    pred_labels = []
    true_labels = []

    print("\n--- Starting ADM Decoding ---")
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        vote = votes[i]
        
        # Track Graph-Level Satisfiability Prediction
        true_class.append(is_sat_ground_truth)
        pred_class.append(1 if vote >= 0.5 else 0)
            
        # We only decode assignments for ground-truth SAT problems
        if not is_sat_ground_truth:
            continue

        # and predicted SAT problems
        if vote >= 0.5:
            pred_sat_count += 1  
        else:
            continue

        # ADM predictions for this specific graph
        candidate_prob = adm_probs[i]

        # Compare positive literal truth probability vs negative literal truth probability
        prob_pos_true = candidate_prob[:n_vars, 1]
        prob_neg_true = candidate_prob[n_vars:, 1]
        candidate = (prob_pos_true > prob_neg_true).astype(int)
        
        # Verify assignment
        count, direct_solved = count_satisfy(candidate, clauses) 
        
        if direct_solved:
            adm_solved_count += 1
        else:
            # If it failed, save the labels for the F1 Classification Report
            assignment = get_close_assignment(candidate, clauses, n_vars)
            if assignment is None:
                print(f"Warning: WalkSAT failed to find a solution for instance {i}.")
                continue
            pred_labels.extend(candidate)
            true_labels.extend(assignment)

        if (i+1) % 100 == 0:
            print(f"Processed {i+1} SAT problems...")

    # Print Final Report
    print("\n" + "="*40)
    print(" ADM DECODING RESULTS REPORT")
    print("="*40)
    print(f"Solved by Supervised ADM: {adm_solved_count}/{pred_sat_count}")
    print("-" * 40)

    target_names = ['Label 0 (False)', 'Label 1 (True)']
    print("\nGraph Satisfiability Classification (NeuroSAT)")
    print(classification_report(true_class, pred_class, target_names=target_names))
    
    if true_labels:
        print("\nLiteral Assignment Classification (ADM on UNSOLVED)")
        print(classification_report(true_labels, pred_labels, target_names=target_names))


if __name__ == "__main__":
    test_adm_pipeline()