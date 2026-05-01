import torch
import os
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from data_loader import SATDataset
from Neurosat import NeuroSAT
from utils.utils import get_load_path

# Optimised inference function that abstract the entire process of loading a trained model, running inference on the test set, and returning clean, split embeddings for downstream analysis (no assignment decoding as doing so will move the computation to CPU).

LOAD_PATH = get_load_path()

def NN_inference(model_name:str, test_data='Test_40', T_val=26):
    # 1. Initialize Options
    opts = {
        'd_model': 64,
        'T': T_val,
        'learning_rate': 3.27e-05,
        'seed': 42,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    BATCH_SIZE = 128

    # 2. Setup Solver and Load Weights
    print(f"Initializing NeuroSAT on {opts['device']}...")
    solver = NeuroSAT(opts)

    print(f"Loading weights from {model_name}...")
    load_model_path = os.path.join(LOAD_PATH, f"{model_name}")
    solver.restore(load_model_path)

    # 3. Setup DataLoader
    print("Loading inference datasets...")
    test_sat_data = SATDataset(data_file=f"{test_data}_SAT", is_training=False, fixed_label=1)
    test_unsat_data = SATDataset(data_file=f"{test_data}_UNSAT", is_training=False, fixed_label=0)
    
    merged_test_data = ConcatDataset([test_sat_data, test_unsat_data])
    test_loader = DataLoader(merged_test_data, batch_size=BATCH_SIZE, shuffle=False) # type: ignore

    # 4. Run Inference via the Class (Returns massive continuous tensors)
    print("Extracting embeddings...")
    votes, lit_emb, clause_emb, var_votes, latency = solver.inference(test_loader)
    print(f"Extraction complete! Latency: {latency:.4f} ms/sample")
    
    # 5. Un-batch the massive tensors into lists of Numpy arrays for the decoder
    print("Splitting embeddings into per-graph arrays...")
    
    # Get the exact number of literals and clauses for every graph in the dataset
    n_lits_per_graph = [data.n_vars.item() * 2 for data in merged_test_data] # type: ignore
    n_clauses_per_graph = [data['clause'].num_nodes for data in merged_test_data] # type: ignore
    
    # Split the massive tensors instantly
    split_lit_emb = [emb.numpy() for emb in torch.split(lit_emb, n_lits_per_graph)]
    split_clause_emb = [emb.numpy() for emb in torch.split(clause_emb, n_clauses_per_graph)]
    
    n_vars_per_graph = [data.n_vars.item() for data in merged_test_data] # type: ignore
    split_var_votes = [vote.numpy() for vote in torch.split(var_votes, n_vars_per_graph)]

    # Return the clean, split lists
    return votes.numpy(), split_lit_emb, split_clause_emb, split_var_votes, latency