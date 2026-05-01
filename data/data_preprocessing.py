import shutil
import torch
import numpy as np
from pathlib import Path
from utils.utils import get_truth_assignment, clause_satlit_count, get_data_path, get_test_path

# Extracting Data and Building Appropriate Representations (dense adjacency matrices, or sparse edge indices)


# Parsing and Conversion

def dimacs_parser(filepath):
    """Lazily parses a DIMACS CNF file."""
    if not filepath.exists():
        print(f"Error: File not found at {filepath}")
        return 0, 0, []

    n_var, n_clause = 0, 0
    clauses = []
    
    with filepath.open('r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            
            parts = line.split()
            if parts[0] == "p":
                n_var = int(parts[2])
                n_clause = int(parts[3])
            elif line.endswith("0"):
                # Exclude the trailing '0'
                clauses.append([int(i) for i in parts[:-1]])
                
    return n_var, n_clause, clauses


def convert_matrix(clauses, n_var):
    """Builds a dense bipartite adjacency matrix. """
    n_clauses = len(clauses)
    adjacency_matrix = np.zeros((n_var * 2, n_clauses), dtype=np.float32)

    for j, clause in enumerate(clauses):
        for l in clause:
            if l > 0:
                adjacency_matrix[l - 1][j] = 1.0  
            else:
                adjacency_matrix[n_var + abs(l) - 1][j] = 1.0  
                
    return torch.tensor(adjacency_matrix)


def build_sparse_edges(clauses, n_var):
    """Builds PyTorch sparse edge indices directly."""
    source_nodes = []  
    target_nodes = []  

    for clause_idx, clause in enumerate(clauses):
        for literal in clause:
            # Positive: 0 to (n_var - 1). Negative: n_var to (2*n_var - 1)
            lit_idx = (literal - 1) if literal > 0 else (n_var + abs(literal) - 1)
            source_nodes.append(lit_idx)
            target_nodes.append(clause_idx)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.ones(len(source_nodes), dtype=torch.float32)

    return edge_index, edge_attr


# Dataset Loading and Operations

def read_data(folder_name, is_training=True, fixed_label=None, limit=float('inf'), generate_labels=False):
    """Unified function to read and process CNF datasets."""
    data_set = []  
    
    # Dynamically resolve base path based on execution mode
    base_dir = Path(get_data_path()) if is_training else Path(get_test_path())
    folder_path = base_dir / folder_name
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {folder_path}")

    filenames = sorted(f for f in folder_path.iterdir() if f.suffix == '.cnf')
    
    for count, filepath in enumerate(filenames):
        if count >= limit:
            break

        n_var, n_clauses, clauses = dimacs_parser(filepath)

        # 1. Determine Satisfiability
        if is_training:
            if '_SAT' in filepath.name:
                label = 1
            elif '_UNSAT' in filepath.name:
                label = 0
            else:
                continue 
        else:
            label = fixed_label 
            
        is_sat_ground_truth = (label == 1)

        # 2. Process Data Payload
        if not generate_labels:
            data_set.append((clauses, n_var, label))
        else:
            if is_sat_ground_truth:            
                assignment = np.array(get_truth_assignment(clauses))
                full_assignment = np.concatenate([assignment, 1 - assignment], axis=0) 
                
                clause_counts = np.array(clause_satlit_count(clauses, assignment))
                # Map counts: 1 -> 0, 2 -> 1, >2 -> 2
                clause_labels = np.where(clause_counts == 1, 0, np.where(clause_counts == 2, 1, 2))
                
                data_set.append({
                    'clauses': clauses,
                    'n_var': n_var,
                    'is_sat': 1,
                    'lit_labels': torch.tensor(full_assignment, dtype=torch.long),
                    'clause_labels': torch.tensor(clause_labels, dtype=torch.long)
                })
            else:
                data_set.append({
                    'clauses': clauses,
                    'n_var': n_var,
                    'is_sat': 0,
                    'lit_labels': torch.zeros(n_var * 2, dtype=torch.long),
                    'clause_labels': torch.zeros(n_clauses, dtype=torch.long)
                })

    return data_set


def split_training_data(folder_name):
    """Splits an integrated folder into distinct SAT and UNSAT subdirectories."""
    source_folder = Path(get_data_path()) / folder_name
    target_sat = source_folder.parent / f"{folder_name}_SAT"
    target_unsat = source_folder.parent / f"{folder_name}_UNSAT"
    
    target_sat.mkdir(exist_ok=True)
    target_unsat.mkdir(exist_ok=True)
    
    filenames = sorted(f for f in source_folder.iterdir() if f.suffix == '.cnf')

    for filepath in filenames:
        if '_SAT' in filepath.name:
            shutil.copy(filepath, target_sat / filepath.name)
        elif '_UNSAT' in filepath.name:
            shutil.copy(filepath, target_unsat / filepath.name)
    
    return folder_name
