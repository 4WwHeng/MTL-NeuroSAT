import os
import torch
from utils.utils import get_log_path, get_batch_size
from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from data.data_preprocessing import read_data, build_sparse_edges

# class for building disconneected graph dataset for NeuroSAT, required for batching of variable sized inputs.

CACHE_PATH = get_log_path()
BATCH_SIZE = get_batch_size()

class SATDataset(Dataset): 
    """
    A dataset class that handles both training and testing data, processing graphs into HeteroData objects and caching them to disk.
    """
    def __init__(self, data_file, is_training=True, fixed_label=None, limit=float('inf'), generate_labels=False):
        super().__init__()
        self.data_file = data_file
        self.is_training = is_training
        self.generate_labels = generate_labels
        
        # Name the cache file uniquely based on the data type
        prefix = "train" if is_training else f"test_{fixed_label}"
        label_tag = "_labels" if generate_labels else ""
        self.cache_file_path = os.path.join(CACHE_PATH, f"cached_{prefix}_{data_file}{label_tag}.pt")
        
        if os.path.exists(self.cache_file_path):
            print(f"Loading cached data from {self.cache_file_path}...")
            self.data_list = torch.load(self.cache_file_path, weights_only=False)
        else:
            print(f"Cache file not found. Reading and converting raw data...")
            raw_problems = read_data(data_file, is_training=is_training, fixed_label=fixed_label, limit=limit, generate_labels=generate_labels)
            if not generate_labels:   
                self.data_list = self._process(raw_problems)
            else:
                self.data_list = self._process_labels(raw_problems)
                
        print(f"Loaded {len(self.data_list)} samples total.")

    def _process(self, raw_problems):
        processed_list = []

        for clauses, n_var, label in raw_problems:
            n_clauses = len(clauses)
            n_literals = n_var * 2
    
            # Get edges directly 
            edge_index, edge_attr = build_sparse_edges(clauses, n_var)
    
            # Build PyTorch Geometric Heterogeneous graph
            data = HeteroData()
            data['literal'].num_nodes = n_literals
            data['clause'].num_nodes = n_clauses
            data['literal', 'to', 'clause'].edge_index = edge_index
            data['literal', 'to', 'clause'].edge_attr = edge_attr
    
            data.y = torch.tensor([label], dtype=torch.float)
            data.n_vars = torch.tensor([n_var], dtype=torch.long)
            data.n_clauses = torch.tensor([n_clauses], dtype=torch.long)

            processed_list.append(data)

        print(f"Saving processed data to {self.cache_file_path}...")
        torch.save(processed_list, self.cache_file_path)

        return processed_list

    # used when generate_labels is True (for MTL)
    def _process_labels(self, raw_problems):
        processed_list = []

        for instance in raw_problems:
            clauses = instance['clauses']
            n_var = instance['n_var']
            label = instance['is_sat']
            lit_label = instance['lit_labels']
            clause_label = instance['clause_labels']

            n_clauses = len(clauses)
            n_literals = n_var * 2
    
            # Get edges directly 
            edge_index, edge_attr = build_sparse_edges(clauses, n_var)
    
            # Build PyTorch Geometric Heterogeneous graph
            data = HeteroData()
            data['literal'].num_nodes = n_literals
            data['clause'].num_nodes = n_clauses
            data['literal', 'to', 'clause'].edge_index = edge_index
            data['literal', 'to', 'clause'].edge_attr = edge_attr
    
            data.y = torch.tensor([label], dtype=torch.float)
            data.n_vars = torch.tensor([n_var], dtype=torch.long)
            data.n_clauses = torch.tensor([n_clauses], dtype=torch.long)
            data.lit_label = lit_label
            data.clause_label = clause_label

            processed_list.append(data)

        print(f"Saving processed data to {self.cache_file_path}...")
        torch.save(processed_list, self.cache_file_path)

        return processed_list
            
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def training_data_setup(dataset, generate_labels=False):
    train_data = SATDataset(data_file=dataset, is_training=True, generate_labels=generate_labels)
    return DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True) #type: ignore


def testing_data_setup(sat, unsat, generate_labels=False):
    test_sat_data = SATDataset(data_file=sat, is_training=False, fixed_label=1, generate_labels=generate_labels)
    test_unsat_data = SATDataset(data_file=unsat, is_training=False, fixed_label=0, generate_labels=generate_labels)
    
    merged_test_data = ConcatDataset([test_sat_data, test_unsat_data])
    return DataLoader(merged_test_data, batch_size=BATCH_SIZE, shuffle=False) #type: ignore