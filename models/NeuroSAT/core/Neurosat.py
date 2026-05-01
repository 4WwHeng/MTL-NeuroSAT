import os
import time
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import scatter

# NeuroSAT implementation based on the original paper.

# Helper Modules 
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)


def flip_index(L, n_literals_list):
    device = L.device
    
    # 1. Get the number of variables (nV) for each graph in the batch
    n_vars = n_literals_list // 2 # [nV1, nV2, ...]
    
    # 2. Create the shift values: [[nV1, -nV1], [nV2, -nV2], ...]
    shift_vals = torch.stack([n_vars, -n_vars], dim=1).view(-1)
    
    # 3. Create the repeat counts: [[nV1, nV1], [nV2, nV2], ...]
    repeat_vals = torch.stack([n_vars, n_vars], dim=1).view(-1)
    
    # 4. Interleave to get the exact shift for every single literal: [[nV1 * nV1 times, -nV1 * nV1 times], [nV2 * nV2 times, -nV2 * nV2 times], ...]
    shifts = torch.repeat_interleave(shift_vals, repeat_vals)
    
    # 5. Add the shifts to a standard arange index: [0, 1, 2, ..., n_literals-1] + shifts 
    # Every postive literal index will get a +nV shift, every negative literal index will get a -nV shift, effectively flipping them within each graph.
    flip_idx = torch.arange(L.size(0), device=device) + shifts
    
    return flip_idx

# The Core Network
class NeuroSATNetwork(nn.Module):
    def __init__(self, d_model=64, T=26):
        super().__init__()
        self.d = d_model
        self.T = T

        self.L_init = nn.Parameter(torch.randn(1, d_model))
        self.C_init = nn.Parameter(torch.randn(1, d_model))

        self.L_msg = MLP(d_model, d_model, d_model)
        self.C_msg = MLP(d_model, d_model, d_model)

        self.L_update = nn.LSTMCell(3 * d_model, d_model)
        self.C_update = nn.LSTMCell(2 * d_model, d_model)

        self.L_norm = nn.LayerNorm(d_model)
        self.C_norm = nn.LayerNorm(d_model)
        
        self.L_vote = MLP(d_model, d_model, 1)

    def forward(self, data, return_embeddings=False):
        edge_index = data['literal', 'to', 'clause'].edge_index
        n_literals = data['literal'].num_nodes 
        n_clauses = data['clause'].num_nodes 
        n_graphs = data.num_graphs
        
        n_literals_list = torch.bincount(data['literal'].batch)      

        L_h = self.L_init.repeat(n_literals, 1) / (self.d ** 0.5)
        C_h = self.C_init.repeat(n_clauses, 1) / (self.d ** 0.5)
        L_c = torch.zeros_like(L_h)
        C_c = torch.zeros_like(C_h)

        # Pre-compute Flip Index Once
        flip_idx = flip_index(L_h, n_literals_list)

        edge_index = edge_index.to(L_h.device)

        for t in range(self.T):
            L_pre = self.L_msg(L_h)  
            L_msgs = L_pre[edge_index[0]]  
            C_aggr = scatter(L_msgs, edge_index[1], dim=0, dim_size=n_clauses, reduce='sum') 
            
            C_input = torch.cat([C_h, C_aggr], dim=1)
            C_h, C_c = self.C_update(C_input, (C_h, C_c))
            C_h = self.C_norm(C_h)

            C_pre = self.C_msg(C_h)
            C_msgs = C_pre[edge_index[1]]
            L_aggr = scatter(C_msgs, edge_index[0], dim=0, dim_size=n_literals, reduce='sum')
            
            L_flipped = L_h[flip_idx]
            L_input = torch.cat([L_h, L_flipped, L_aggr], dim=1)
            L_h, L_c = self.L_update(L_input, (L_h, L_c))
            L_h = self.L_norm(L_h)

        L_votes = self.L_vote(L_h).squeeze(-1) 
        batch_idx = data['literal'].batch
        graph_logits = scatter(L_votes, batch_idx, dim=0, dim_size=n_graphs, reduce='mean')

        if return_embeddings:
            # 1. Use the flip_idx we pre-computed to align positive and negative votes instantly
            L_votes_flipped = L_votes[flip_idx]
            
            # 2. Average the positive and negative literal votes
            averaged_literal_votes = (L_votes + L_votes_flipped) / 2
            
            # 3. Create a mask to grab only the first half of each graph's literals (the positive ones) 
            n_vars_tensor = n_literals_list // 2
            # Create an alternating pattern: [True, False, True, False...]
            bool_pattern = torch.tensor([True, False], device=L_h.device).repeat(n_graphs)
            # Create the repeat counts: [V1, V1, V2, V2...]
            repeat_counts = torch.stack([n_vars_tensor, n_vars_tensor], dim=1).view(-1)
            # Interleave to generate the exact mask instantly
            pos_mask = torch.repeat_interleave(bool_pattern, repeat_counts)
            
            var_votes = averaged_literal_votes[pos_mask]

            # Return: Graph Votes, Literal Embeddings, Clause Embeddings, Variable Votes
            return graph_logits, L_h, C_h, var_votes
            
        return graph_logits

        
# Orchestrator Class 
class NeuroSAT(object):
    def __init__(self, opts):
        """opts: a dictionary containing hyperparameters."""
        self.opts = opts
        self.device = opts.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.init_random_seeds()
        self.build_network()

    def init_random_seeds(self):
        seed = self.opts.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def build_network(self):
        self.model = NeuroSATNetwork(
            d_model=self.opts.get('d_model', 64), 
            T=self.opts.get('T', 26)
        ).to(self.device)
        
        self.loss_fn = nn.BCEWithLogitsLoss() 
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.opts.get('learning_rate', 2e-5)
        )

    def save(self, path, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, path)

    def restore(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            print(f"--- Checkpoint loaded. Epoch {checkpoint['epoch'] + 1} ---")
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        epoch_start = time.time()

        for i, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(self.device)
            target = batch_data.y.float().view(-1)
            
            self.optimizer.zero_grad()
            logits = self.model(batch_data).view(-1)
            
            loss = self.loss_fn(logits, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.get('clip_val', 0.65))
            self.optimizer.step()

            total_loss += loss.item() * batch_data.num_graphs
            total_samples += batch_data.num_graphs

        epoch_end = time.time()
        avg_loss = total_loss / total_samples
        return avg_loss, epoch_end - epoch_start

    def test(self, dataloader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = batch_data.to(self.device)
                target = batch_data.y.float().view(-1)
                
                logits = self.model(batch_data).view(-1)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

                total_correct += (preds == target).sum().item()
                total_samples += batch_data.num_graphs

        accuracy = total_correct / total_samples
        return accuracy, confusion_matrix(all_labels, all_preds)

    def check_satisfaction(self, edge_index, num_clauses, assignment):
        """check if an instances is satisfied by the assignment"""
        if not isinstance(assignment, torch.Tensor):
            assignment = torch.tensor(assignment, device=edge_index.device)
            
        flipped = 1 - assignment
        full_assignment = torch.cat([assignment, flipped], dim=0)
        
        true_literals_mask = full_assignment[edge_index[0]] == 1
        satisfied_clauses = torch.unique(edge_index[1][true_literals_mask])
        
        return len(satisfied_clauses) == num_clauses, len(satisfied_clauses)

    def find_solutions(self, batch_data):
        """
        K-Means Clustering to extract assignments from the embeddings.
        Must be done in CPU as K-Means implementation of scikit-learn does not accept inputs on GPU.
        Vectorized operations are not prioritized
        """
        self.model.eval()
        batch_data = batch_data.to(self.device)
        solutions = []

        with torch.no_grad():
            # Get logits, node embeddings, and node votes
            graph_logits, L_h, C_h, var_votes = self.model(batch_data, return_embeddings=True)
            
            # Split by graph in the batch
            n_lits_list = [(batch_data['literal'].batch == i).sum().item() for i in range(batch_data.num_graphs)]
            vars_per_graph = batch_data.n_vars.tolist()
            
            L_h_split = torch.split(L_h, n_lits_list)

            for i in range(batch_data.num_graphs):
                n_vars_i = vars_per_graph[i]
                num_clauses_i = (batch_data['clause'].batch == i).sum().item()
                
                # Isolate the graph's specific edge_index
                l_mask = (batch_data['literal'].batch == i)
                c_mask = (batch_data['clause'].batch == i)
                edge_index = batch_data['literal', 'to', 'clause'].edge_index
                my_edges = edge_index[:, l_mask[edge_index[0]]].clone()
                my_edges[0] -= l_mask.nonzero()[0].min().item()
                my_edges[1] -= c_mask.nonzero()[0].min().item()

                # K-Means Clustering
                embeddings_i = L_h_split[i].cpu().numpy()
                pos_emb = embeddings_i[:n_vars_i, :]
                neg_emb = embeddings_i[n_vars_i:, :]
                
                # Stack to [2 * n_vars, d_model] and cluster
                L_concat = np.concatenate([pos_emb, neg_emb], axis=0)
                kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(L_concat)
                c1, c2 = kmeans.cluster_centers_

                # Calculate distances to centers
                d1 = np.sum((pos_emb - c1)**2, axis=1) + np.sum((neg_emb - c2)**2, axis=1)
                d2 = np.sum((pos_emb - c2)**2, axis=1) + np.sum((neg_emb - c1)**2, axis=1)
                
                # Generate Candidate A and its logical inverse, Candidate B
                decode_kmeans_A = np.where(d1 > d2, 1, 0).astype(float)
                decode_kmeans_B = 1 - decode_kmeans_A

                # Test Candidate A and Candidate B, and return the best solution
                satA, c_satA = self.check_satisfaction(my_edges, num_clauses_i, decode_kmeans_A)
                satB, c_satB = self.check_satisfaction(my_edges, num_clauses_i, decode_kmeans_B)
                if satA:
                    solutions.append({"assignment": decode_kmeans_A, "satisfied": True})
                elif satB:
                    solutions.append({"assignment": decode_kmeans_B, "satisfied": True})
                elif c_satA > c_satB:
                    solutions.append({"assignment": decode_kmeans_A, "satisfied": False}) # Return the best candidate
                else:
                    solutions.append({"assignment": decode_kmeans_B, "satisfied": False})

        return solutions

    def inference(self, dataloader):
        """Runs a full forward pass on a dataloader to extract all embeddings and votes."""
        self.model.eval()
        
        all_graph_votes = []
        all_lit_embs = []
        all_clause_embs = []
        all_var_votes = []
        
        total_samples = 0
        start_time = time.perf_counter()

        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = batch_data.to(self.device)

                # Get the outputs
                graph_logits, L_h, C_h, var_votes = self.model(batch_data, return_embeddings=True)
                votes = torch.sigmoid(graph_logits)
                
                # Move to CPU immediately to save GPU RAM
                all_graph_votes.append(votes.cpu())
                all_lit_embs.append(L_h.cpu())
                all_clause_embs.append(C_h.cpu())
                all_var_votes.append(var_votes.cpu())
                
                total_samples += batch_data.num_graphs

        end_time = time.perf_counter()
        
        # Concatenate all batch lists into massive single tensors that will be handled later outside of GPU
        votes = torch.cat(all_graph_votes, dim=0)
        lit_emb = torch.cat(all_lit_embs, dim=0)
        clause_emb = torch.cat(all_clause_embs, dim=0)
        paired_var_votes = torch.cat(all_var_votes, dim=0)

        # Latency
        total_time_ms = (end_time - start_time) * 1000
        latency_ms = total_time_ms / max(1, total_samples)

        return votes, lit_emb, clause_emb, paired_var_votes, latency_ms