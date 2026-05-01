import numpy as np
from pysat.solvers import Minisat22
from core.inference import NN_inference
from data.data_preprocessing import read_data
from models.NeuroSAT.decoding.votes import hammingd_experiment
from utils.utils import get_unsat_cores


# variable brittleness analysis based on UNSAT core participation and literal propagation
def participation_unsat_core(embeddings, participation_counts):
    """Analyzes brittleness of literals based on their participation in UNSAT cores."""
    center = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - center, axis=1)
    
    unique_bins = np.sort(np.unique(participation_counts))
    bin_stats = {}
    
    for count in unique_bins:
        mask = (participation_counts == count)
        bin_dists = distances[mask]
        
        bin_stats[int(count)] = {
            "mean_dist": np.mean(bin_dists),
            "std_dist": np.std(bin_dists),
            "num_literals": len(bin_dists)
        }
        
    return bin_stats


def rank_literal_propagation(embeddings, clauses, n_vars):
    """Ranks literals by how many other literals they force via unit propagation."""
    center = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - center, axis=1)

    dist_lc = [0.0, 0.0] # [Count, Total Distance]
    n_conflicts = np.zeros(n_vars * 2)
    lit_conflicts = np.zeros(n_vars * 2)
    
    with Minisat22(bootstrap_with=clauses) as solver:
        for var in range(1, n_vars + 1):
            for lit in (var, -var):
                # propagate() checks forcing under temporary assumptions
                res, prop = solver.propagate(assumptions=[lit]) # type: ignore
                
                # Map literal to 0-indexed array [pos_1...pos_n, neg_1...neg_n]
                lit_idx = (lit - 1) if lit > 0 else (n_vars + abs(lit) - 1)
                dist = distances[lit_idx]
                
                if not res:
                    # Direct Conflict: Assuming this literal instantly breaks the formula
                    dist_lc[0] += 1
                    dist_lc[1] += dist
                else:
                    # Successfully propagated 'prop' number of literals
                    count = len(prop)
                    # Safeguard against arrays larger than allocated
                    if count < len(n_conflicts):
                        lit_conflicts[count] += dist
                        n_conflicts[count] += 1
                        
    return lit_conflicts, n_conflicts, dist_lc


def rank_lit_edge_degree(embeddings, clauses, n_vars):
    """Vectorized calculation of literal occurrences (degree) in the graph."""
    center = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - center, axis=1)

    # Flatten clauses and count occurrences instantly
    flat_clauses = np.concatenate(clauses)
    indices = np.where(flat_clauses < 0, 
                       n_vars + np.abs(flat_clauses) - 1, 
                       np.abs(flat_clauses) - 1)
    
    degrees = np.bincount(indices, minlength=n_vars * 2)

    full_degree = np.zeros(100)
    n_degree = np.zeros(100)
    
    for i, deg in enumerate(degrees):
        if deg < 100: 
            full_degree[deg] += distances[i]
            n_degree[deg] += 1
            
    return full_degree, n_degree


# Experiment
def lit_brittleness_experiment(model_name, test_data="Test_40"):
    print("Running NN Inference...")
    votes, lit_emb, clause_emb, var_votes, latency = NN_inference(model_name, test_data)
    
    data_sat = read_data(f"{test_data}_SAT", is_training=False, fixed_label=1)
    data_unsat = read_data(f"{test_data}_UNSAT", is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    dist_bin = [[] for _ in range(100)]
    n_bin = np.zeros(100)
    
    ranked_dist = np.zeros(40 * 2) 
    ranked_num = np.zeros(40 * 2)

    ranked_degree = np.zeros(100)
    n_degree = np.zeros(100)

    n_lit_direct_conflict = 0
    d_lit_direct_conflict = 0

    print("\n--- Starting Brittleness Extraction ---")
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if is_sat_ground_truth or votes[i] >= 0.5:
            continue

        vote = votes[i]
        L_h = lit_emb[i]
        if vote > 0.5:
            continue
        
        # 1. UNSAT Core Participation
        core_indices = get_unsat_cores(clauses, n_vars)
        if len(core_indices) > 0:
            core_lits = np.concatenate([clauses[k] for k in core_indices]) 
            indices = np.where(core_lits < 0, n_vars + np.abs(core_lits) - 1, np.abs(core_lits) - 1)
            participation_counts = np.bincount(indices.astype(int), minlength=n_vars * 2) 
    
            res = participation_unsat_core(L_h, participation_counts)
            for j in res:
                if j < 100: # Safeguard
                    dist_bin[j].append(res[j]["mean_dist"])
                    n_bin[j] += res[j]['num_literals']

        # 2. Literal Propagation Ranking
        rd, rn, lc = rank_literal_propagation(L_h, clauses, n_vars)
        
        # Dynamically pad ranked arrays if n_vars > 40
        if len(rd) > len(ranked_dist):
            ranked_dist = np.pad(ranked_dist, (0, len(rd) - len(ranked_dist)))
            ranked_num = np.pad(ranked_num, (0, len(rn) - len(ranked_num)))
            
        ranked_dist[:len(rd)] += rd
        ranked_num[:len(rn)] += rn
        
        n_lit_direct_conflict += lc[0]
        d_lit_direct_conflict += lc[1]

        # 3. Literal Edge Degree
        rd_deg, rn_deg = rank_lit_edge_degree(L_h, clauses, n_vars)
        ranked_degree += rd_deg
        n_degree += rn_deg

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} instances...")

    # Print Final Reports
    print("\n" + "="*50)
    print(" BRITTLE LITERALS EVALUATION")
    print("="*50)
    
    print("\n--- By UNSAT Core Participation ---")
    for i, c in enumerate(n_bin):
        if c > 0:
            print(f"Participation Count: {i:2d} | Literals: {int(c):4d} | Mean Dist: {np.mean(dist_bin[i]):.4f}")

    print("\n--- By Unit Propagation Count ---")
    prop_means = np.where(ranked_num != 0, ranked_dist / ranked_num, 0)
    # Only print up to the max propagation count we actually observed
    max_prop_idx = np.max(np.nonzero(ranked_num)) if np.any(ranked_num) else 0
    
    for i in range(max_prop_idx + 1):
        if ranked_num[i] > 0:
            print(f"Propagated {i:2d} | Literals: {int(ranked_num[i]):4d} | Mean Dist: {prop_means[i]:.4f}")

    if np.sum(ranked_num) > 0:
        print(f"\nTotal Mean Distance: {np.sum(ranked_dist) / np.sum(ranked_num):.4f}")
        
    if n_lit_direct_conflict > 0:
        print(f"Literals causing direct conflict: {int(n_lit_direct_conflict)}")
        print(f"Mean dist of conflict literals:   {d_lit_direct_conflict / n_lit_direct_conflict:.4f}")

    print("\n--- By Literal Edge Degree ---")
    degree_means = np.where(n_degree != 0, ranked_degree / n_degree, 0)
    max_deg_idx = np.max(np.nonzero(n_degree)) if np.any(n_degree) else 0
    
    for i in range(max_deg_idx + 1):
        if n_degree[i] > 0:
            print(f"Degree {i:2d} | Literals: {int(n_degree[i]):4d} | Mean Dist: {degree_means[i]:.4f}")

