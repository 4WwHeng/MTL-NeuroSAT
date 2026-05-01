import os
import math
import numpy as np
import matplotlib.pyplot as plt
from core.inference import NN_inference
from data.data_preprocessing import read_data
from sat_lit import decode_kmeans_dist
from utils.utils import get_close_assignment, find_backbone, get_log_path

# Votes Analysis for NeuroSAT

LOG_PATH = get_log_path()

# Backbone Confidence Analysis
def backbone_experiment(model_name, test_data="Test_40"):
    """Analyzes the relationship between NeuroSAT's variable votes and the backbone variables of the SAT problem. Overlapping std error bars so not included in the report."""
    votes, lit_emb, clause_emb, var_votes, latency = NN_inference(model_name, test_data)
    
    data_sat = read_data(f"{test_data}_SAT", is_training=False, fixed_label=1)
    data_unsat = read_data(f"{test_data}_UNSAT", is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    all_mean_bb, all_mean_nbb, all_bb_acc = [], [], []
    plot_bb, plot_nbb = [], []
    correct_conf, wrong_conf = [], []

    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if not is_sat_ground_truth or votes[i] < 0.5:
            continue
            
        L_h = lit_emb[i]
        vvotes = np.array(var_votes[i])

        # Get Predictions & Truth
        k_means_candidate, var_dist, direct_solved = decode_kmeans_dist(L_h, clauses, n_vars)
        assignments = get_close_assignment(k_means_candidate, clauses, n_vars)
        
        k_means_candidate = np.array(k_means_candidate)
        
        # Track confidence (magnitude) of correct vs wrong predictions
        correct_conf.extend(np.abs(vvotes[k_means_candidate == assignments]))
        wrong_conf.extend(np.abs(vvotes[k_means_candidate != assignments]))
        
        # Backbone Analysis
        backbone_set = set(abs(x) - 1 for x in find_backbone(clauses))   
        bb_mask = np.array([idx in backbone_set for idx in range(n_vars)])
        nbb_mask = ~bb_mask
        
        bb_vals = vvotes[bb_mask]
        nbb_vals = vvotes[nbb_mask]
    
        m_bb = bb_vals.mean() if len(bb_vals) > 0 else None
        m_nbb = nbb_vals.mean() if len(nbb_vals) > 0 else None

        if m_bb is not None:
            all_mean_bb.append(m_bb)
            all_bb_acc.append((bb_vals > 0).mean()) 
            
        if m_nbb is not None:
            all_mean_nbb.append(m_nbb)

        if m_bb is not None and m_nbb is not None:
            plot_bb.append(m_bb)
            plot_nbb.append(m_nbb)
    
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} instances...")
    
    # Print Final Report
    print("\n" + "="*50)
    print(f"{'Global Backbone Statistics':^50}")
    print("="*50)
    print(f"{'Metric':<25} | {'Mean':<10} | {'Std':<10}")
    print("-" * 50)
    print(f"{'BB Vote Magnitude':<25} | {np.mean(all_mean_bb):>10.3f} | {np.std(all_mean_bb):>10.3f}")
    print(f"{'Non-BB Vote Magnitude':<25} | {np.mean(all_mean_nbb):>10.3f} | {np.std(all_mean_nbb):>10.3f}")
    print(f"{'BB Polarity Accuracy':<25} | {np.mean(all_bb_acc)*100:>9.1f}% | {np.std(all_bb_acc)*100:>9.1f}%")
    print("-" * 50)
    print(f"{'Correct Assignment Conf':<25} | {np.mean(correct_conf):>10.3f} | {np.std(correct_conf):>10.3f}")
    print(f"{'Wrong Assignment Conf':<25} | {np.mean(wrong_conf):>10.3f} | {np.std(wrong_conf):>10.3f}")
    print("="*50)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(plot_nbb, plot_bb, alpha=0.5, color='teal', s=20, label='Instances')
    
    high = max(max(plot_bb), max(plot_nbb)) if plot_bb else 8
    low = min(min(plot_bb), min(plot_nbb)) if plot_bb else 0
    plt.plot([low, high], [low, high], color='red', linestyle='--', label='Line of Indifference')
    
    plt.title("Neural Confidence: Backbone vs. Non-Backbone", fontsize=14)
    plt.xlabel("Mean Non-Backbone Vote Magnitude", fontsize=12)
    plt.ylabel("Mean Backbone Vote Magnitude", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    filename = os.path.join(LOG_PATH, "backbone_clustering.jpg")
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot to {filename}")


# Hamming Distance 
def hamming_distance(pred, truth):
    return np.mean(np.array(pred) != np.array(truth))

def hammingd_experiment(model_name, test_data="Test_40"):
    votes, lit_emb, clause_emb, var_votes, latency = NN_inference(model_name, test_data)
    
    data_sat = read_data(f"{test_data}_SAT", is_training=False, fixed_label=1)
    data_unsat = read_data(f"{test_data}_UNSAT", is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    # dist_bin:[[count, total_hamming_distance], ...]
    dist_bin = [[0, 0] for _ in range(10)]
    
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if not is_sat_ground_truth:
            continue

        vote = votes[i]
        L_h = lit_emb[i]
        
        k_means_candidate, var_dist, direct_solved = decode_kmeans_dist(L_h, clauses, n_vars)
        
        truth_assignment = k_means_candidate if direct_solved else get_close_assignment(k_means_candidate, clauses, n_vars)
            
        if truth_assignment is not None:
            hd = hamming_distance(k_means_candidate, truth_assignment)
            idx = min(math.floor(vote * 10), 9)
            dist_bin[idx][0] += 1
            dist_bin[idx][1] += hd

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} instances...")

    # Print Final Report
    print("\n" + "="*50)
    print(f"{'Hamming Distance by Network Confidence':^50}")
    print("="*50)
    
    for idx, (count, total_hd) in enumerate(dist_bin):
        lower_bound = idx * 0.1
        upper_bound = lower_bound + 0.1
        
        if count == 0:
            print(f"Confidence [{lower_bound:.1f} - {upper_bound:.1f}) : N/A (0 problems)")
        else:
            mean_hd = total_hd / count
            print(f"Confidence [{lower_bound:.1f} - {upper_bound:.1f}) : Mean HD = {mean_hd:.4f} | Count = {count}")

