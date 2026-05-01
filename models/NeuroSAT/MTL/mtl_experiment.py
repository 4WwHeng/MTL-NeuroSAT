import numpy as np
import math
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from data.data_preprocessing import read_data
from mtl_inference import mtl_inference
from utils.utils import count_satisfy, hamming_distance, get_close_assignment


# Experiments for MTL models: 1. classification and solving performancance accross T values, 2. hamming distance vs confidence analysis


# T-Sweep Experiment
def evaluate_T_sweep(checkpoint_filename, test_data_sat="Test_200_SAT", test_data_unsat="Test_200_UNSAT", T_values=[26, 50, 100, 250, 500, 1000]):
    data_sat   = read_data(test_data_sat,   is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat
    
    # Ground truth labels: 1 for SAT, 0 for UNSAT
    y_true = np.array([1] * len(data_sat) + [0] * len(data_unsat))

    print("=" * 80)
    print(f"STARTING T-SWEEP ({test_data_sat} / {test_data_unsat})")
    print(f"Testing T values: {T_values}")
    print("=" * 80)

    for T in T_values:
        print(f"\n\n{'='*30} T = {T} {'='*30}")
        
        graph_votes, split_lit_emb, split_clause_emb, split_adm_prob, split_ctp_prob, latency = mtl_inference(
            checkpoint_filename=checkpoint_filename,
            test_data_sat=test_data_sat,
            test_data_unsat=test_data_unsat,
            T_val=T
        ) # type: ignore
        
        y_probs = np.array(graph_votes)
        pred_class_labels = (y_probs >= 0.5).astype(int)
        
        solved = 0
        pred_sat_count = np.sum(pred_class_labels)

        # Evaluate Assignment Validity (Only for true SAT instances)
        for i, (clauses, n_vars, is_sat) in enumerate(data):
            if not is_sat:
                continue
            
            candidate_prob = split_adm_prob[i]
            candidate = np.argmax(candidate_prob, axis=1)
            
            _, direct_solved = count_satisfy(candidate, clauses) 
            
            if direct_solved:
                solved += 1

        # Print Final Report
        total_inference_time = latency * len(data) / 1000  # Converted to seconds
        
        print(f"Total Inference Time: {total_inference_time:.3f} s ({latency:.4f} ms/sample)")
        print(f"Directly Solved:      {solved} / {pred_sat_count} (Predicted SAT)")
        print("-" * 80)
        
        # Threshold Analysis (Precision, Recall, False Positives)
        thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'False Positives (Wasted WalkSATs)'}")
        print("-" * 80)
        
        for t in thresholds:
            preds = (y_probs > t).astype(int)
            precision = precision_score(y_true, preds, zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
            
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            except ValueError:
                fp = 0 
                
            print(f"{t:<10.2f} | {precision:<10.4f} | {recall:<10.4f} | {fp}")

        print("-" * 80)


# Hamming Distance vs Confidence Analysis for MTL models
def hammingd_experiment_mtl(checkpoint_filename, test_data_sat="Test_40_SAT", test_data_unsat="Test_40_UNSAT"):
    graph_votes, _, _, split_adm_prob, _, latency = mtl_inference(
        checkpoint_filename, test_data_sat, test_data_unsat
    ) # type: ignore
    
    data_sat = read_data(test_data_sat, is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    dist_bin = [[0, 0.0] for _ in range(10)] 
    
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if not is_sat_ground_truth:
            continue

        vote = graph_votes[i]
        
        candidate_prob = split_adm_prob[i] 
        candidate = np.argmax(candidate_prob, axis=1)
        
        _, direct_solved = count_satisfy(candidate, clauses) 
        
        assignment = None
        if not direct_solved:
            assignment = get_close_assignment(candidate, clauses, n_vars)
            
        if assignment is not None:
            hd = hamming_distance(candidate, assignment)
            idx = min(math.floor(vote * 10), 9) 
            dist_bin[idx][0] += 1
            dist_bin[idx][1] += hd

    # Print Final Report
    print("\n" + "="*50)
    print("HAMMING DISTANCE BY CONFIDENCE BINS")
    print("="*50)
    
    confidence_floor = 0.0
    for count, total_hd in dist_bin:
        confidence_ceil = round(confidence_floor + 0.1, 2)
        
        if count == 0:
            print(f"Confidence {confidence_floor:.2f}-{confidence_ceil:.2f}: mean distance 0.0000 between 0 problems.")
        else:
            mean_hd = total_hd / count
            print(f"Confidence {confidence_floor:.2f}-{confidence_ceil:.2f}: mean distance {mean_hd:.4f} between {count} problems.")
            
        confidence_floor = confidence_ceil


if __name__ == "__main__":
    evaluate_T_sweep('MTL_NueroSAT_loss_best.pth')
    hammingd_experiment_mtl("MTL_NueroSAT_loss_best.pth")
