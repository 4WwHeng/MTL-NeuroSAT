from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
from mtl_inference import mtl_inference
from data.data_preprocessing import read_data
from utils.utils import count_satisfy

# Test the MTL pipeline on the test set, including both the SAT/UNSAT classification and the candidate assignment decoding for predicted SAT instances.

def load_and_test_mtl(checkpoint_filename, test_data_sat="Test_40_SAT", test_data_unsat="Test_40_UNSAT"):
    graph_votes, split_lit_emb, split_clause_emb, split_adm_prob, split_ctp_prob, latency = mtl_inference(checkpoint_filename, test_data_sat, test_data_unsat) #type: ignore

    data_sat   = read_data(test_data_sat,   is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    solved = 0
    pred_sat = 0
    pred_class_labels = []
    true_class_labels = []
    y_probs = []

    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        vote = graph_votes[i]
        y_probs.append(vote)

        pred_class_labels.append(1 if vote >= 0.5 else 0)
        if vote >= 0.5:
            pred_sat += 1
        true_class_labels.append(is_sat_ground_truth)

        if not is_sat_ground_truth:
            continue

        # assignment decoding
        candidate_prob = split_adm_prob[i]   
        prob_pos_true = candidate_prob[:n_vars, 1]
        prob_neg_true = candidate_prob[n_vars:, 1]
        candidate = (prob_pos_true > prob_neg_true).astype(int)

        count, direct_solved = count_satisfy(candidate, clauses)
        if direct_solved:
            solved += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}...")

    # Print Final Report
    y_true  = np.array(true_class_labels)
    y_probs = np.array(y_probs)
    thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'False Positives (UNSATs sent to WalkSAT)'}")
    print("-" * 80)

    for t in thresholds:
        preds     = (y_probs > t).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall    = recall_score(y_true, preds, zero_division=0)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        except ValueError:
            fp = 0
        print(f"{t:<10.2f} | {precision:<10.4f} | {recall:<10.4f} | {fp}")

    print(f"\nDirectly Solved: {solved}/{pred_sat}")
    print(f"Latency: {latency}")
    print("\nSatisfiability Classification")
    print(classification_report(
        true_class_labels, pred_class_labels,
        target_names=['Label 0 (UNSAT)', 'Label 1 (SAT)']
    ))
