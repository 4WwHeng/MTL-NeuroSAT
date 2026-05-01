import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score
from core.inference import NN_inference
from data.data_preprocessing import read_data
from utils.utils import get_unsat_cores, get_log_path

LOG_PATH = get_log_path()


# Visualization
def visualize_clause_embedding(embeddings, unsat_core_indices=None, satisfied_literals=None, sat=True):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    if sat and satisfied_literals is not None:
        unique_vals = sorted(list(set(satisfied_literals)))
        cmap = plt.get_cmap('tab10', len(unique_vals))
        bounds = np.array(unique_vals + [unique_vals[-1] + 1]) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=satisfied_literals, cmap=cmap, alpha=0.6, edgecolors='w')
        cbar = plt.colorbar(scatter, ticks=unique_vals)
        cbar.set_label('Number of Satisfied Literals in Clause', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        plt.title("PCA of Clause Embeddings (Satisfiable Instance)", fontsize=14)
        filename = os.path.join(LOG_PATH, "PCA_clause_SAT.jpg")
        
    elif unsat_core_indices is not None:
        # Create color mask: Red for core clauses, Blue for non-core
        colors = ['red' if i in unsat_core_indices else 'blue' for i in range(len(embeddings))]
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                    alpha=0.6, c=colors, edgecolors='k')
        
        plt.title("PCA of Clause Embeddings (Red = UNSAT Core)", fontsize=14)
        filename = os.path.join(LOG_PATH, "PCA_clause_UNSAT.jpg")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


# Clustering and Distance Analysis
def calculate_metrics(pred, true):
    acc = accuracy_score(true, pred)
    prec = precision_score(true, pred, zero_division=0)
    rec = recall_score(true, pred, zero_division=0)
    return acc, prec, rec
     
def kmeans_clause(embeddings, core_indices):
    num_clauses = len(embeddings)
    
    true_labels = np.zeros(num_clauses)
    if len(core_indices) > 0:
        true_labels[core_indices] = 1
    
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    pred_labels = kmeans.fit_predict(embeddings)

    # Calculate metrics for both possible cluster assignments 
    # (Since K-means arbitrarily assigns 0 and 1)
    acc1, prec1, rec1 = calculate_metrics(pred_labels, true_labels)
    acc2, prec2, rec2 = calculate_metrics(1 - pred_labels, true_labels)

    # Oracle Selection (Best possible match)
    if acc1 > acc2:
        acc_opt, prec_opt, rec_opt = acc1, prec1, rec1
    else:
        acc_opt, prec_opt, rec_opt = acc2, prec2, rec2

    # Cluster Density / Sparsity Analysis
    centroids = kmeans.cluster_centers_
    
    # Measure mean distance to centroid for Cluster 1
    idx1 = np.where(pred_labels == 1)[0]
    sparse_1 = np.mean(np.linalg.norm(embeddings[idx1] - centroids[1], axis=1)) if len(idx1) > 0 else 0
    
    # Measure mean distance to centroid for Cluster 0
    idx0 = np.where(pred_labels == 0)[0]
    sparse_2 = np.mean(np.linalg.norm(embeddings[idx0] - centroids[0], axis=1)) if len(idx0) > 0 else 0
    
    # Density Heuristic: Assume the more "sparse" cluster is the UNSAT Core
    if sparse_1 > sparse_2:
        accd, precd, recd = acc1, prec1, rec1
    else:
        accd, precd, recd = acc2, prec2, rec2    

    return (acc_opt, prec_opt, rec_opt,   # Oracle Best
            acc1, prec1, rec1,            # Always Label 1
            acc2, prec2, rec2,            # Always Label 0 (Inverse)
            accd, precd, recd)            # Sparsity Heuristic


def clause_dist_center(embeddings, core_indices):
    num_clauses = len(embeddings)

    # Find the global center of all clauses
    center = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - center, axis=1)
    
    core_dists = distances[core_indices] if len(core_indices) > 0 else np.array([0])
    
    ncore_indices = [i for i in range(num_clauses) if i not in core_indices]
    ncore_dists = distances[ncore_indices] if len(ncore_indices) > 0 else np.array([0])

    true_labels = np.zeros(num_clauses)
    if len(core_indices) > 0:
        true_labels[core_indices] = 1

    # Label clauses far away from the center as unsat core
    threshold = np.mean(distances) + np.std(distances)
    pred_labels = (distances > threshold).astype(int)

    acc, prec, rec = calculate_metrics(pred_labels, true_labels)

    return np.mean(core_dists), np.mean(ncore_dists), np.std(core_dists), acc, prec, rec


# Experiment
def unsat_core_experiment(model_name, test_data="Test_40"):
    print("Running NN Inference for UNSAT Cores...")
    votes, lit_emb, clause_emb, var_votes, latency = NN_inference(model_name, test_data)
    
    data_sat = read_data(f"{test_data}_SAT", is_training=False, fixed_label=1)
    data_unsat = read_data(f"{test_data}_UNSAT", is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    kmeans_result = []
    center_dist_result = []
    
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if is_sat_ground_truth or votes[i] > 0.5:
            continue

        C_h = clause_emb[i]
        
        # get unsat core
        core_indices = get_unsat_cores(clauses, n_vars)

        # Skip if no core was found (shouldn't happen on true UNSAT)
        if len(core_indices) == 0:
            continue

        if i == 1002: 
            visualize_clause_embedding(C_h, unsat_core_indices=core_indices, sat=False)

        # 1. K-Means Clustering Results
        kmeans_res = kmeans_clause(C_h, core_indices)
        kmeans_result.append(kmeans_res)

        # 2. Distance to Center Results
        dist_res = clause_dist_center(C_h, core_indices)
        center_dist_result.append(dist_res)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} instances...")

    # Print Final Reports
    kmeans_result = np.array(kmeans_result)
    
    print("\n" + "="*50)
    print(" UNSAT CORE EMBEDDING RESULTS")
    print("="*50)
    
    if len(kmeans_result) == 0:
        print("No valid UNSAT cores were processed. Check your dataset!")
        return

    print('\n--- K-Means on Clause Embeddings ---')
    print(f"{'Method':<30} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6}")
    print("-" * 55)
    print(f"{'Oracle (Best Match)':<30} | {np.mean(kmeans_result[:,0]):.4f} | {np.mean(kmeans_result[:,1]):.4f} | {np.mean(kmeans_result[:,2]):.4f}")
    print(f"{'Label 1 Always':<30} | {np.mean(kmeans_result[:,3]):.4f} | {np.mean(kmeans_result[:,4]):.4f} | {np.mean(kmeans_result[:,5]):.4f}")
    print(f"{'Label 0 Always':<30} | {np.mean(kmeans_result[:,6]):.4f} | {np.mean(kmeans_result[:,7]):.4f} | {np.mean(kmeans_result[:,8]):.4f}")
    print(f"{'Density Heuristic (Sparcer)':<30} | {np.mean(kmeans_result[:,9]):.4f} | {np.mean(kmeans_result[:,10]):.4f} | {np.mean(kmeans_result[:,11]):.4f}")

    center_dist_result = np.array(center_dist_result)
    print('\n--- Distance to Global Center Thresholding ---')
    print(f"Mean Core distance to centroid:     {np.mean(center_dist_result[:,0]):.4f} (Std: {np.std(center_dist_result[:,0]):.4f})")
    print(f"Mean Non-Core distance to centroid: {np.mean(center_dist_result[:,1]):.4f} (Std: {np.std(center_dist_result[:,1]):.4f})")
    print("-" * 55)
    print(f"Accuracy of Core Identification:  {np.mean(center_dist_result[:,3]):.4f}")
    print(f"Precision of Core Identification: {np.mean(center_dist_result[:,4]):.4f}")
    print(f"Recall of Core Identification:    {np.mean(center_dist_result[:,5]):.4f}")
    print("="*50)
