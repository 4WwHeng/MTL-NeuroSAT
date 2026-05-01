import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from core.inference import NN_inference
from data.data_preprocessing import read_data
from utils.utils import get_close_assignment, count_satisfy

# Analysis for literal embeddings of SAT instances (satsifying assignments decoding and assignment confidence).

LOG_PATH = "./NNSAT_Project/Logs"
LOAD_PATH = "./NNSAT_Project/Checkpoints/NeuroSAT_SR40.pth"

# anchor for semi-supervised K-Means initialisation, derived from the mean literal embeddings of assigned True and False literals one one randomly selected validation data. 
lit_anchors = [
    [1.0246180295944214, 1.6246752738952637, -1.103594422340393, 0.4757509231567383, 0.7383733987808228, -0.11564525216817856, 0.8675805330276489, 0.45566171407699585, -0.2687462866306305, 0.33757829666137695, 1.0733284950256348, -0.7989670038223267, 0.5840482115745544, -0.18581733107566833,-0.34534627199172974, -1.8724615573883057, 0.5281398296356201, -0.18048445880413055, -1.2205777168273926, 0.2410271316766739, 0.8657766580581665,0.48036739230155945, 1.0539436340332031, 1.3146182298660278, 1.1731195449829102, 0.06316627562046051, -2.340776205062866, -1.1716054677963257, 0.44447383284568787, -0.3806634545326233, 0.9724876880645752, -1.9516284465789795, 0.5410627722740173, -1.5549780130386353, 0.41599443554878235,0.04803603142499924, -0.17041164636611938, 1.0397132635116577, 0.33680060505867004, 0.08121398836374283, 0.29226937890052795, 1.5810422897338867,-2.116121530532837, -0.8445603847503662, -0.21354471147060394, 0.5375057458877563, 0.024192631244659424, 0.05012292042374611, -2.038912534713745,0.3523831367492676, 0.7910587787628174, 0.947466254234314, -1.6272895336151123, 0.5856972336769104, 0.5381253361701965, 0.5720170736312866, -1.770911455154419, -0.07459181547164917, 1.25898277759552, -1.5047311782836914, -0.021777868270874023, 0.5342482328414917, -0.9161946177482605,-0.12551714479923248],
    [1.5001904964447021, -0.5755941867828369, -0.5544725656509399, 0.8695420026779175, 0.3717515468597412, 0.011020340025424957, 0.11017096042633057,0.2953210175037384, 0.05436628311872482, -1.8207696676254272, 0.9966498613357544, -1.3512933254241943, 0.17549742758274078, 0.08492794632911682,-0.10514052957296371, -0.3865259885787964, 0.3368930518627167, 0.12239592522382736, 0.16986960172653198, -0.3718695640563965, 0.5945349335670471, -0.6836046576499939, 1.507506251335144, 1.3057329654693604, 0.4064779281616211, 0.5167398452758789, -0.2943613529205322, -0.4624800384044647, 0.2891490161418915, -2.134695529937744, -2.129471778869629, -1.8944940567016602, 0.25806325674057007, -0.07148939371109009, 0.28309547901153564, 0.19176331162452698, 1.9690961837768555, 0.8790593147277832, 0.921879768371582, 0.240884467959404, 0.12686505913734436, -0.7783605456352234, 0.2531958818435669, -2.2975215911865234, 0.014868654310703278, 0.35576266050338745, -0.29722732305526733, 0.06231429800391197, -0.2720595598220825, 0.1569918990135193, 0.6367924809455872, 1.412329912185669, -2.199267864227295, 0.5468512773513794, 0.5390158295631409, 1.1232075691223145, 1.0948176383972168, -1.1888923645019531, -2.45424222946167, 0.31571245193481445, 1.1019477844238281, 0.7024862170219421, -1.0506951808929443, 0.3503265678882599]
]

# Decoding Algorithms
def decode_kmeans_dist(L_emb, clauses, n_vars):
    """
    Decodes assignments using K-Means and distances to cluster centers.
    Exactly what is done in the paper and in find solutions of NeuroSAT.
    This is a copy of the same method in the core Neurosat.py file, but adapted to be used as a standalone decoder for analysis.
    """
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(L_emb)
    c1, c2 = kmeans.cluster_centers_  

    pos_h = L_emb[:n_vars] 
    neg_h = L_emb[n_vars:] 

    d1 = np.sum((pos_h - c1)**2, axis=1) + np.sum((neg_h - c2)**2, axis=1)
    d2 = np.sum((pos_h - c2)**2, axis=1) + np.sum((neg_h - c1)**2, axis=1)
    
    candidate_A = np.where(d1 > d2, 1, 0)
    candidate_B = 1 - candidate_A

    cA, solvedA = count_satisfy(candidate_A, clauses) 
    if solvedA:
        return candidate_A, np.where(candidate_A == 1, d2, d1), True
        
    cB, solvedB = count_satisfy(candidate_B, clauses)
    if solvedB:
        return candidate_B, np.where(candidate_B == 1, d1, d2), True
        
    if cA >= cB:
        return candidate_A, np.where(candidate_A == 1, d2, d1), False
    return candidate_B, np.where(candidate_A == 1, d2, d1), False


def decode_kmeans_initialisation(L_emb, clauses, n_vars, lit_anchors):
    """kmeans using anchor initialization."""
    kmeans = KMeans(n_clusters=2, init=lit_anchors, n_init=1, random_state=42).fit(L_emb)
    
    # kmeans.transform returns the distance to each cluster center
    distances = kmeans.transform(L_emb)
    
    # Isolate the distances to the 'True' cluster (Index 1)
    dist_to_true = distances[:, 1]
    
    # Split into positive and negative literal distances
    pos_dist_to_true = dist_to_true[:n_vars]
    neg_dist_to_true = dist_to_true[n_vars:]
    
    # If the positive literal is closer to 'True' than the negative literal is, assign 1 (True). Otherwise, assign 0 (False).
    candidate = np.where(pos_dist_to_true < neg_dist_to_true, 1, 0)
    
    # Verify how many clauses it satisfied 
    cA, solved = count_satisfy(candidate, clauses) 
    
    return candidate, solved


# Visualization
def visualize_lit_embeddings(embeddings, labels=None, 
                             label_f='Assigned False (0)', 
                             label_t='Assigned True (1)', 
                             sat=True):
    """Projects literal embeddings down to 2D using PCA and plots them."""
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    if labels is None:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.6, s=50)
    else:
        sat_indices = np.where(labels == 1)[0]
        unsat_indices = np.where(labels == 0)[0]
        
        plt.scatter(embeddings_2d[unsat_indices, 0], embeddings_2d[unsat_indices, 1], 
                    c='blue', label=label_f, alpha=0.6, s=50)
        plt.scatter(embeddings_2d[sat_indices, 0], embeddings_2d[sat_indices, 1], 
                    c='red', label=label_t, alpha=0.6, s=50)
        plt.legend(fontsize=12)

    plt.title("PCA of Literal Embeddings", fontsize=16)
    plt.xlabel("Singular Vector 1", fontsize=12)
    plt.ylabel("Singular Vector 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Ensure LOG_PATH is defined globally, or pass it in
    filename = os.path.join(LOG_PATH, f"Lit_PCA_{'sat' if sat else 'unsat'}.jpg")
    plt.savefig(filename)
    plt.show()


# Experiments
def assignment_decoding(model_name, test_data="Test_40"):
    votes, lit_emb, clause_emb, var_votes, latency = NN_inference(model_name, test_data)
    
    data_sat = read_data(f"{test_data}_SAT", is_training=False, fixed_label=1)
    data_unsat = read_data(f"{test_data}_UNSAT", is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    pred_sat = 0
    kmeans_solved = 0
    init_solved = 0
    
    kpred_labels = []
    init_labels = []
    true_labels = []
    
    correct_dist = []
    wrong_dist = []

    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if not is_sat_ground_truth:
            continue

        vote = votes[i]
        if vote >= 0.5:
            pred_sat += 1
        else:
            continue

        L_h = lit_emb[i]
        
        # 1. K-Means Distance
        k_means_candidate, var_dist, direct_solved = decode_kmeans_dist(L_h, clauses, n_vars)

        # 2. K-Means Initialisation
        init_candidate, init_s = decode_kmeans_initialisation(L_h, clauses, n_vars, lit_anchors)

        # 3. Ground truth assignment (for analysis only, not used in decoding)
        assignments = get_close_assignment(k_means_candidate, clauses, n_vars)
        if assignments is None:
            print(f"Warning: WalkSAT failed to find a solution for problem {i}.")
            continue

        if direct_solved:
            kmeans_solved += 1
            
        if init_s:
            init_solved += 1

        correct_dist.append(np.mean(var_dist[(k_means_candidate == assignments)]))
        wrong_dist_list = var_dist[(k_means_candidate != assignments)]
        if len(wrong_dist_list) > 0:
            wrong_dist.append(np.mean(wrong_dist_list))

        if not direct_solved:
            kpred_labels.extend(k_means_candidate)
            true_labels.extend(assignments)
            init_labels.extend(init_candidate)

        if i == 7:
            visualize_lit_embeddings(L_h, assignments)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} ground-truth SAT problems...")

    # Print Final Report
    print("\n" + "="*40)
    print(" DECODING RESULTS REPORT")
    print("="*40)
    print(f"Total True SAT Problems Predicted as SAT: {pred_sat}")
    print("-" * 40)
    print(f"Solved by K-Means Dist: {kmeans_solved}/{pred_sat} ({(kmeans_solved/max(1, pred_sat))*100:.2f}%)")
    print(f"Solved by K-Means Init: {init_solved}/{pred_sat} ({(init_solved/max(1, pred_sat))*100:.2f}%)")
    
    print("\n--- Distance Analysis ---")
    print(f"Mean distance to center (Correctly assigned vars): {np.mean(correct_dist):.4f}")
    print(f"Mean distance to center (Wrongly assigned vars):   {np.mean(wrong_dist):.4f}")
    
    target_names = ['Label 0 (False)', 'Label 1 (True)']
    
    if len(true_labels) > 0:
        print("\n--- Classification Report (K-Means on UNSOLVED) ---")
        print(classification_report(true_labels, kpred_labels, target_names=target_names))

        print("\n--- Classification Report (Init on UNSOLVED) ---")
        print(classification_report(true_labels, init_labels, target_names=target_names))


# Assignment Confidence
def extract_confident_variable(var_dist):
    mean = np.mean(var_dist)
    std = np.std(var_dist)

    return np.where(var_dist < 0.5 * mean + std)[0]
    
        
def assignment_confidence(model_name, test_data="Test_40"):
    votes, lit_emb, clause_emb, var_votes, latency = NN_inference(model_name, test_data)
    
    data_sat = read_data(f"{test_data}_SAT", is_training=False, fixed_label=1)
    data_unsat = read_data(f"{test_data}_UNSAT", is_training=False, fixed_label=0)
    data = data_sat + data_unsat
    
    conf_acc = []
    conf_count = []
    total = 0
    all_precision = 0
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if not is_sat_ground_truth:
            continue

        vote = votes[i]
        if vote < 0.5:
            continue
            
        L_h = lit_emb[i]
        k_means_candidate, var_dist, direct_solved = decode_kmeans_dist(L_h, clauses, n_vars)
        if direct_solved:
            continue

        total += 1
        assignments = get_close_assignment(k_means_candidate, clauses, n_vars)
        if assignments is None:
            print(f"Warning: WalkSAT failed to find a solution for problem {i}.")
            continue

        c_idx = extract_confident_variable(var_dist)

        if len(c_idx) > 0:
            correct_count = np.sum(k_means_candidate[c_idx] == assignments[c_idx])
            conf_count.append(len(c_idx))
            acc = correct_count / len(c_idx)
            if correct_count == len(c_idx):
                all_precision += 1
            conf_acc.append(acc)
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}...")

    
    print(f"Mean acc of Confidence Var: {np.mean(conf_acc)}")
    print(f"Mean number of Confidence Var: {np.mean(conf_count)}")
    print(f"Number of instances with 100% precision: {all_precision}/{total}")


