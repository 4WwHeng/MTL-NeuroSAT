import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_distances
from core.inference import NN_inference
from data.data_preprocessing import read_data
from sat_lit import decode_kmeans_dist
from utils.utils import get_close_assignment

# analysis of clause embeddings for SAT instances.

LOG_PATH = "./NNSAT_Project/Logs"

# anchors for semi-supervised K-Means initialisation, derived from the mean clause embeddings of known brittle, medium, and slack clauses of one randomly selected validation data. 
anchors = np.array([
    [-1.0688973665237427, -0.21951384842395782, -0.46788105368614197, 0.8302081227302551, -0.0454467236995697, 0.6869118213653564, 0.43211111426353455, -0.5045120716094971, 0.14679738879203796, -1.4756536483764648, -0.3749856948852539, 2.807408332824707, 0.007980316877365112, -1.8174999952316284, 0.14887690544128418, -1.4717880487442017, -0.8639873266220093, 0.08314420282840729, 0.3478657603263855, -0.650973379611969, 1.0505726337432861, 0.8313286900520325, 0.597281813621521, -0.8978406190872192, -0.06983242183923721, -0.6767086386680603, 0.7024723291397095, -0.48745113611221313, -0.7980130910873413, -1.0489367246627808, -0.262912392616272, 0.28391972184181213, -0.39430558681488037, 0.8514984250068665, 1.1722832918167114, -0.501785933971405, 0.30672329664230347, -1.257267951965332, 0.7028354406356812, -0.2929333746433258, 2.1056532859802246, 2.5524742603302, -1.8616507053375244, -2.092884063720703, -0.7544057369232178, -0.3862724006175995, 0.079732745885849, 0.10422994196414948, 1.1329216957092285, 0.40780699253082275, 0.10329237580299377, -1.3803635835647583, 1.0668517351150513, 1.6234571933746338, 0.8954693675041199, 0.8460555076599121, 0.3137615919113159, 0.3239768147468567, -0.5162566900253296, 0.6249728798866272, -0.48900002241134644, -0.07234808802604675, -0.017445452511310577, -0.8922260999679565],
    
    [-1.5279502868652344, 0.3235637843608856, -0.4720613956451416, 1.1833091974258423, -0.20366644859313965, 1.3114039897918701, 0.7232760787010193, -0.35900038480758667, 0.04146215692162514, -2.0341343879699707, 0.7787375450134277, 2.5895705223083496, 0.1882336437702179, -1.4363977909088135, 0.13801339268684387, -1.4863892793655396, -0.40857747197151184, 0.042216889560222626, 0.19380693137645721, -1.3005728721618652, 0.34111329913139343, 0.05142998695373535, 0.8670247793197632, -0.47594577074050903, 0.006219431757926941, -1.0281274318695068, 0.16415943205356598, -0.6184557676315308, -0.42798495292663574, -1.1463019847869873, 0.3335362672805786, -0.20489177107810974, -0.18151524662971497, 1.2750316858291626, 1.3170021772384644, -0.645646870136261, 0.190614253282547, -1.8306357860565186, 0.18610264360904694, -0.18428221344947815, 2.219907283782959, 2.024059295654297, -1.015639305114746, -2.1484124660491943, -0.21311832964420319, -0.3078881502151489, 0.016720682382583618, 0.039940718561410904, 1.441049337387085, 0.23558026552200317, -0.0911223292350769, -1.5367647409439087, 0.42624956369400024, 1.9830310344696045, 1.0848362445831299, -0.7424460649490356, 0.4697723686695099, 0.1417718529701233, -0.4244024157524109, 0.5251449942588806, -0.644385814666748, 0.4455810487270355, 0.00969057809561491, -0.14682269096374512],
    
    [-0.9205459356307983, 0.5068937540054321, -0.30897730588912964, 1.4885197877883911, -0.3312697410583496, 0.7609518766403198, 0.8773278594017029, -0.13085158169269562, 0.01613633707165718, -1.9206739664077759, 0.6348748207092285, 2.3576290607452393, 0.16138938069343567, -1.3286898136138916, 0.13784661889076233, -1.5230414867401123, -0.25222063064575195, 0.061305854469537735, 0.10643445700407028, -1.7906959056854248, 0.05307993292808533, -0.006894350051879883, 1.026405930519104, -0.14851722121238708, -0.4092711806297302, -0.8168495893478394, 0.08391366899013519, -0.38607364892959595, -0.10577154159545898, -1.0019550323486328, 0.5971019864082336, -0.09321681410074234, -0.174107164144516, 1.4654673337936401, 0.9603861570358276, -1.1297334432601929, 0.12878839671611786, -2.178971290588379, 0.08837005496025085, -0.019934415817260742, 2.349114179611206, 1.6984260082244873, -0.5487681031227112, -2.293565511703491, -0.02514365315437317, -0.2457481324672699, 0.05995485186576843, 0.09765034168958664, 1.5366042852401733, 0.1790771484375, -0.028973836451768875, -1.7222709655761719, 0.22677424550056458, 1.858218789100647, 0.7919908761978149, -1.1730715036392212, 0.44154781103134155, 0.12040876597166061, -0.21332620084285736, 0.43571311235427856, -1.0171802043914795, 0.6571588516235352, 0.04707786813378334, 0.3010561168193817]
])

# Clause satisfaction count function
def clause_satlit_count(clauses, assignment):
    assignment = np.array(assignment)
    lit_sat = []
    
    for c in clauses:
        c_arr = np.array(c)
        var_indices = np.abs(c_arr) - 1
        is_positive = (c_arr > 0)
        
        # A literal is satisfied if it's positive and assignment is 1, 
        # OR if it's negative and assignment is 0
        satisfied = (assignment[var_indices] == is_positive)
        lit_sat.append(np.sum(satisfied))
        
    return np.array(lit_sat)


# Visualization of clause embeddings
def visualise_clause_embeddings_with_labels(embeddings, labels=None):
    if labels is None:
        kmeans_model = KMeans(n_clusters=3, n_init=10, random_state=42)
        labels = kmeans_model.fit_predict(embeddings)
        
    pca = PCA(n_components=2)
    sat_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(sat_2d[:, 0], sat_2d[:, 1], 
                          c=labels, cmap='Set1', alpha=0.7, edgecolors='k', s=60)

    legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
    plt.gca().add_artist(legend1)

    plt.title("Clause Embedding SAT 3-Cluster Partitioning (K-Means)", fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.4)
    
    filename = os.path.join(LOG_PATH, "PCA_3Cluster_SAT.jpg")
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


# 3. K-Means Clustering Strategies
def map_true_labels(lit_sat):
    """helper to map satisfaction counts to Tiers: 0 (Brittle), 1 (Medium), 2 (Slack)"""
    return np.where(lit_sat == 1, 0, np.where(lit_sat == 2, 1, 2))

def kmeans_clause_oracle(embeddings, lit_sat):
    """kmeans using the labels to find the best possible cluster mapping to the true tiers, representing an upper bound on K-Means performance."""
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    true_labels = map_true_labels(lit_sat)

    max_acc = 0
    pred_labels = []
    
    # Brute-force find the best cluster mapping (Oracle)
    for i, j, k in permutations([0, 1, 2]):
        mapp = [i, j, k]
        cpred_labels = np.array([mapp[lbl] for lbl in labels])
        acc = accuracy_score(true_labels, cpred_labels)
        if acc > max_acc:
            max_acc = acc
            pred_labels = cpred_labels
            
    return true_labels, pred_labels 

def kmeans_clause_similarity(embeddings, lit_sat):
    """kmeans using geometric information of clusters."""
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    center = kmeans.cluster_centers_
    true_labels = map_true_labels(lit_sat)

    # Find the two cluster centers furthest from each other
    dist_matrix = cosine_distances(center)
    idx1, idx2 = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    
    all_indices = {0, 1, 2}
    medium_idx = list(all_indices - {idx1, idx2})[0] # type: ignore
    
    d_med_to_idx1 = dist_matrix[medium_idx, idx1]
    d_med_to_idx2 = dist_matrix[medium_idx, idx2]
    
    if d_med_to_idx1 < d_med_to_idx2:
        dense_idx, sparse_idx = idx1, idx2
    else:
        dense_idx, sparse_idx = idx2, idx1

    mapp = [-1, -1, -1]
    mapp[sparse_idx] = 0
    mapp[medium_idx] = 1
    mapp[dense_idx] = 2
    
    pred_labels = np.array([mapp[lbl] for lbl in labels])
    return true_labels, pred_labels 

def kmeans_clause_initialisation(embeddings, lit_sat):
    """kmeans using anchor initialization."""
    kmeans = KMeans(n_clusters=3, init=anchors, n_init=1, random_state=42)
    pred_labels = kmeans.fit_predict(embeddings)
    true_labels = map_true_labels(lit_sat)
    return true_labels, pred_labels 


# Experiment 
def sat_core_experiment(model_name, test_data="Test_40"):
    print("Running NN Inference for SAT Clause Brittleness...")
    votes, lit_emb, clause_emb, var_votes, latency = NN_inference(model_name, test_data)
    
    data_sat = read_data(f"{test_data}_SAT", is_training=False, fixed_label=1)
    data_unsat = read_data(f"{test_data}_UNSAT", is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    otrue_labels, opred_labels = [], []
    strue_labels, spred_labels = [], []
    itrue_labels, ipred_labels = [], []
    
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if not is_sat_ground_truth or votes[i] < 0.5:
            continue

        L_h = lit_emb[i]
        C_h = clause_emb[i]

        k_means_candidate, var_dist, direct_solved = decode_kmeans_dist(L_h, clauses, n_vars)
        
        if not direct_solved:
            truth_assignment = get_close_assignment(k_means_candidate, clauses, n_vars)
        else:
            truth_assignment = k_means_candidate
            
        if truth_assignment is None:
            continue
            
        sat_lit_count = clause_satlit_count(clauses, truth_assignment)

        # 1. K-Means Oracle
        true_labels, pred_labels = kmeans_clause_oracle(C_h, sat_lit_count)
        otrue_labels.extend(true_labels)
        opred_labels.extend(pred_labels)

        # 2. K-Means Similarity
        true_labels, pred_labels = kmeans_clause_similarity(C_h, sat_lit_count)
        strue_labels.extend(true_labels)
        spred_labels.extend(pred_labels)

        # 3. K-Means Initialisation
        true_labels, pred_labels = kmeans_clause_initialisation(C_h, sat_lit_count)
        itrue_labels.extend(true_labels)
        ipred_labels.extend(pred_labels)

        if i == 7: 
            print(f"Plotting Graph {i}...")
            visualise_clause_embeddings_with_labels(C_h)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} instances...")

    # Print Final Reports
    target_names = ['Tier 0 (Brittle)', 'Tier 1 (Medium)', 'Tier 2 (Slack)']
    
    print("\n" + "="*55)
    print(" CLAUSE BRITTLENESS EVALUATION (K-MEANS ONLY)")
    print("="*55)
    
    if len(otrue_labels) > 0:
        print("\nK-Means 3-Clustering (Oracle)")
        print(classification_report(otrue_labels, opred_labels, target_names=target_names, zero_division=0))

    if len(strue_labels) > 0:
        print("\nK-Means 3-Clustering (Sorted by Cluster Similarity)")
        print(classification_report(strue_labels, spred_labels, target_names=target_names, zero_division=0))

    if len(itrue_labels) > 0:
        print("\nK-Means 3-Clustering (Anchor Initialised)")
        print(classification_report(itrue_labels, ipred_labels, target_names=target_names, zero_division=0))

