import numpy as np
from feature_probing import dpll_features, saps_features


def entropy(labels):
    """
    Computes entropy of label distribution.
    """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    ent = -np.sum(probs * np.log2(probs[probs > 0]))
    return ent


def literal_to_row_index(idx, n_var):
    if idx > 0:
        return idx - 1
    elif idx < 0:
        return n_var + (abs(idx) - 1)
    else:
        raise Exception


def calculate_stat(data):
    """
    mean, variation coefficient, min, max and entropy.
    """
    stat = {"mean": np.mean(data), "min": np.min(data), "max": np.max(data)}

    std_dev = np.std(data)
    stat["vari_coef"] = std_dev / stat["mean"]
    stat["entropy"] = entropy(data)

    return stat


def problem_size_1(clauses, n_var):
    features = {}
    n_clauses = len(clauses)
    ratio = n_clauses / n_var
    features["n_clause"] = n_clauses
    features["n_var"] = n_var
    features["cv_ratio"] = ratio
    return features


def variable_clause_graph_2(clauses, n_var):
    """
    Variable-Clause Graph is a bipartite graph representing which variables participate in which clause.
    """
    n_clauses = len(clauses)
    adjacency_matrix = np.zeros((n_var * 2, n_clauses), dtype=np.int8)
    for j, clause in enumerate(clauses):
        for literal in clause:
            i = literal_to_row_index(literal, n_var)
            adjacency_matrix[i][j] = 1
    return adjacency_matrix


def variable_graph_3(clauses, n_var):
    """
    Variable Graph has nodes representing variables, and an edge between any variables that occur in a clause together.
    """
    variable_matrix = np.zeros((n_var * 2, n_var * 2), dtype=np.int8)
    for clause in clauses:
        v_c = len(clause)
        for i in range(v_c):
            for j in range(i + 1, v_c):
                idx_i = literal_to_row_index(clause[i], n_var)
                idx_j = literal_to_row_index(clause[j], n_var)
                variable_matrix[idx_i][idx_j] = 1
                variable_matrix[idx_j][idx_i] = 1
    np.fill_diagonal(variable_matrix, 1)
    return variable_matrix


def conflict_graph_4(clauses, n_var):
    """
    Conflict Graph (CG) has nodes representing clauses, and an edge between two clauses whenever they share a negated literal.
    """
    n_clauses = len(clauses)
    conflict_matrix = np.zeros((n_clauses, n_clauses), dtype=np.int8)
    for i in range(n_clauses):
        for j in range(i + 1, n_clauses):
            if any(-v in clauses[j] for v in clauses[i]):
                conflict_matrix[i][j] = 1
                conflict_matrix[j][i] = 1
    return conflict_matrix


def balance_5(clauses, n_var):
    """
    Ratio of positive and negative literals in each clause
    Ratio of positive and negative occurrences of each variable
    Fraction of binary and ternary clauses
    """
    features = {}
    n_clauses = len(clauses)
    ratio_lit = [[0, 0] for _ in range(n_clauses)]
    ratio_var = [[0, 0] for _ in range(n_var)]
    binary_clause = 0
    ternary_clause = 0
    for i in range(n_clauses):
        count = 0
        for l in clauses[i]:
            count += 1
            if l > 0:
                ratio_lit[i][0] += 1
                ratio_var[l - 1][0] += 1
            else:
                ratio_lit[i][1] += 1
                ratio_var[abs(l) - 1][1] += 1
        if count == 2:
            binary_clause += 1
        elif count == 3:
            ternary_clause += 1
    ratio_lit = [p/n if n != 0 else 0 for [p,n] in ratio_lit]
    ratio_var = [p/n if n != 0 else 0 for [p,n] in ratio_var]

    stat_lit = calculate_stat(ratio_lit)
    features["lit_mean"] = stat_lit["mean"]
    features["lit_vari_coef"] = stat_lit["vari_coef"]
    features["lit_entropy"] = stat_lit["entropy"]

    stat_var = calculate_stat(ratio_var)
    features["var_mean"] = stat_var["mean"]
    features["var_vari_coef"] = stat_var["vari_coef"]
    features["var_min"] = stat_var["min"]
    features["var_max"] = stat_var["max"]
    features["var_entropy"] = stat_var["entropy"]

    features["fraction_binary"] = binary_clause / n_clauses
    features["fraction_ternary"] = ternary_clause / n_clauses

    return features


def horn_similarity_6(clauses, n_var):
    """
    Fraction of Horn clauses
    Number of occurrences in a Horn clause for each variable
    """
    features = {}
    occurrence = [0] * (n_var * 2)
    n_clauses = len(clauses)
    n_horn = 0
    for i in range(n_clauses):
        pos_count = 0
        is_horn = True
        for l in clauses[i]:
            if l > 0:
                pos_count += 1
                if pos_count > 1:
                    is_horn = False
                    break
        if is_horn:
            n_horn += 1
            for l in clauses[i]:
                occurrence[literal_to_row_index(l, n_var)] += 1

    features["fraction_horn"] = n_horn / n_clauses

    stat_horn = calculate_stat(occurrence)
    features["horn_mean"] = stat_horn["mean"]
    features["horn_vari_coef"] = stat_horn["vari_coef"]
    features["horn_min"] = stat_horn["min"]
    features["horn_max"] = stat_horn["max"]
    features["horn_entropy"] = stat_horn["entropy"]

    return features


def extract_graph_features(clauses, n_var):
    features = {}

    variable_clause_graph = variable_clause_graph_2(clauses, n_var)

    variable_degree = np.sum(variable_clause_graph, axis=1)
    variable_degree = variable_degree[variable_degree > 0]
    stat_vcgv = calculate_stat(variable_degree)
    features["vcgv_mean"] = stat_vcgv["mean"]
    features["vcgv_vari_coef"] = stat_vcgv["vari_coef"]
    features["vcgv_min"] = stat_vcgv["min"]
    features["vcgv_max"] = stat_vcgv["max"]
    features["vcgv_entropy"] = stat_vcgv["entropy"]

    clause_degree = np.sum(variable_clause_graph, axis=-2)
    clause_degree = clause_degree[clause_degree > 0]
    stat_vcgc = calculate_stat(clause_degree)
    features["vcgc_mean"] = stat_vcgc["mean"]
    features["vcgc_vari_coef"] = stat_vcgc["vari_coef"]
    features["vcgc_min"] = stat_vcgc["min"]
    features["vcgc_max"] = stat_vcgc["max"]
    features["vcgc_entropy"] = stat_vcgc["entropy"]

    variable_graph = variable_graph_3(clauses, n_var)
    node_degree = np.sum(variable_graph, axis=1)
    node_degree = node_degree[node_degree > 0]
    stat_vg = calculate_stat(node_degree)
    features["vg_mean"] = stat_vg["mean"]
    features["vg_vari_coef"] = stat_vg["vari_coef"]
    features["vg_min"] = stat_vg["min"]
    features["vg_max"] = stat_vg["max"]

    return features


def generate_full_feature_vector(clauses, n_var, probe):
    problem_size_features = problem_size_1(clauses, n_var)
    balance_features = balance_5(clauses, n_var)
    horn_features = horn_similarity_6(clauses, n_var)
    graph_features = extract_graph_features(clauses, n_var)

    if probe:
        DPLL_features = dpll_features(clauses, n_var, 1)
        SAPS_features = saps_features(clauses, n_var, 1)

        all_features = {**problem_size_features, **balance_features, **horn_features, **graph_features, **DPLL_features, **SAPS_features}
    else:
        all_features = {**problem_size_features, **balance_features, **horn_features, **graph_features}

    feature_names = sorted(all_features.keys())
    feature_vector = np.array([all_features[name] for name in feature_names])

    return feature_vector
