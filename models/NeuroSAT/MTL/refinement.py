import torch
from torch_geometric.utils import scatter
from mtl_train import MTLParadigm


# unsupervised refinement loss. 


def compute_clause_unsat_loss(adm_logits, edge_index, n_vars, n_clauses, device=None):
    """Simple unsat probability loss based on the product of literal probabilities. Encourages at least one literal in each clause to be satisfied."""
    probs = torch.softmax(adm_logits, dim=1)[:, 1]

    lit_offsets = torch.cat([torch.tensor([0], device=device),
                              torch.cumsum(n_vars * 2, dim=0)[:-1]])
    lit_to_graph_offset = torch.repeat_interleave(lit_offsets, n_vars * 2)
    v_counts = torch.repeat_interleave(n_vars, n_vars * 2)
    rel_idx = torch.arange(len(probs), device=device) - lit_to_graph_offset
    is_neg_node = rel_idx >= v_counts

    lit_node = edge_index[0]
    lit_true_prob = torch.where(
        is_neg_node[lit_node],
        1.0 - probs[lit_node - v_counts[lit_node]],
        probs[lit_node]
    )

    # Log-space for numerical stability
    log_unsat = scatter(torch.log(1.0 - lit_true_prob + 1e-8),
                        edge_index[1], dim=0,
                        dim_size=n_clauses.sum(), reduce='sum')
    loss_j = torch.exp(log_unsat)  # per-clause unsat probability

    # Entropy reg to encourage confident assignments
    entropy_reg = 0.05 * (probs * (1.0 - probs)).mean()

    return loss_j, entropy_reg


def compute_penalty_diffsat_loss(adm_logits, edge_index, n_vars, n_clauses, device=None):
    """DiffSAT-inspired loss that penalizes clauses based on how many literals are likely unsatisfied. Encourages the model to push unsatisfied clauses towards having at least one satisfied literal."""
    probs = torch.softmax(adm_logits, dim=1)[:, 1]  # shape: [2 * sum(n_vars)]

    # 1. Identify negative literal nodes
    lit_offsets = torch.cat([torch.tensor([0], device=device),
                              torch.cumsum(n_vars * 2, dim=0)[:-1]])
    lit_to_graph_offset = torch.repeat_interleave(lit_offsets, n_vars * 2)
    v_counts = torch.repeat_interleave(n_vars, n_vars * 2)

    rel_idx = torch.arange(len(probs), device=device) - lit_to_graph_offset
    is_neg_node = rel_idx >= v_counts  # shape: [2 * sum(n_vars)]

    # 2. Map each literal node to its corresponding positive literal's index
    # Negative literal at index k maps to k - n_vars (within its graph)
    pos_lit_idx = torch.where(is_neg_node,
                               torch.arange(len(probs), device=device) - v_counts,
                               torch.arange(len(probs), device=device))
    # pos_lit_idx[k] always points to the positive version of literal k

    # 3. Compute v_tilde per variable: v_tilde = 2*p - 1 in [-1, 1]
    # Only defined for positive literal nodes (one per variable)
    v_tilde_all = 2.0 * probs - 1.0  # shape: [2 * sum(n_vars)]

    # 4. For each edge, get s_ij and the correct v_tilde_i
    lit_node = edge_index[0]
    s_ij = torch.where(is_neg_node[lit_node], -1.0, 1.0)      # sign
    v_tilde_i = v_tilde_all[pos_lit_idx[lit_node]]             # v_tilde of the variable

    # 5. Compute clause sum: sum_i(s_ij * v_tilde_i)
    clause_sum = scatter(s_ij * v_tilde_i, edge_index[1],
                         dim=0, dim_size=n_clauses.sum(), reduce='sum')

    # 6. Clause sizes m_j
    m_j = scatter(torch.ones(lit_node.shape[0], dtype=torch.float, device=device),
                  edge_index[1], dim=0, dim_size=n_clauses.sum(), reduce='sum')

    # 7. DiffSAT loss per clause (relu to not incentivize overshooting beyond 1 satisfied literal)
    loss_j = ((clause_sum-1).pow(2) - (m_j - 1).pow(2)) / (4.0 * m_j)
    loss = torch.relu(loss_j)

    # 8. Entropy regularization to encourage confident assignments
    entropy_reg = 0.15 * (probs * (1.0 - probs)).mean()

    return loss, entropy_reg
    

def refine_training(model, train, val):
    """Runs a short unsupervised refinement phase on the MTL model using the provided training and validation datasets. Unsupervised loss needs to be manually changed in the MtlTrainer class (compute_unsupervised_loss) to switch between different unsupervised objectives. Can be easily integrated into the main training loop with automated hyperparameter tuning but kept separate for clarity and time constraints."""
    opts = {
        'd_model': 64,
        'T': 26,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'seed': 42
    }
    
    experiment = MTLParadigm(opts, train_dataset=train)
    
    experiment.run_unsupervised_refinement(
        checkpoint_filename=model,
        test_sat=f"{val}_SAT", 
        test_unsat=f"{val}_UNSAT"
    )