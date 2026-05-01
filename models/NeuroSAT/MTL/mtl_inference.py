import torch
from torch.utils.data import ConcatDataset
from core.data_loader import SATDataset
from mtl_model import MtlNeuroSAT
from mtl_trainer import MtlTrainer

# MTL Inference Function similar to NN_inference but adapted for the MTL model and its specific outputs. Uses seperate test_data_sat and test_data_unsat as we use outside test datasets for evaluation.


def mtl_inference(checkpoint_filename, test_data_sat='Test_40_SAT', test_data_unsat='Test_40_UNSAT', T_val=26):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing inference on {device}...")

    # 1. Create the blank model and the trainer
    model = MtlNeuroSAT(d_model=64, T=T_val)
    trainer = MtlTrainer(model, device)

    # 2. Use the built-in restore method
    epoch_trained = trainer.restore(checkpoint_filename)
    
    if epoch_trained is None:
        print("Inference aborted: Checkpoint not found.")
        raise Exception("Checkpoint not found")

    print(f"Model successfully loaded! (Trained for {epoch_trained} epochs)")
    print(f"Evaluating on {test_data_sat} and {test_data_unsat}...\n")
    test_sat_data = SATDataset(data_file=test_data_sat, is_training=False, fixed_label=1)
    test_unsat_data = SATDataset(data_file=test_data_unsat, is_training=False, fixed_label=0)
    
    merged_test_data = ConcatDataset([test_sat_data, test_unsat_data])

    # 3. Run the test inference and get all outputs in one go
    result = trainer.inference(test_data_sat, test_data_unsat)
    print(f"Extraction complete! Latency: {result['latency']:.4f} ms/sample")

    # Get the exact number of literals and clauses for every graph in the dataset
    n_lits_per_graph = [data.n_vars.item() * 2 for data in merged_test_data] #type: ignore
    n_clauses_per_graph = [data.n_clauses.item() for data in merged_test_data] #type: ignore
    
    # Split the massive tensors instantly
    lit_embs    = result['lit_embs']
    clause_embs = result['clause_embs']
    adm_probs   = result['adm_probs']
    ctp_probs   = result['ctp_probs']
    split_lit_emb    = [e.numpy() for e in torch.split(lit_embs,    n_lits_per_graph)]
    split_clause_emb = [e.numpy() for e in torch.split(clause_embs, n_clauses_per_graph)]
    split_adm_prob   = [p.numpy() for p in torch.split(adm_probs,   n_lits_per_graph)]
    split_ctp_prob   = [p.numpy() for p in torch.split(ctp_probs,   n_clauses_per_graph)]

    # 4. Return results
    return result["graph_votes"], split_lit_emb, split_clause_emb, split_adm_prob, split_ctp_prob, result["latency"]


