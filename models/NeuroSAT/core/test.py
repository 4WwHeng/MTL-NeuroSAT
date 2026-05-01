import torch
import os
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from data_loader import SATDataset
from Neurosat import NeuroSAT
from utils.utils import get_load_path

LOAD_PATH = get_load_path()

# function for testing a trained model on a test set with assignments decoding (k-means).

def load_and_test(model_name, test_data="Test_40"):
    # 1. Match Training Hyperparameters
    opts = {
        'd_model': 64,         
        'T': 26,                
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'seed': 42
    }
    BATCH_SIZE = 128

    # 2. Setup the Test Data
    print("Loading test datasets...")
    
    # Load SAT and UNSAT datasets separately, then merge them
    test_sat_data = SATDataset(data_file=f"{test_data}_SAT", is_training=False, fixed_label=1)
    test_unsat_data = SATDataset(data_file=f"{test_data}_UNSAT", is_training=False, fixed_label=0)
    
    merged_test_data = ConcatDataset([test_sat_data, test_unsat_data])
    test_loader = DataLoader(merged_test_data, batch_size=BATCH_SIZE, shuffle=False) # type: ignore

    # 3. Initialize and Load the Model
    print(f"Initializing NeuroSAT on {opts['device']}...")
    solver = NeuroSAT(opts)

    print(f"Loading weights from {model_name}...")
    load_model_path = os.path.join(LOAD_PATH, f"{model_name}.pth")
    solver.restore(load_model_path)

    # 4. Run Evaluation
    print("\nRunning test evaluation...")
    accuracy, conf_matrix = solver.test(test_loader)
    
    print("\n--- Test Results ---")
    print(f"Overall Accuracy:  {accuracy * 100:.2f}%")
    print("Confusion Matrix (TN, FP | FN, TP):")
    print(conf_matrix)

    # 5. Extracting an actual solution
    print("\nAttempting to decode assignments for the entire test set...")
    
    total_solved = 0
    total_actual_sat = 0
    total_problems = len(test_loader.dataset) # type: ignore

    # Iterate through every batch in the test set
    for batch_idx, batch in enumerate(test_loader):
        # Count how many problems in this batch are actually SAT
        actual_sat_in_batch = int(batch.y.sum().item())
        total_actual_sat += actual_sat_in_batch
        
        # Extract solutions
        solutions = solver.find_solutions(batch)
        
        # Count how many valid satisfying assignments we found
        solved_in_batch = sum(1 for s in solutions if s['satisfied'])
        total_solved += solved_in_batch

    # Final Extraction Metrics
    print("\n--- Final Decoding Results ---")
    print(f"Total Problems Tested: {total_problems}")
    print(f"Total Actual SAT Problems: {total_actual_sat}")
    print(f"Successfully Decoded Solutions: {total_solved}")
    
    if total_actual_sat > 0:
        decode_rate = (total_solved / total_actual_sat) * 100
        print(f"Decoding Success Rate: {decode_rate:.2f}%")


if __name__ == "__main__":
    load_and_test("M-Trial4-T26-D64-L3.27e-05_epoch127_BEST.pth")
