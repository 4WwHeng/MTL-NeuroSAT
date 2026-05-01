from solvers.walksat import walksat
import numpy as np
from pysat.solvers import Minisat22


def is_satisfiable(clause):
    with Minisat22(bootstrap_with=clause) as solver:
        return solver.solve()
    

def get_solver_stats(clauses):
    with Minisat22(bootstrap_with=clauses) as solver:
        is_sat = solver.solve()
        stats = solver.accum_stats()
        if stats is None:
            return is_sat, 0
        return is_sat, stats['propagations']


def get_close_assignment(assignment, clauses, n_vars):
    sol, flips = walksat(clauses, n_vars=n_vars, initial_assignment=assignment, max_flips=200000)
    # if sol is None:
    #     raise Exception("Solution not found")
    # else:
    #     return sol
    return sol


def count_satisfy(candidate, clauses):
    """Counts how many clauses are satisfied by a given boolean assignment."""
    sat_count = 0
    all_satisfied = True
    
    for clause in clauses:
        lits = np.array(clause)
        idxs = np.abs(lits) - 1
        
        # Check if the literal is satisfied by the current assignment
        is_satisfied = (candidate[idxs] == (lits > 0))
        
        if np.any(is_satisfied):
            sat_count += 1
        else:
            all_satisfied = False
            
    return sat_count, all_satisfied


def get_truth_assignment(clauses):
    """Finds a single satisfying assignment using Minisat."""
    with Minisat22(bootstrap_with=clauses) as solver:
        if solver.solve():
            literals = set(solver.get_model()) # type: ignore
            # Convert PySAT model to a binary [0, 1] array
            return [1 if (i + 1) in literals else 0 for i in range(len(literals))]
        return None


def find_backbone(clauses):
    """Finds the backbone of a formula by finding one valid model, and then aggressively trying to falsify each literal in that model.
    """
    sat_assignment = get_truth_assignment(clauses)
    if sat_assignment is None:
        return set()

    backbone = []
    for i, is_true in enumerate(sat_assignment):
        var = i + 1
        lit = var if is_true else -var
        
        # If forcing the opposite literal makes it UNSAT, it's a backbone variable
        if not is_satisfiable(clauses + [[-lit]]):
            backbone.append(lit)
            
    return set(backbone)


def get_unsat_cores(clauses, n_vars):
    """Finds the clauses that make up an UNSAT core using selector variables."""
    with Minisat22() as solver:
        selectors = []
        # Dynamically offset selectors safely above the maximum variable ID
        selector_offset = n_vars + 1 
        
        for i, clause in enumerate(clauses):
            s = selector_offset + i
            solver.add_clause(clause + [s])
            selectors.append(-s)
            
        sat = solver.solve(assumptions=selectors)
        
        if sat: 
            return [] # Should be UNSAT, but safeguard just in case
            
        core = solver.get_core() 
        if core is None:
            return [] # No core found, should not happen if unsat 
        core_clause_ids = [selectors.index(lit) for lit in core]
        
    return core_clause_ids


def clause_satlit_count(clauses, assignment):
    """Vectorized counter for satisfied literals per clause."""
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


def hamming_distance(pred, truth):
    """Vectorized calculation of percentage difference between two arrays."""
    return np.mean(np.array(pred) != np.array(truth))


def get_data_path():
    return "load/train_data"

def get_test_path():
    return "load/test_data"

def get_log_path():
    return "log"

def get_load_path():
    return "load/models"

def get_checkpoint_path():
    return "checkpoint"

def get_model_name():
    return "M-Trial4-T26-D64-L3.27e-05_epoch127_BEST.pth"

def get_batch_size():
    return 128



