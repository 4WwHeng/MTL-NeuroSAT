import random
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import is_satisfiable, get_solver_stats, get_data_path, get_test_path
from pathlib import Path

# Data generation and phase transition analysis for SR(n) distribution


# Generation logic

def generate_clause(n_var):
    """Generates a single random clause based on the SR(n) distribution."""
    geo_sample = np.random.geometric(p=0.4)
    bernoulli_sample = np.random.binomial(n=1, p=0.7)
    k = 1 + geo_sample + bernoulli_sample
    variable = random.sample(range(1, n_var + 1), min(k, n_var))
    clause = [v * random.choice([-1, 1]) for v in variable]
    return clause


def generate_sr_pair(n_var):
    """Generates a pair of SAT/UNSAT instance in phase transition."""
    clauses = []
    while True:
        new_clause = generate_clause(n_var)
        clauses.append(new_clause)

        if not is_satisfiable(clauses):
            un_sat = clauses[:-1] + [new_clause[::]]
            clauses[-1][random.randint(0, len(new_clause) - 1)] *= -1
            sat = clauses

            return un_sat, sat


def dimacs_format(clauses, n_var):
    """Converts clauses to DIMACS string."""
    lines = [f"p cnf {n_var} {len(clauses)}"]
    lines.extend(" ".join(map(str, clause)) + " 0" for clause in clauses)
    return "\n".join(lines) + "\n"


# File I/O 

def _write_cnf_file(path: Path, clauses: list[list[int]], n_var: int) -> None:
    """Helper method to write DIMACS data to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dimacs_format(clauses, n_var))


def generate_val_data(n_data: int, n_var: int) -> str:
    """Generates standard SR(n) validation data."""
    base_dir = Path(get_test_path())
    folder_sat = base_dir / f"Val_{n_var}_SAT"
    folder_unsat = base_dir / f"Val_{n_var}_UNSAT"
    
    for i in range(1, n_data + 1):
        unsat, sat = generate_sr_pair(n_var)
        _write_cnf_file(folder_unsat / f"{i:04d}_UNSAT.cnf", unsat, n_var)
        _write_cnf_file(folder_sat / f"{i:04d}_SAT.cnf", sat, n_var)

    return f"Val_{n_var}"


def generate_test_data(n_data: int, n_var: int) -> str:
    """Generates SR(n) testing data."""
    base_dir = Path(get_test_path())
    folder_sat = base_dir / f"Test_{n_var}_SAT"
    folder_unsat = base_dir / f"Test_{n_var}_UNSAT"
    
    for i in range(1, n_data + 1):
        unsat, sat = generate_sr_pair(n_var)
        _write_cnf_file(folder_unsat / f"{i:04d}_UNSAT.cnf", unsat, n_var)
        _write_cnf_file(folder_sat / f"{i:04d}_SAT.cnf", sat, n_var)

    return f"Test_{n_var}"


def generate_data_uniform(n_data: int, lower_bound: int, upper_bound: int) -> str:
    """Generates uniform distribution training data."""
    base_dir = Path(get_data_path())
    folder = base_dir / f"SR_Uniform_{lower_bound}-{upper_bound}_Dataset"
    
    for i in range(1, n_data + 1):
        current_n_var = random.randint(lower_bound, upper_bound)
        unsat, sat = generate_sr_pair(current_n_var)
        _write_cnf_file(folder / f"SR{current_n_var}_id{i:04d}_UNSAT.cnf", unsat, current_n_var)
        _write_cnf_file(folder / f"SR{current_n_var}_id{i:04d}_SAT.cnf", sat, current_n_var)

    return f"SR_Uniform_{lower_bound}-{upper_bound}_Dataset"


# Phase transition analysis and plotting

def frac_unsat() -> None:
    N = 50
    r, frac_unsat_list = [], []
    for cv in range(20, 81, 2):
        clause_var_ratio = cv / 10.0
        M = int(clause_var_ratio * N)
        
        unsat = sum(1 for _ in range(1000) if not is_satisfiable([generate_clause(N) for _ in range(M)]))
        
        r.append(clause_var_ratio)
        frac_unsat_list.append(unsat / 1000.0)
        
    print(r)
    print(frac_unsat_list)

def num_DP_calls() -> None:
    N = 50
    r = []
    for cv in range(20, 81, 2):
        clause_var_ratio = cv / 10.0
        M = int(clause_var_ratio * N)
        
        calls = []
        for _ in range(1000):
            clauses = [generate_clause(N) for _ in range(M)]
            _, call_count = get_solver_stats(clauses)
            calls.append(call_count)
            
        calls.sort()
        median = (calls[499] + calls[500]) / 2.0
        r.append(median)
    print(r)

def plot_phase_transition() -> None:
    cv_range = [x / 10.0 for x in range(20, 81, 2)]

    # phase transition data points
    n20 = [0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.007, 0.005, 0.008, 0.011, 0.02, 0.033, 0.05, 0.072, 0.12, 0.159, 0.219, 0.288, 0.34, 0.404, 0.503, 0.563, 0.622, 0.728, 0.774, 0.801, 0.884, 0.879, 0.903, 0.921, 0.948]
    n40 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.0, 0.002, 0.008, 0.007, 0.016, 0.039, 0.049, 0.106, 0.156, 0.25, 0.357, 0.467, 0.592, 0.652, 0.738, 0.82, 0.886, 0.921, 0.955, 0.981, 0.984, 0.995, 0.998, 0.998]
    n50 = [0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.0, 0.0, 0.003, 0.001, 0.006, 0.01, 0.031, 0.038, 0.081, 0.185, 0.246, 0.368, 0.45, 0.622, 0.736, 0.793, 0.9, 0.929, 0.962, 0.982, 0.99, 0.995, 0.997, 0.999, 0.999]

    plt.figure(figsize=(10, 6))
    plt.plot(cv_range, n20, 'o-', label='N = 20', markersize=4)
    plt.plot(cv_range, n40, 's-', label='N = 40', markersize=4)
    plt.plot(cv_range, n50, '^-', label='N = 50', markersize=4)

    # SR(n) m/n threshold
    plt.axvline(x=5.76, color='red', linestyle='--', label='SR(40) Threshold (~5.76)')

    plt.title('Phase Transition: Fraction of Unsatisfiable Formulas vs. Clause/Variable Ratio')
    plt.xlabel(r'Clause-to-Variable Ratio ($\alpha = M/N$)')
    plt.ylabel('Fraction of Unsatisfiable Formulas')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    plt.savefig('phase_transition_plot2.jpg')


if __name__ == "__main__":
    plot_phase_transition()