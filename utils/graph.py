# plotting

import matplotlib.pyplot as plt
import numpy as np
import os
from utils.utils import get_log_path

LOG_PATH = get_log_path()

def plot_heavy_tail(**datasets):
    plt.figure(figsize=(10, 7))
    
    for label, flips in datasets.items():
        # Sort data
        sorted_data = np.sort(flips)
        
        # Calculate CCDF: y = 1 - (index / n)
        y = 1.0 - np.arange(len(sorted_data)) / len(sorted_data)
        
        plt.loglog(sorted_data, y, marker='.', linestyle='none', alpha=0.6, label=label)
    
    plt.xlabel('Flips to Solve (Log Scale)')
    plt.ylabel('Probability of remaining unsolved (Log Scale)')
    plt.title('Tail Distribution Analysis (Log-Log CCDF)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()

    filename = os.path.join(LOG_PATH, "heavy_tail_dynamic.jpg")
    plt.savefig(filename)
    plt.show()


def plot_cactus(**datasets):
    plt.figure(figsize=(12, 7))
    
    for label, flips in datasets.items():
        sorted_data = np.sort(flips)
        x_axis = np.arange(1, len(sorted_data) + 1)
        
        plt.semilogy(x_axis, sorted_data, label=label, linewidth=2)
        
        zero_shots = np.sum(sorted_data == 0)
        

    plt.xlabel('Number of Instances Solved')
    plt.ylabel('Flips Required (Log Scale)')
    plt.title('Cactus Plot: Solver Efficiency Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)


    # Annotate the Zero-Shot area
    plt.axvline(x=zero_shots, color='red', linestyle='--', alpha=0.5)
    plt.text(zero_shots/2, 1, f"Zero-Shot Region\n({zero_shots} instances)", color='red', ha='center')
    
    filename = os.path.join(LOG_PATH, "cactus_plot_dynamic.jpg")
    plt.savefig(filename)
    plt.show()