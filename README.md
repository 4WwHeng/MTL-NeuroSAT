# NeuroSAT: Latent Representations and Multi-Task Learning for SAT Solving

## Original Aims of the Project
This dissertation aims to provide a clearer understanding of how NeuroSAT behaves on SAT: whether it learns meaningful structural representations, what those representations encode, and whether they can guide decisions in classical solvers. 

During the reimplementation of NeuroSAT under tighter computational constraints, the model displayed over-smoothing behaviour at reduced embedding dimensions, a behaviour not reported in the original paper. This motivated the investigation of whether multi-task learning using more informative supervised signals can mitigate the issue.

## Work Completed
This dissertation makes four primary contributions:
1. A PyTorch reimplementation of NeuroSAT.
2. A detailed analysis of latent representations of NeuroSAT, revealing previously unreported correlation with clause brittleness.
3. A multi-task learning (MTL) architecture that recovers NeuroSAT’s generalization behaviour using ≤ 5% of original training data and half the embedding dimensions.
4. A demonstration of the potential of NeuroSAT as an effective filter in a neuro-symbolic pipeline with state-of-the-art solvers.

## Repository Structure

```text
project/
 ┣ load/
 ┃ ┣ models/             # Saved model .pt files (See Pre-trained Models section)
 ┃ ┣ train_data/         # Saved training dataset
 ┃ ┗ test_data/          # Saved test dataset
 ┣ checkpoint/           # Training checkpoints
 ┣ log/                  # Execution and training logs
 ┣ data/
 ┃ ┣ data_generation.py  # SR(n) generation pipeline
 ┃ ┗ data_preprocessing.py # CNF parsing and conversion
 ┣ models/
 ┃ ┣ baseline/           # Feature-based baseline
 ┃ ┗ neurosat/
 ┃   ┣ core/             # Core NeuroSAT implementation
 ┃   ┣ decoding/         # Representation learning
 ┃   ┗ mtl/              # Multi-Task Learning
 ┣ solvers/
 ┃ ┣ walksat.py          # Guided WalkSAT
 ┃ ┣ ranger.py           # Guided RANGER
 ┃ ┗ pipeline.py         # End-to-end solver pipeline
 ┣ utils/
 ┃ ┣ utils.py            # Common utility functions
 ┃ ┗ graph.py            # Graph plotting functions
 ┣ notebooks/            
 ┃ ┣ phase1.ipynb        # Complete Notebook for phase 1 
 ┃ ┗ phase2.ipynb        # Complete Notebook for phase 2 
 ┗ run.py                # Main execution script for experiments
```

## Requirements
The code is written in Python 3 and requires the following dependencies:
- PyTorch & PyTorch Geometric
- PySAT
- scikit-learn
- Optuna

## Pre-trained model and Datasets
To comply with the submission guidelines, the trained PyTorch weights (.pt files) and the generated datasets are not included directly in this source code bundle. Only the `data`, `models`, `solvers`, `utils` and `run.py` are attached in the zip file.

To evaluate the model without retraining from scratch, please download the pre-trained weights from the `models` folder at the following repository:

**[MTL-NeuroSAT GitHub Repository](https://github.com/4WwHeng/MTL-NeuroSAT)**

**Instructions for loading models:**
1. Download the `[model_name].pt` file from the link above.
2. Place the downloaded file into the `load/models/` directory.
3. Alternatively: change the model load path directly inside the `utils/utils.py` file.

The training and testing datasets are automatically generated when executing `run.py`. Additional benchmark datasets can be downloaded from the SATLIB website and saved in the `load/test_data/` directory.

## Usage
Hardware Constraint: A GPU is strictly required to execute this code, including for the inference phases of the NeuroSAT model.

The primary entry point for all experiments is the execution script:

```bash
python run.py
```

### Important execution note for Evaluation:
`run.py` is configured to execute the entire pipeline sequentially (Data Generation → Baseline Processing → NeuroSAT Training → Representation Experiments → MTL Evaluation → Guided Solvers Pipeline).

By default, all model training steps are commented out, and evaluation uses the pre-trained models mentioned above. If you wish to train your own models, the respective lines must be uncommented. After training completes, the best model's checkpoint must be manually moved from the `checkpoint/` directory to `load/models/` for downstream analysis. 


