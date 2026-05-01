import pickle
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data.data_preprocessing import read_data
from feature_engineering import generate_full_feature_vector
from utils.utils import get_load_path


# Global Configurations

dpll_features_list = ['prop_1', 'prop_4', 'prop_16', 'prop_64', 'prop_256', 'log_nodes', 'mean_depth_conflict']
saps_features_list = ['mean_steps_best', 'median_steps_best', 'ninetieth_steps_best', 'tenth_steps_best', 'n_unsat_clause_mean', 'avg_improvements_mean', 'frac_first_min_mean']

non_probe_features = ['cv_ratio', 'fraction_binary', 'fraction_horn', 'fraction_ternary', 'horn_entropy', 'horn_max', 'horn_mean', 'horn_min', 'horn_vari_coef', 'lit_entropy', 'lit_mean', 'lit_vari_coef', 'n_clause', 'n_var', 'var_entropy', 'var_max', 'var_mean', 'var_min', 'var_vari_coef', 'vcgc_entropy', 'vcgc_max', 'vcgc_mean', 'vcgc_min', 'vcgc_vari_coef', 'vcgv_entropy', 'vcgv_max', 'vcgv_mean', 'vcgv_min', 'vcgv_vari_coef', 'vg_max', 'vg_mean', 'vg_min', 'vg_vari_coef']


# Core Pipeline Functions

def process_dataset(dataset, probe):
    """Extracts features and labels using optimized list comprehensions."""
    if not dataset:
        raise ValueError("Dataset is empty. Cannot process features.")

    print(f"Extracting features for {len(dataset)} instances...")
    
    # dataset item structure from read_data: (clauses, n_var, label)
    X_features = [generate_full_feature_vector(item[0], item[1], probe) for item in dataset]
    y_labels = [item[2] for item in dataset]

    X = np.array(X_features, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)

    print(f"Data processing complete. Feature dimensions: {X.shape[1]}.")
    return X, y


def evaluate_model_metrics(y_true, y_pred, phase="Test Set"):
    """Function to calculate and print evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("-" * 50)
    print(f"Model Performance on {phase}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['UNSAT (0)', 'SAT (1)']))
    print("-" * 50)


def train_and_evaluate_baseline(X, y, model_name, features_name):
    print("-" * 50)
    print("Starting Baseline Model Training (Random Forest with Grid Search)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set size: {len(X_train)} | Testing set size: {len(X_test)}")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }

    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    print("\nRunning Grid Search...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\n" + "=" * 30)
    print(f"BEST PARAMETERS FOUND: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    print("=" * 30 + "\n")

    y_pred = best_model.predict(X_test)
    evaluate_model_metrics(y_test, y_pred, phase="Validation Split")

    importance = best_model.feature_importances_
    if len(features_name) == importance.shape[0]:
        sorted_indices = np.argsort(importance)[::-1]
        print("\nRANKED MOST IMPORTANT FEATURES:")
        for i in sorted_indices:
            print(f"  {features_name[i]:<40}: {importance[i]:.6f}")
    else:
        print("\n(Warning: Feature names list length does not match feature vector dimension.)")

    save_model(best_model, model_name)


# Serialization 

def save_model(model: RandomForestClassifier, filename: str) -> None:
    """Saves the model to the designated load directory."""
    save_dir = Path(get_load_path())
    save_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = save_dir / filename
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"\nModel successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(filename):
    """Loads a model from the designated load directory."""
    file_path = Path(get_load_path()) / filename
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print(f"\nModel successfully loaded from {file_path}")
        return model
    except FileNotFoundError:
        print(f"\nModel file not found at {file_path}. Returning None.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Main Execution

def run_baseline(model_name="", probe=True, train="SR_Uniform_10-40_Dataset", test="Test_40"):
    if model_name:
        MODEL_NAME = model_name
    else:
        MODEL_NAME = f"{train}_baseline" + "_probe" if probe else "_noprobe"

    if probe:
        FEATURE_NAMES = non_probe_features + dpll_features_list + saps_features_list
    else:
        FEATURE_NAMES = non_probe_features

    # 1. Load Model for Testing
    loaded_model = load_model(MODEL_NAME)
    if loaded_model is None:
        print("No saved model found. Training new model.")

        # 2. Train Model
        data = read_data(train, is_training=True)
        X_data, y_labels = process_dataset(data, probe)
        train_and_evaluate_baseline(X_data, y_labels, MODEL_NAME, FEATURE_NAMES)

        loaded_model = load_model(MODEL_NAME)

    if loaded_model is None:
        raise Exception
    
    # 3. Test on Unseen Data
    data_sat = read_data(f"{test}_SAT", is_training=False, fixed_label=1)
    data_unsat = read_data(f"{test}_UNSAT", is_training=False, fixed_label=0)
    test_data = data_sat + data_unsat

    t = {}
    
    # Feature Extraction Timing
    t0 = time.perf_counter()
    X_test, y_test = process_dataset(test_data, probe)
    t['probe'] = time.perf_counter() - t0

    # Inference Timing
    t0 = time.perf_counter()
    y_pred = loaded_model.predict(X_test)
    t['inference'] = time.perf_counter() - t0
    
    t['total'] = t['probe'] + t['inference']

    print(f"Timing Stats -> Extraction: {t['probe']:.4f}s | Inference: {t['inference']:.4f}s | Total: {t['total']:.4f}s")
    
    # 4. Final Evaluation
    evaluate_model_metrics(y_test, y_pred, phase="Unseen Test Data")