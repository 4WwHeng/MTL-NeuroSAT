from data.data_generation import generate_data_uniform, generate_test_data, generate_val_data
from data.data_preprocessing import split_training_data
from models.baseline.Baseline import run_baseline
from models.NeuroSAT.core.train import run_NeuroSAT_training
from models.NeuroSAT.core.test import load_and_test
from models.NeuroSAT.decoding.sat_lit import assignment_confidence, assignment_decoding
from models.NeuroSAT.decoding.sat_cls import sat_core_experiment
from models.NeuroSAT.decoding.votes import hammingd_experiment, backbone_experiment
from models.NeuroSAT.decoding.adm_training import save_labelled_lit_embedding, auto_hyperparameter_tuning_adm
from models.NeuroSAT.decoding.ctp_training import save_labelled_clauses_embedding, auto_hyperparameter_tuning_ctp
from models.NeuroSAT.decoding.adm_test import test_adm_pipeline
from models.NeuroSAT.decoding.ctp_test import test_batch_ctp_model
from models.NeuroSAT.MTL.mtl_train import auto_hyperparameter_tuning, run_mtl_staged_training
from models.NeuroSAT.MTL.refinement import refine_training
from models.NeuroSAT.MTL.mtl_test import load_and_test_mtl
from models.NeuroSAT.MTL.mtl_experiment import hammingd_experiment_mtl, evaluate_T_sweep
from solvers.walksat import weights_tuning, walksat_experiment
from solvers.pipeline import weights_tuning_ic, solver_pipeline, restart_filter_eval

from math import sqrt

def main():
    # 1. Data generation
    train_data = generate_data_uniform(20000, 10, 40)
    train_data = split_training_data(train_data)
    val_data = generate_val_data(2000, 40)
    test_data = {}
    for i in [20, 40, 100, 200]:
        test_data[i] = generate_test_data(2000, i)


    # 2. RF Baseline
    for i in [20, 40, 100, 200]:
        run_baseline(probe=False, train=train_data, test=test_data[i])
        run_baseline(probe=True, train=train_data, test=test_data[i])


    # 3. train and test NeuroSAT
    MODEL_NAME = run_NeuroSAT_training(train=train_data, val=val_data)
    load_and_test(MODEL_NAME)
    

    # 4. representation
    # literal embeddings
    assignment_decoding(MODEL_NAME, test_data=test_data[40])
    assignment_confidence(MODEL_NAME, test_data=test_data[40])

    # clause embeddings
    sat_core_experiment(MODEL_NAME, test_data=test_data[40])

    # votes
    backbone_experiment(MODEL_NAME, test_data=test_data[40])
    hammingd_experiment(MODEL_NAME, test_data=test_data[40])

    # supervised methods
    # save_labelled_lit_embedding(MODEL_NAME, test_data=train_data, save_name="train")
    # save_labelled_lit_embedding(MODEL_NAME, test_data=val_data, save_name="val")
    # save_labelled_clauses_embedding(MODEL_NAME, test_data=train_data, save_name="train")
    # save_labelled_clauses_embedding(MODEL_NAME, test_data=val_data, save_name="val")
    # auto_hyperparameter_tuning_ctp()
    # auto_hyperparameter_tuning_adm()
    test_adm_pipeline(f"{test_data[40]}_SAT", f"{test_data[40]}_UNSAT")
    test_batch_ctp_model(f"{test_data[40]}_SAT", f"{test_data[40]}_UNSAT")
    

    # 5. MTL 
    # auto_hyperparameter_tuning(train_data, val_data, "sequential")
    # auto_hyperparameter_tuning(train_data, val_data, "staged")
    # run_mtl_staged_training(train_data, val_data)
    # refine_training("MTL_Staged_stage3_best.pth", train_data, val_data)

    models = ["MTL_Naive.pth", "MTL_CDCL(pre).pth", "MTL_CDCL.pth", "MTL_Oracle(pre).pth", "MTL_Oracle.pth"]

    for m in models:
        load_and_test_mtl(m, f"{test_data[40]}_SAT", f"{test_data[40]}_UNSAT")
        hammingd_experiment_mtl(m, f"{test_data[40]}_SAT", f"{test_data[40]}_UNSAT")

    evaluate_T_sweep("MTL_CDCL.pth", f"{test_data[200]}_SAT", f"{test_data[200]}_UNSAT")
    evaluate_T_sweep("MTL_Oracle.pth", f"{test_data[200]}_SAT", f"{test_data[200]}_UNSAT")


    # 6. WalkSAT
    # weights_tuning(tuning_type="var_us")
    # weights_tuning(tuning_type="var_uc")
    # weights_tuning(tuning_type="clause")
    walksat_experiment(f"{test_data[40]}_SAT", f"{test_data[40]}_UNSAT")
    walksat_experiment(f"{test_data[100]}_SAT", f"{test_data[100]}_UNSAT")

    # 7. Incomplete Solver      
    # tuning_stats = weights_tuning_ic("40_SAT", "40_UNSAT")
    def flip(i):
        return int(250 * sqrt(i/40))
    models = ["NNs", "MTL_CDCL(pre).pth", "MTL_CDCL.pth", "MTL_Oracle(pre).pth", "MTL_Oracle.pth"]
    for m in models:
        solver_pipeline(m, f"{test_data[40]}_SAT", f"{test_data[40]}_UNSAT", mflips=flip(40))

    for i in [50, 100, 200]:
        for m in models:
            solver_pipeline(m, f"uf{i}", f"uuf{i}", mflips=flip(i))
    solver_pipeline(m, f"flat30", "", mflips=flip(90))


    # 8. Complete Solver
    restart_filter_eval("MTL_CDCL.pth", "uf200", "uuf200", mflips=flip(200), num_runs=5)
    restart_filter_eval("MTL_Oracle.pth", "uf200", "uuf200", mflips=flip(200), num_runs=5)

















