import numpy as np
import time
from sklearn.metrics import classification_report
import torch
import random
from utils.utils import is_satisfiable, count_satisfy
from models.NeuroSAT.decoding.neurosat_s import NNs_inference
from data.data_preprocessing import read_data
from models.NeuroSAT.MTL.mtl_inference import mtl_inference
from solvers.walksat import guided_walksat
from pysat.solvers import Cadical153


# Neuro-symbolic incomplete solver and complete solver


def solver_pipeline(checkpoint_filename, test_data_sat="Test_40_SAT", test_data_unsat="Test_40_UNSAT", mflips=280):
    """Incomplete solver pipeline using NN hueristics to guide WalkSAT"""
    print("Running Inference for Solver Pipeline...")
    
    # NN heuristics
    if checkpoint_filename == "NNs":
        graph_votes, _, _, split_adm_prob, _, _, latency = NNs_inference(
            test_data_sat, test_data_unsat
        )
    else:
        graph_votes, _, _, split_adm_prob, _, latency = mtl_inference(
            checkpoint_filename, test_data_sat, test_data_unsat
        )

    data_sat = read_data(test_data_sat, is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    true_labels = []
    pred_labels = []

    walksat_count = 0
    rescued_count = 0
    reject_count = 0
    avg_flips = []

    # Only time the actual WalkSAT search loop, since NN time is tracked via latency
    start_search_time = time.perf_counter()

    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        true_labels.append(1 if is_sat_ground_truth else 0)

        vote = graph_votes[i]
        candidate_prob = split_adm_prob[i]

        candidate = np.argmax(candidate_prob, axis=1)
        _, direct_solved = count_satisfy(candidate, clauses) 
        
        if direct_solved:
            pred_labels.append(1)
            
        elif vote > 0.5:
            walksat_count += 1
            
            sol, flips = guided_walksat(clauses, n_vars=n_vars, max_flips=mflips, initial_assignment=candidate)
            avg_flips.append(flips)
            
            if sol is not None:
                pred_labels.append(1)
                rescued_count += 1
            else:
                pred_labels.append(0)
                if not is_sat_ground_truth:
                    reject_count += 1
        else:
            pred_labels.append(0)
            
    search_time = time.perf_counter() - start_search_time
    
    # Calculate exact total times
    inference_time_s = (latency * len(data)) / 1000
    total_time_s = inference_time_s + search_time

    # Print Final Results
    target_names = ['Label 0 (Unsatisfiable)', 'Label 1 (Satisfiable)']
    print("\n" + "="*60)
    print(f"SOLVER PIPELINE RESULTS (Max Flips: {mflips})")
    print("="*60)
    
    print(classification_report(true_labels, pred_labels, target_names=target_names))
    print("-" * 60)
    
    print(f"Total time for {len(data)} instances:   {total_time_s:.3f} s")
    print(f"Avg latency per instance:         {(total_time_s * 1000) / len(data):.2f} ms")
    print(f"Total pure NN inference time:     {inference_time_s:.3f} s")
    print("-" * 60)
    
    print(f"Total instances solved:                 {np.sum(pred_labels)}")
    print(f"No. of Guided WalkSAT attempted:        {walksat_count}")
    print(f"No. of rescued Satisfiable instances:   {rescued_count}")
    print(f"No. of rejected Unsatisfiable instances:{reject_count}")
    
    mean_flips = np.mean(avg_flips) if avg_flips else 0
    print(f"Mean number of WalkSAT flips:           {mean_flips:.2f}")

    return np.sum(pred_labels), mean_flips, search_time
    

def weights_tuning_ic(checkpoint_filename, test_sat="40_SAT", test_unsat="40_UNSAT"):
    """Tuning the max_flips parameter for the incomplete solver pipeline using three different random seeds for robustness."""
    # Warmup run to initialize CUDA and caches
    print("Performing warmup run...")
    _, _, _ = solver_pipeline(checkpoint_filename, test_data_sat=test_sat, test_data_unsat=test_unsat, mflips=10)

    seeds = [25, 50, 75]
    flip_range = list(range(50, 501, 50))
    
    all_solved = []
    all_flips = []
    all_times = [] 
    
    print("\n" + "="*50)
    print("STARTING MAX FLIPS TUNING")
    print("="*50)

    for seed in seeds:
        print(f"\n--- Running Seed: {seed} ---")
        
        # Enforce reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        seed_solved = []
        seed_flips = []
        seed_times = []
        
        for mflips in flip_range:
            print(f"Testing Max Flips: {mflips}")
            s, f, t = solver_pipeline(checkpoint_filename, test_data_sat=test_sat, test_data_unsat=test_unsat, mflips=mflips)
            
            seed_solved.append(s)
            seed_flips.append(f)
            seed_times.append(t)

        all_solved.append(seed_solved)
        all_flips.append(seed_flips)
        all_times.append(seed_times)

    solved_arr = np.array(all_solved)
    flips_arr = np.array(all_flips)
    time_arr = np.array(all_times)

    solved_mean = np.mean(solved_arr, axis=0)
    solved_std = np.std(solved_arr, axis=0)
    
    flips_mean = np.mean(flips_arr, axis=0)
    flips_std = np.std(flips_arr, axis=0)
    
    time_mean = np.mean(time_arr, axis=0)
    time_std = np.std(time_arr, axis=0)

    # Print Final Results
    print("\n" + "="*50)
    print("TUNING RESULTS SUMMARY")
    print("="*50)
    print(f"Tested Flip Ranges: {flip_range}\n")
    
    print(f"Solved Mean: {np.round(solved_mean, 2)}")
    print(f"Solved Std:  {np.round(solved_std, 2)}\n")
    
    print(f"Flips Mean:  {np.round(flips_mean, 2)}")
    print(f"Flips Std:   {np.round(flips_std, 2)}\n")
    
    print(f"Time Mean:   {np.round(time_mean, 3)}")
    print(f"Time Std:    {np.round(time_std, 3)}")
    
    return {
        "flip_range": flip_range,
        "solved_mean": solved_mean, "solved_std": solved_std,
        "flips_mean": flips_mean, "flips_std": flips_std,
        "time_mean": time_mean, "time_std": time_std
    }


def is_satisfiable(clause):
    with Cadical153(bootstrap_with=clause) as solver:
        return solver.solve()


def benchmark(test_data_sat="Test_100_SAT", test_data_unsat="Test_100_UNSAT"):
    """Benchmarking the CaDiCaL solver on the provided test datasets."""
    data_sat = read_data(test_data_sat, is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    true_labels = []
    pred_labels = []

    start_time = time.perf_counter()
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        true_labels.append(1 if is_sat_ground_truth else 0)
        pred_labels.append(1 if is_satisfiable(clauses) else 0)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} / {len(data)}...")

    total_time = time.perf_counter() - start_time

    target_names = ['Label 0 (Unsatisfiable)', 'Label 1 (Satisfiable)']
    
    print("\n" + "="*60)
    print("CADICAL BENCHMARK RESULTS")
    print("="*60)
    print(classification_report(true_labels, pred_labels, target_names=target_names))
    print("-" * 60)
    print(f"Total time for {len(data)} instances:   {total_time:.3f} s")
    print(f"Avg latency per instance:         {(total_time * 1000) / len(data):.2f} ms")
    print("="*60)


def sat_filter_simulated_parallel(checkpoint_filename, test_data_sat="Test_40_SAT", test_data_unsat="Test_40_UNSAT", mflips=100):
    """Neuro-symbolic Complete Solver pipeline both sequential and parallel. Simulates a parallel solver pipeline where a NN-guided WalkSAT and a pure CaDiCaL are run in parallel, and we take the result of whichever finishes first."""

    print("Running initial NN Inference...")
    graph_votes, _, _, split_adm_prob, _, latency = mtl_inference(
        checkpoint_filename, test_data_sat, test_data_unsat, T_val=45
    )
    
    data_sat = read_data(test_data_sat, is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    # Convert latency to seconds for the systems calculations.
    avg_inference_time = latency / 1000.0  
    total_inference_time = avg_inference_time * len(data)

    true_labels = []
    pred_labels = []

    walksat_count = 0
    rescued_count = 0
    direct_count = 0
    avg_flips = []

    pure_cadical_time = 0.0
    
    # Sequential starts with the cost of running inference on the whole dataset
    sequential_pipeline_time = total_inference_time 
    
    # Parallel starts at 0, time is accumulated instance-by-instance in the race
    simulated_parallel_time = 0.0

    # Evaluation loop
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        true_labels.append(1 if is_sat_ground_truth else 0)

        vote = graph_votes[i]
        candidate_prob = split_adm_prob[i]
        candidate = np.argmax(candidate_prob, axis=1)
        
        _, direct_solved = count_satisfy(candidate, clauses) 

        # TIME A: PURE CADICAL
        t0_cad = time.perf_counter()
        cadical_result = is_satisfiable(clauses)  
        t_cadical = time.perf_counter() - t0_cad
        pure_cadical_time += t_cadical

        # TIME B: HYBRID WALKSAT
        t_walksat = 0.0
        walksat_solved = False

        if direct_solved:
            walksat_solved = True
            direct_count += 1
            pred_labels.append(1)
            
        elif vote > 0.5:
            walksat_count += 1
            
            t0_ws = time.perf_counter()
            sol, flips = guided_walksat(clauses, n_vars=n_vars, max_flips=mflips, initial_assignment=candidate)
            t_walksat = time.perf_counter() - t0_ws
            
            avg_flips.append(flips)
            
            if sol is not None:
                walksat_solved = True
                rescued_count += 1
                pred_labels.append(1)
        
        # Fallback using CaDiCaL if NN + WalkSAT failed to prove SAT 
        if not walksat_solved:
            pred_labels.append(1 if cadical_result else 0)

        # CALCULATE SYSTEMS METRICS
        # Sequential: We pay for WalkSAT attempt. If it failed, we ALSO pay for CaDiCaL.
        sequential_pipeline_time += t_walksat
        if not walksat_solved:
            sequential_pipeline_time += t_cadical

        # Parallel: CaDiCaL and (NN+WalkSAT) race each other.
        if walksat_solved:
            # Time taken is whichever thread finished first.
            simulated_parallel_time += min(t_cadical, avg_inference_time + t_walksat)
        else:
            # WalkSAT failed quietly. CaDiCaL finishes at its normal time.
            # We assume parallel threads launch simultaneously, so we only use CaDiCaL time.
            simulated_parallel_time += t_cadical

    # Print Final Results
    target_names = ['Label 0 (Unsatisfiable)', 'Label 1 (Satisfiable)']
    print(classification_report(true_labels, pred_labels, target_names=target_names))
    
    print("\n" + "="*50)
    print("      SYSTEMS TIMING RESULTS")
    print("="*50)
    print(f"Total NN Inference Time:  {total_inference_time:.3f} s")
    print("-" * 50)
    print(f"1. Pure CaDiCaL Time:     {pure_cadical_time:.3f} s")
    print(f"2. Sequential Pipeline:   {sequential_pipeline_time:.3f} s")
    print(f"3. Simulated Parallel:    {simulated_parallel_time:.3f} s  <-- (VBS Limit)")
    print("="*50)
    
    print(f"Total instances evaluated:     {len(data)}")
    print(f"Total instances correctly classified: {np.sum(np.array(true_labels) == np.array(pred_labels))}")
    print(f"Direct Solved by NN:           {direct_count}")
    print(f"WalkSAT attempted:             {walksat_count}")
    print(f"WalkSAT successfully rescued:  {rescued_count}")
    
    mean_flips = np.mean(avg_flips) if avg_flips else 0
    print(f"Mean WalkSAT flips (when attempted): {mean_flips:.1f}")
    
    return total_inference_time, pure_cadical_time, sequential_pipeline_time, simulated_parallel_time, direct_count, walksat_count, rescued_count, mean_flips


def restart_filter_eval(checkpoint_filename, test_data_sat="Test_40_SAT", test_data_unsat="Test_40_UNSAT", mflips=100, num_runs=5):
    """Complete Solver evaluation averaged over multiple runs to account for variability in WalkSAT performance."""
    nn_inf = []
    cadical = []
    sequential = []
    parallel = []
    dc = []
    wa = []
    wr = []
    fl = []
    
    print(f"Starting {num_runs} evaluation runs.")
    
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        nn_t, pct, seq_t, par_t, d_count, w_count, r_count, m_flips_val = sat_filter_simulated_parallel(checkpoint_filename, test_data_sat, test_data_unsat, mflips)
        
        nn_inf.append(nn_t)
        cadical.append(pct)
        sequential.append(seq_t)
        parallel.append(par_t)
        dc.append(d_count)
        wa.append(w_count)
        wr.append(r_count)
        fl.append(m_flips_val)

    # Print Final Results
    print("\n" + "="*55)
    print(f"      AGGREGATED SYSTEMS TIMING RESULTS (N={num_runs})")
    print("="*55)
    print(f"0. NN Inference Time:     {np.mean(nn_inf):.3f} ± {np.std(nn_inf):.3f} s")
    print(f"1. Pure CaDiCaL Time:     {np.mean(cadical):.3f} ± {np.std(cadical):.3f} s")
    print(f"2. Sequential Pipeline:   {np.mean(sequential):.3f} ± {np.std(sequential):.3f} s")
    print(f"3. Simulated Parallel:    {np.mean(parallel):.3f} ± {np.std(parallel):.3f} s")
    print("-" * 55)
    print("            AGGREGATED SEARCH STATISTICS")
    print("-" * 55)
    print(f"Mean Direct solved:             {np.mean(dc):.1f} ± {np.std(dc):.1f}")
    print(f"Mean WalkSAT attempted:         {np.mean(wa):.1f} ± {np.std(wa):.1f}")
    print(f"Mean WalkSAT rescues:           {np.mean(wr):.1f} ± {np.std(wr):.1f}")
    print(f"Mean WalkSAT flips (when used): {np.mean(fl):.1f} ± {np.std(fl):.1f}")
    print("="*55)

