import random
import numpy as np
import optuna
from utils.utils import count_satisfy
from models.NeuroSAT.decoding.neurosat_s import NNs_inference
from data.data_preprocessing import read_data
import pandas as pd
import time
from scipy.stats import gmean, sem, t
from utils.graph import plot_heavy_tail, plot_cactus


# Guided Satisfiable Local Search experiments. Uses NNs_inference (base NeuroSAT with supervised heads) to isolate the contribution of NeuroSAT. Analysis is done only of SAT instance predicted correctly. 


# walksat
def walksat(clauses, n_vars, max_flips=10000, p_noise=0.5, initial_assignment=None):
    if initial_assignment is not None:
        assignment = np.array(initial_assignment, dtype=int)
    else:
        assignment = np.random.randint(2, size=n_vars)

    # Pre-process clauses for faster access
    c_vars = [[abs(l)-1 for l in c] for c in clauses] # list of lists of variable indices 
    c_pols = [[1 if l > 0 else 0 for l in c] for c in clauses] # list of lists of polarities (1 if pos, 0 if neg)
    
    # Build a map of {variable_index -> [clause_indices]}
    # This allows us to only update relevant clauses when we flip a variable
    var_to_clauses = [[] for _ in range(n_vars)]
    for i, c in enumerate(c_vars):
        for v in c:
            var_to_clauses[v].append(i)

    # Compute Initial Satisfaction
    sat_counts = np.zeros(len(clauses), dtype=int)
    unsat_stack = [] 
    for i, (lits, pols) in enumerate(zip(c_vars, c_pols)):
        satisfied_term_count = 0
        for var_idx, pol in zip(lits, pols):
            if assignment[var_idx] == pol:
                satisfied_term_count += 1
        
        sat_counts[i] = satisfied_term_count
        if satisfied_term_count == 0:
            unsat_stack.append(i)

    for step in range(max_flips):
        if len(unsat_stack) == 0:
            return assignment, step # solved
            
        rand_idx = random.randint(0, len(unsat_stack) - 1)
        target_c_idx = unsat_stack[rand_idx]
        
        candidates = c_vars[target_c_idx]

        best_break = float('inf')
        best_candidates = []
        
        for candidate in candidates:
            break_score = 0
            
            # Check all clauses this variable appears in
            for c_idx in var_to_clauses[candidate]:
                if sat_counts[c_idx] == 1:
                    # Verify if candidate is the one satisfying it
                    c_lits = c_vars[c_idx]
                    c_ps = c_pols[c_idx]
                    idx_in_c = c_lits.index(candidate)
                    
                    if assignment[candidate] == c_ps[idx_in_c]:
                        break_score += 1
            
            if break_score < best_break:
                best_break = break_score
                best_candidates = [candidate]
            elif break_score == best_break:
                best_candidates.append(candidate)

        # Free Move
        if best_break == 0:
            flip_var = random.choice(best_candidates)
        # Random Move
        elif random.random() < p_noise:
            flip_var = random.choice(candidates)
        # Greedy Move
        else:            
            flip_var = random.choice(best_candidates)

        # apply flip
        old_val = assignment[flip_var]
        new_val = 1 - old_val
        assignment[flip_var] = new_val
        
        # Update SAT Counts
        # We only check clauses that contain the flipped variable
        for clause_idx in var_to_clauses[flip_var]:
            # Get the polarity of the flipped variable in this specific clause
            lits = c_vars[clause_idx]
            pols = c_pols[clause_idx]
            idx = lits.index(flip_var)
            pol = pols[idx]
            
            # If new_val matches polarity -> We just satisfied it -> count +1
            # If old_val matched polarity -> We just broke it -> count -1
            if new_val == pol:
                sat_counts[clause_idx] += 1
            else:
                sat_counts[clause_idx] -= 1
        
        # refresh UNSAT Stack
        unsat_stack = np.where(sat_counts == 0)[0]

    return None, max_flips # Failed


# Guided WalkSAT 
def guided_walksat(clauses, n_vars, max_flips=10000, p_noise=0.5, initial_assignment=None, var_weights=None, clause_weights=None):
    if initial_assignment is not None:
        assignment = np.array(initial_assignment, dtype=int)
    else:
        assignment = np.random.randint(2, size=n_vars)

    # Pre-process clauses for faster access
    c_vars = [[abs(l)-1 for l in c] for c in clauses] # list of lists of variable indices 
    c_pols = [[1 if l > 0 else 0 for l in c] for c in clauses] # list of lists of polarities (1 if pos, 0 if neg)
    
    # Build a map of {variable_index -> [clause_indices]}
    # This allows us to only update relevant clauses when we flip a variable
    var_to_clauses = [[] for _ in range(n_vars)]
    for i, c in enumerate(c_vars):
        for v in c:
            var_to_clauses[v].append(i)

    # Compute Initial Satisfaction
    sat_counts = np.zeros(len(clauses), dtype=int)
    unsat_stack = [] 
    for i, (lits, pols) in enumerate(zip(c_vars, c_pols)):
        satisfied_term_count = 0
        for var_idx, pol in zip(lits, pols):
            if assignment[var_idx] == pol:
                satisfied_term_count += 1
        
        sat_counts[i] = satisfied_term_count
        if satisfied_term_count == 0:
            unsat_stack.append(i)

    for step in range(max_flips):
        if len(unsat_stack) == 0:
            return assignment, step # solved
            
        rand_idx = random.randint(0, len(unsat_stack) - 1)
        target_c_idx = unsat_stack[rand_idx]
        
        candidates = c_vars[target_c_idx]

        best_break = float('inf')
        best_candidates = []
        
        for candidate in candidates:
            break_score = 0
            
            # Check all clauses this variable appears in
            for c_idx in var_to_clauses[candidate]:
                if sat_counts[c_idx] == 1:
                    # Verify if candidate is the one satisfying it
                    c_lits = c_vars[c_idx]
                    c_ps = c_pols[c_idx]
                    idx_in_c = c_lits.index(candidate)

                    if assignment[candidate] == c_ps[idx_in_c]:
                        if clause_weights is not None:
                            # higher score for breaking brittle clause
                            break_score += clause_weights[c_idx]
                        else:
                            # uniform break score
                            break_score += 1
            
            if break_score < best_break:
                best_break = break_score
                best_candidates = [candidate]
            elif break_score == best_break:
                best_candidates.append(candidate)

        # Free Move
        if best_break == 0:
            flip_var = random.choice(best_candidates)
        # Random Move
        elif random.random() < p_noise:
            # Guided Walk
            if var_weights is not None:
                flip_var = random.choices(candidates, weights=[var_weights[i] for i in candidates], k=1)[0]
            # Random Walk
            else:
                flip_var = random.choice(candidates)
        # Greedy Move
        else:            
            flip_var = random.choice(best_candidates)

        # apply flip
        old_val = assignment[flip_var]
        new_val = 1 - old_val
        assignment[flip_var] = new_val
        
        # Update SAT Counts
        # We only check clauses that contain the flipped variable
        for clause_idx in var_to_clauses[flip_var]:
            # Get the polarity of the flipped variable in this specific clause
            lits = c_vars[clause_idx]
            pols = c_pols[clause_idx]
            idx = lits.index(flip_var)
            pol = pols[idx]
            
            # If new_val matches polarity -> We just satisfied it -> count +1
            # If old_val matched polarity -> We just broke it -> count -1
            if new_val == pol:
                sat_counts[clause_idx] += 1
            else:
                sat_counts[clause_idx] -= 1
        
        # refresh UNSAT Stack
        unsat_stack = np.where(sat_counts == 0)[0]

    return None, max_flips # Failed


# hybrid sat search weights tuning
def calculate_clause_weights(clause_prob, weights=[8.885, 3.130, 0.305]):
    # Optimal Weights Found: {'w0': 6.442364140659167, 'w1': 2.3627843943281865, 'w2': 0.36800651526224787} [only median]
    # Optimal Weights Found: {'w0': 8.885423673735296, 'w1': 3.1296780044059127, 'w2': 0.3045233486978076} [median + mean]
    """Calculates weights for clauses based on predicted tier probabilities."""
    # Dot product of probabilities and weights
    cweights = clause_prob @ weights
    
    # Normalize
    cweights /= cweights.sum()
    return cweights

def calculate_uncertain_weights(votes, bias_factor=1.56):
    # Optimal Weights Found: {'bf': 1.5645092041327024}
    """Calculates variable weights based on prediction uncertainty."""
    
    votes = np.array(votes)
    weights = np.zeros_like(votes, dtype=float)
    
    k_pos = 1.0  
    k_neg = k_pos / bias_factor 

    # Vectorized exponential weighting based on certainty
    pos_mask = votes >= 0
    weights[pos_mask] = np.exp(-k_pos * np.abs(votes[pos_mask]))
    
    neg_mask = votes < 0
    weights[neg_mask] = np.exp(-k_neg * np.abs(votes[neg_mask]))
    
    # Normalize
    weights /= weights.sum()
    return weights

def calculate_unsat_weights(votes, bias_factor=4.65):  
    # Optimal Weights Found: {'bf': 4.6480789040525705}
    """Calculates variable weights favoring variables predicted as UNSAT."""
    
    votes = np.array(votes)
    
    # Scale votes and apply tanh
    s = np.tanh(votes / bias_factor)
    
    # Weight calculation pushing towards UNSAT variables
    weights = ((1 - s) / 2) ** 2
    
    # Normalize
    weights /= weights.sum()
    return weights


# Automated hyperparameter tuning with Optuna
def clause_objective(trial, hard_instances):
    # Suggest weights for the three tiers
    w0 = trial.suggest_float("w0", 2.0, 10.0)  # Brittle: high range to 'freeze' them
    w1 = trial.suggest_float("w1", 1.0, 5.0)   # Medium
    w2 = trial.suggest_float("w2", 0.1, 1.0)   # Slack
    
    total_flips = []

    for clauses, n_vars, assignment, var_vote, tier_prob in hard_instances:
        cweights = calculate_clause_weights(tier_prob, weights=[w0, w1, w2])

        sol, flips = guided_walksat(
            clauses, 
            n_vars=n_vars, 
            initial_assignment=assignment,
            max_flips=10000, 
            clause_weights=cweights
        )
        
        if sol is not None:
            total_flips.append(flips)
        else:
            total_flips.append(20000) # Penalty for failing to solve

    return np.median(total_flips) + 0.2 * np.mean(total_flips)

def var_uc_objective(trial, hard_instances):
    bf = trial.suggest_float("bf", 0.0, 5.0)  
    total_flips = []

    for clauses, n_vars, assignment, var_vote, tier_prob in hard_instances:
        vweights = calculate_uncertain_weights(var_vote, bias_factor=bf)

        sol, flips = guided_walksat(
            clauses, 
            n_vars=n_vars, 
            initial_assignment=assignment,
            max_flips=10000, 
            var_weights=vweights
        )
        
        if sol is not None:
            total_flips.append(flips)
        else:
            total_flips.append(20000)

    return np.median(total_flips) + 0.2 * np.mean(total_flips)

def var_us_objective(trial, hard_instances):
    bf = trial.suggest_float("bf", 1.0, 5.0)  
    total_flips = []

    for clauses, n_vars, assignment, var_vote, tier_prob in hard_instances:
        vweights = calculate_unsat_weights(var_vote, bias_factor=bf)

        sol, flips = guided_walksat(
            clauses, 
            n_vars=n_vars, 
            initial_assignment=assignment,
            max_flips=10000, 
            var_weights=vweights
        )
        
        if sol is not None:
            total_flips.append(flips)
        else:
            total_flips.append(20000)

    return np.median(total_flips) + 0.2 * np.mean(total_flips)


def weights_tuning(test_data_sat="40_SAT", test_data_unsat="40_UNSAT", tuning_type="var_us"):
    """Main tuning loop. tuning_type options: clause, var_uc, var_us"""
    print("Running Inference for parameter tuning...")
    
    graph_votes, split_lit_emb, split_clause_emb, split_adm_prob, split_ctp_prob, split_var_votes, latency = NNs_inference(test_data_sat, test_data_unsat) # type: ignore
    
    print("Loading datasets...")
    data_sat = read_data(test_data_sat, is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    hard_instances = [] # Store: (clauses, n_vars, assignment, var_vote, tier_prob)

    print("Filtering for hard instances...")
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if not is_sat_ground_truth:
            continue

        vote = graph_votes[i]
        if vote < 0.5:
            continue

        var_vote = split_var_votes[i]
        candidate_prob = split_adm_prob[i]
        tier_prob = split_ctp_prob[i]

        candidate = np.argmax(candidate_prob, axis=1)
        _, direct_solved = count_satisfy(candidate, clauses) 
        
        # We only want to tune on problems the NN couldn't solve directly
        if direct_solved:
            continue
       
        hard_instances.append((clauses, n_vars, candidate, var_vote, tier_prob))
      
    print(f"Found {len(hard_instances)} hard instances for tuning.")
    
    if not hard_instances:
        print("No hard instances found. Exiting tuning.")
        return

    # Set up Optuna Study
    study = optuna.create_study(direction="minimize")
    
    # Select the correct objective function based on the tuning type
    if tuning_type == "clause":
        objective_func = lambda trial: clause_objective(trial, hard_instances)
    elif tuning_type == "var_uc":
        objective_func = lambda trial: var_uc_objective(trial, hard_instances)
    elif tuning_type == "var_us":
        objective_func = lambda trial: var_us_objective(trial, hard_instances)
    else:
        raise ValueError("Invalid tuning_type. Choose 'clause', 'var_uc', or 'var_us'.")

    print(f"Starting Optuna Study ({tuning_type})...")
    study.optimize(objective_func, n_trials=20) #type: ignore
    
    print("\n" + "="*50)
    print(f"OPTIMAL WEIGHTS FOUND ({tuning_type}):")
    print(study.best_params)
    print("="*50)


# WalkSAT experiment
def walksat_experiment(test_data_sat="Test_100_SAT", test_data_unsat="Test_100_UNSAT", num_runs=5):
    print(f"Running Inference for WalkSAT experiments (Total Runs per instance: {num_runs})...")
    
    votes, split_lit_emb, split_clause_emb, split_adm_prob, split_ctp_prob, split_var_votes, latency = NNs_inference(
        test_data_sat, test_data_unsat
    ) # type: ignore
    
    data_sat = read_data(test_data_sat, is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    records = []
    
    print("Executing Guided WalkSAT across all instances...")
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if not is_sat_ground_truth:
            continue

        # Extract NN heuristics for this specific instance
        vote = votes[i]
        if vote < 0.5:
            continue
        
        var_vote = split_var_votes[i]
        candidate_prob = split_adm_prob[i]
        tier_prob = split_ctp_prob[i]

        candidate = np.argmax(candidate_prob, axis=1)
        _, direct_solved = count_satisfy(candidate, clauses) 

        # Compute heuristic weights once per instance
        heuristic_configs = {
            "ws": {"args": {"initial_assignment": candidate}},
            "bc": {"args": {"initial_assignment": candidate, "clause_weights": calculate_clause_weights(tier_prob)}},
            "uc": {"args": {"initial_assignment": candidate, "var_weights": calculate_uncertain_weights(var_vote)}},
            "us": {"args": {"initial_assignment": candidate, "var_weights": calculate_unsat_weights(var_vote)}}
        }

        # 5x runs per instance to get a distribution of performance metrics
        for run_id in range(num_runs):
            instance_record = {
                'id': i,
                'run_id': run_id,
                'n_vars': n_vars,
                'nn_solved_directly': direct_solved
            }
        
            # Baseline: Pure Random Start
            start_time = time.perf_counter()
            sol, flips = guided_walksat(clauses, n_vars=n_vars, max_flips=10000)
            instance_record["rd_success"] = (sol is not None)
            instance_record["rd_flips"] = flips
            instance_record["rd_time"] = (time.perf_counter() - start_time) * 1000
        
            # Neural Heuristics
            for name, config in heuristic_configs.items():
                start_time = time.perf_counter()
                
                if direct_solved:
                    flips = 0
                    success = True
                else:
                    sol, flips = guided_walksat(clauses, n_vars=n_vars, max_flips=10000, **config["args"])
                    success = (sol is not None)
                    
                # add the NN latency
                instance_record[f"{name}_time"] = latency + ((time.perf_counter() - start_time) * 1000)
                instance_record[f"{name}_flips"] = flips
                instance_record[f"{name}_success"] = success
        
            records.append(instance_record)

        if (i+1) % 100 == 0:
            print(f"Processed {i+1} / {len(data_sat)}...")

    # Print Final Report
    df = pd.DataFrame(records)
    methods = ['rd', 'ws', 'bc', 'uc', 'us']
    
    def print_aggregated_metrics(dataframe, title, runs):
        if dataframe.empty:
            print(f"\nNo instances available for {title} analysis.")
            return

        print("\n" + "="*80)
        print(f"  {title} (Averaged over {runs} runs with 95% CI)")
        print("="*80)
        
        # Calculate stats strictly per run first, then average those stats
        run_stats = {m: {'success': [], 'median_flips': [], 'mean_flips': [], 'sgm': [], 'throughput': []} for m in methods}
        
        for r in range(runs):
            run_df = dataframe[dataframe['run_id'] == r]
            if run_df.empty: continue
            
            for m in methods:
                run_stats[m]['success'].append(run_df[f'{m}_success'].mean() * 100)
                
                succ_df = run_df[run_df[f'{m}_success']]
                if not succ_df.empty:
                    run_stats[m]['median_flips'].append(succ_df[f'{m}_flips'].median())
                    run_stats[m]['mean_flips'].append(succ_df[f'{m}_flips'].mean())
                    
                    total_time = succ_df[f'{m}_time'].sum()
                    run_stats[m]['throughput'].append((len(succ_df) * 1000) / total_time if total_time > 0 else 0)
                
                run_stats[m]['sgm'].append(gmean(run_df[f'{m}_flips'] + 10) - 10)

        # Helper to calculate Mean ± 95% Confidence Interval
        def calc_ci(data_list):
            if len(data_list) < 2: return np.mean(data_list) if data_list else 0, 0
            mean_val = np.mean(data_list)
            # t-distribution multiplier for 95% CI
            ci_val = sem(data_list) * t.ppf((1 + 0.95) / 2., len(data_list)-1)
            return mean_val, ci_val

        print("\n[1] SOLVABILITY (%)")
        for m in methods:
            mean_val, ci_val = calc_ci(run_stats[m]['success'])
            print(f"{m.upper():<5}: {mean_val:>6.1f} ± {ci_val:>4.1f}%")

        print("\n[2] SEARCH EFFICIENCY (MEDIAN & MEAN FLIPS TO SOLVE)")
        for m in methods:
            med_mean, med_ci = calc_ci(run_stats[m]['median_flips'])
            mean_mean, mean_ci = calc_ci(run_stats[m]['mean_flips'])
            print(f"{m.upper():<5}: {med_mean:>7.1f} ± {med_ci:>5.1f} median | {mean_mean:>7.1f} ± {mean_ci:>5.1f} mean")

        print("\n[3] SHIFTED GEOMETRIC MEAN (SGM-10) FLIPS")
        rd_mean, _ = calc_ci(run_stats['rd']['sgm'])
        print(f"SGM (RD): {rd_mean:.2f}")
        for m in ['ws', 'bc', 'uc', 'us']:
             m_mean, m_ci = calc_ci(run_stats[m]['sgm'])
             speedup = rd_mean / m_mean if m_mean > 0 else 0
             print(f"SGM ({m.upper()}): {m_mean:>7.2f} ± {m_ci:>5.2f}  | Speedup: {speedup:.2f}x")

        print("\n[4] THROUGHPUT (SOLVED / SECOND)")
        for m in methods:
            tp_mean, tp_ci = calc_ci(run_stats[m]['throughput'])
            print(f"{m.upper():<5}: {tp_mean:>7.2f} ± {tp_ci:>5.2f} solved/sec")

    # Execute Aggregated Printing
    print_aggregated_metrics(df, "ALL SAT INSTANCES", num_runs)
    
    needs_refinement = df[~df['nn_solved_directly']].copy()
    print_aggregated_metrics(needs_refinement, "REFINEMENT RESULTS (HARD INSTANCES ONLY)", num_runs)

    print("\n" + "="*80)

    # Plotting for final run only
    final_run_df = df[df['run_id'] == (num_runs - 1)]
    
    results_to_plot = {
        "Random Baseline": final_run_df['rd_flips'].values,
        "Warm Start": final_run_df['ws_flips'].values,
        "Brittle Clause Heuristic": final_run_df['bc_flips'].values,
        "Uncertain Variable Guidance": final_run_df['uc_flips'].values,
        "Unsatisfiable Variable Guidance": final_run_df['us_flips'].values
    }
    
    plot_cactus(**results_to_plot)
    plot_heavy_tail(**results_to_plot)

