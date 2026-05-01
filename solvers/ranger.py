import random
import numpy as np
from data.data_preprocessing import read_data
from models.NeuroSAT.decoding.neurosat_s import NNs_inference
import pandas as pd
import time
from scipy.stats import gmean


# guided Ranger

def binary_closure(knowledge_set, n_vars):
    new_units = set()
    binaries = [c for c in knowledge_set if len(c) == 2]
    
    lit_to_binaries = {l: set() for l in range(-n_vars, n_vars + 1) if l != 0}
    for c in binaries:
        for l in c:
            lit_to_binaries[l].add(c)

    # Resolve binary pairs
    for pivot in range(1, n_vars + 1):
        pos_binaries = lit_to_binaries[pivot]
        neg_binaries = lit_to_binaries[-pivot]
        
        for c1 in pos_binaries:
            for c2 in neg_binaries:
                res_set = (set(c1) - {pivot}) | (set(c2) - {-pivot})
                if any(-lit in res_set for lit in res_set):
                    continue
                    
                new_clause = tuple(sorted(list(res_set)))
                
                if len(new_clause) == 1:
                    new_units.add(new_clause[0])                    
                knowledge_set.add(new_clause)
                    
    return new_units
    

def propagate_unit(active_clauses, lit_to_clauses, new_unit, knowledge_set):
    unit_queue = [new_unit]
    
    while unit_queue:
        u_lit = unit_queue.pop(-1)
        
        clashing_indices = list(lit_to_clauses[-u_lit])
        for idx in clashing_indices:
            target_c = active_clauses[idx]

            res = tuple(l for l in target_c if l != -u_lit)
            
            if len(res) == 0:
                return True
            
            for lit in active_clauses[idx]:
                lit_to_clauses[lit].discard(idx)
            for lit in res:
                lit_to_clauses[lit].add(idx)
            active_clauses[idx] = res

            if len(res) <= 2:
                knowledge_set.add(res)

            if len(res) == 1 and res[0] not in unit_queue:
                unit_queue.append(res[0])
                
    return False


def guided_ranger(clauses, n_vars, k=200, w=20, max_flips=10000, p_i=0.1, p_t=0.9, p_g=0.5, core_prob=None, var_brittleness=None):
    # pure literals
    all_lits = set()
    for c in clauses:
        all_lits.update(c)
    pure_lits = {l for l in all_lits if -l not in all_lits}

    # clause_to_idx_in_source
    clauses = [tuple(sorted(c)) for c in clauses]
    clause_to_source = {c: i for i,c in enumerate(clauses)}

    take = min(len(clauses), k)
    if core_prob is not None:
        # top k core-prob clauses 
        sorted_indices = sorted(range(len(clauses)), key=lambda i: core_prob[i], reverse=True)
        start_indices = set([i for i in sorted_indices[:take]])
    else:
        start_indices = set(random.sample(range(len(clauses)), take))
    unchosen_indices = set(range(len(clauses))) - start_indices
    active_clauses = [clauses[i] for i in start_indices]

    knowledge_set = set() # two binary clause with flips lits are considered different. Painful to solve since using sets.

    run_stats = {"resolvable_pairs": 0, "avg_lengths": [], "min_length": w}
    
    # lit_to_clauses[lit] = {indices of active clauses containing that lit}
    lit_to_clauses = {lit: set() for lit in range(-n_vars, n_vars + 1) if lit != 0}
    for idx, clause in enumerate(active_clauses):
        for lit in clause:
            lit_to_clauses[lit].add(idx)

    for steps in range(max_flips):
        if steps % 50 == 0:
            current_lengths = [len(c) for c in active_clauses]
            run_stats["avg_lengths"].append(np.mean(current_lengths))
            run_stats["min_length"] = min(run_stats["min_length"], min(current_lengths))

        
        if steps % 250 == 0:
            u_lits = binary_closure(knowledge_set, n_vars)
            
            if u_lits:
                for u in u_lits:
                    if (u,) not in knowledge_set:
                        knowledge_set.add((u,))
                        if propagate_unit(active_clauses, lit_to_clauses, u, knowledge_set):
                            return True, steps, run_stats

        
        # refresh memory
        if steps % 50 == 0 and knowledge_set:
            idx_to_remove = random.randint(0, len(active_clauses) - 1) # may duplicate knowledge set but who cares Oo
            old_clause = active_clauses[idx_to_remove]
            new_clause = random.choice(list(knowledge_set))
            
            # Update literal map
            for lit in old_clause:
                lit_to_clauses[lit].remove(idx_to_remove)
            for lit in new_clause:
                lit_to_clauses[lit].add(idx_to_remove)
            active_clauses[idx_to_remove] = new_clause

            # refresh unchosen indices
            unchosen_indices.add(clause_to_source.get(old_clause, -1))
            unchosen_indices.discard(-1) # -1 if not from source

        # Random Move
        if random.random() < p_i:
            # eligible_indices = [i for i, c in enumerate(active_clauses) if len(c) > 2]
            # if eligible_indices:
            #     idx_to_remove = random.choice(eligible_indices)
            # else:
            idx_to_remove = random.randint(0, len(active_clauses) - 1)
            old_clause = active_clauses[idx_to_remove]

            
            if knowledge_set and random.random() < 0.3:
                new_clause = random.choice(list(knowledge_set))
                
                 # Update literal map
                for lit in old_clause:
                    lit_to_clauses[lit].remove(idx_to_remove)
                for lit in new_clause:
                    lit_to_clauses[lit].add(idx_to_remove)
                active_clauses[idx_to_remove] = new_clause
        
                # refresh unchosen indices
                unchosen_indices.add(clause_to_source.get(old_clause, -1))
                unchosen_indices.discard(-1) # -1 if not from source
                
            # Pick a random unchosen index from the original formula to swap in
            elif unchosen_indices:
                new_idx_from_source = random.choice(tuple(unchosen_indices))
                new_clause = clauses[new_idx_from_source]
                
                # Update literal map
                for lit in old_clause:
                    lit_to_clauses[lit].remove(idx_to_remove)
                for lit in new_clause:
                    lit_to_clauses[lit].add(idx_to_remove)
                active_clauses[idx_to_remove] = new_clause
        
                # refresh unchosen indices
                unchosen_indices.remove(new_idx_from_source)
                unchosen_indices.add(clause_to_source.get(old_clause, -1))
                unchosen_indices.discard(-1) # -1 if not from source
        # Greedy Move
        else:
            # guided resolve
            if var_brittleness is not None:
                pivot = random.choices(range(1, n_vars+1), var_brittleness, k=1)[0]
            # random resolve
            else:
                pivot = random.randint(1, n_vars)
                
            pos_indices = lit_to_clauses[pivot]
            neg_indices = lit_to_clauses[-pivot]
            if pos_indices and neg_indices:
                run_stats["resolvable_pairs"] += 1
                # Pick one clause from each side
                idx1 = random.choice(tuple(pos_indices))
                idx2 = random.choice(tuple(neg_indices))
                
                c1, c2 = active_clauses[idx1], active_clauses[idx2]
                
                # Perform Resolution
                res_set = (set(c1) - {pivot}) | (set(c2) - {-pivot})
                res_set = tuple(sorted(list(res_set)))

                if len(res_set) == 0:
                    return True, steps, run_stats #success

                # Tautology check: e.g., if [1, -1] is in result, discard
                if any(-lit in res_set for lit in res_set) or len(res_set) > w:
                    continue

                if len(res_set) <= 2:
                    knowledge_set.add(res_set)
            
                    if len(res_set) == 1:
                        if propagate_unit(active_clauses, lit_to_clauses, res_set[0], knowledge_set):
                            return True, steps, run_stats
                        
                        if len(c1) > len(c2):
                            target_idx = idx1 
                        else:
                            target_idx = idx2
                        old_clause = active_clauses[target_idx]
    
                        for lit in old_clause:
                            lit_to_clauses[lit].remove(target_idx)
                        for lit in res_set:
                            lit_to_clauses[lit].add(target_idx)
                        active_clauses[target_idx] = res_set
    
                        unchosen_indices.add(clause_to_source.get(old_clause, -1))
                        unchosen_indices.discard(-1)

                # Replacement logic (p_g)
                elif random.random() < p_g:
                    # Replace the longer parent
                    if len(c1) > len(c2):
                        target_idx = idx1 
                    else:
                        target_idx = idx2
                    old_clause = active_clauses[target_idx]

                    for lit in old_clause:
                        lit_to_clauses[lit].remove(target_idx)
                    for lit in res_set:
                        lit_to_clauses[lit].add(target_idx)
                    active_clauses[target_idx] = res_set

                    unchosen_indices.add(clause_to_source.get(old_clause, -1))
                    unchosen_indices.discard(-1)
                    
        # Subsumption Check 
        if random.random() < p_t:
            lit = random.randint(1, n_vars) * random.choice([1, -1])
            shares = list(lit_to_clauses[lit])
            
            if len(shares) >= 2:
                idx_a, idx_b = random.sample(shares, 2)
                c_a, c_b = set(active_clauses[idx_a]), set(active_clauses[idx_b])
                
                if c_a.issubset(c_b) and unchosen_indices:
                    new_idx = random.choice(tuple(unchosen_indices))
                    old_clause = active_clauses[idx_b]

                    for l in c_b: 
                        lit_to_clauses[l].remove(idx_b)
                    for l in clauses[new_idx]: 
                        lit_to_clauses[l].add(idx_b)
                    active_clauses[idx_b] = clauses[new_idx]

                    unchosen_indices.remove(new_idx)
                    unchosen_indices.add(clause_to_source.get(old_clause, -1))
                    unchosen_indices.discard(-1)
                elif c_b.issubset(c_a) and unchosen_indices:
                    new_idx = random.choice(tuple(unchosen_indices))
                    old_clause = active_clauses[idx_a]

                    for l in c_a: 
                        lit_to_clauses[l].remove(idx_a)
                    for l in clauses[new_idx]: 
                        lit_to_clauses[l].add(idx_a)
                    active_clauses[idx_a] = clauses[new_idx]

                    unchosen_indices.remove(new_idx)
                    unchosen_indices.add(clause_to_source.get(old_clause, -1))
                    unchosen_indices.discard(-1)
   
        # pure literal elimination
        if random.random() < p_t:
            check_idx = random.randint(0, len(active_clauses) - 1)
            c = active_clauses[check_idx]
            
            if any(lit in pure_lits for lit in c):
                new_idx = random.choice(tuple(unchosen_indices))

                for l in c: 
                    lit_to_clauses[l].remove(check_idx)
                for l in clauses[new_idx]: 
                    lit_to_clauses[l].add(check_idx)
                active_clauses[check_idx] = clauses[new_idx]

                unchosen_indices.add(clause_to_source.get(c, -1))
                unchosen_indices.discard(-1)

    return False, max_flips, run_stats # Failed


# ranger experiment
def calculate_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_var_brittleness(embeddings, n_vars):
    """Calculates variable brittleness using the distance to the embedding centroid."""
    center = embeddings.mean(axis=0)
    lit_distances = np.linalg.norm(embeddings - center, axis=1)
    
    # Vectorized computation instead of a for-loop: 
    stress = (lit_distances[:n_vars] + lit_distances[n_vars:2*n_vars]) / 2.0
    
    weights = calculate_softmax(stress)
    return weights

def get_core_prob(embeddings, clauses):
    """Calculates clause core probability using the distance to the embedding centroid."""
    center = embeddings.mean(axis=0)
    clause_distances = np.linalg.norm(embeddings - center, axis=1)
    
    weights = calculate_softmax(clause_distances)
    return weights


# Single Test Function
def ranger_test(test_data_sat="Test_40_SAT", test_data_unsat="Test_40_UNSAT"):
    graph_votes, split_lit_emb, split_clause_emb, split_adm_prob, split_ctp_prob, split_var_votes, latency = NNs_inference(
        test_data_sat, test_data_unsat
    ) 
    
    data_sat = read_data(test_data_sat, is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    pairs = []
    cl = []
    minn = []
    solved = 0
    
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if is_sat_ground_truth:
            continue

        vote = graph_votes[i]
        if vote > 0.5:
            continue

        L_h = split_lit_emb[i]
        C_h = split_clause_emb[i]

        var_brittleness = get_var_brittleness(L_h, n_vars)
        core_prob = get_core_prob(C_h, clauses)

        res, steps, stats = guided_ranger(clauses, n_vars, var_brittleness=var_brittleness, core_prob=core_prob)
        
        if res:
            solved += 1
            
        pairs.append(stats["resolvable_pairs"])
        cl.append(np.mean(stats["avg_lengths"]))
        minn.append(stats["min_length"])

        if (i+1) % 100 == 0:
            print(f"Processed {i+1} / {len(data)}...")

    # Print Final Results
    print("\n" + "="*60)
    print("RANGER TEST RESULTS (UNSAT INSTANCES)")
    print("="*60)
    
    if len(pairs) > 0:
        print(f"Instances evaluated:                   {len(pairs)}")
        print(f"Total solved:                          {solved}")
        print(f"Mean number of resolvable pairs:       {np.mean(pairs):.2f}")
        print(f"Mean average length of active clauses: {np.mean(cl):.2f}")
        print(f"Mean minimum length of active clauses: {np.mean(minn):.2f}")
    else:
        print("No instances evaluated (all were SAT or falsely predicted as SAT).")


# Experiment 
def ranger_experiment(test_data_sat="Test_40_SAT", test_data_unsat="Test_40_UNSAT"):
    print("Running Inference for Ranger experiments...")
    
    graph_votes, split_lit_emb, split_clause_emb, split_adm_prob, split_ctp_prob, split_var_votes, latency = NNs_inference(
        test_data_sat, test_data_unsat
    )
    
    data_sat = read_data(test_data_sat, is_training=False, fixed_label=1)
    data_unsat = read_data(test_data_unsat, is_training=False, fixed_label=0)
    data = data_sat + data_unsat

    records = []
    count = 0
    
    print("Executing Guided Ranger across UNSAT instances...")
    for i, (clauses, n_vars, is_sat_ground_truth) in enumerate(data):
        if is_sat_ground_truth:
            continue

        vote = graph_votes[i]
        if vote > 0.5:
            continue

        L_h = split_lit_emb[i]
        C_h = split_clause_emb[i]

        var_brittleness = get_var_brittleness(L_h, n_vars)
        core_prob = get_core_prob(C_h, clauses)

        # Helper function to run and time the guided_ranger
        def run_ranger_trial(config_kwargs):
            start = time.perf_counter()
            success, steps, stats = guided_ranger(clauses, n_vars, **config_kwargs)
            search_time = (time.perf_counter() - start) * 1000
            return success, steps, stats, search_time

        
        # 1. Complete (Both Heuristics)
        complete_success, complete_steps, complete_stats, complete_search_time = run_ranger_trial({
            'var_brittleness': var_brittleness, 'core_prob': core_prob
        })
        
        # 2. Core-Only
        core_success, core_steps, core_stats, core_search_time = run_ranger_trial({
            'core_prob': core_prob
        })
        
        # 3. Var-Only
        var_success, var_steps, var_stats, var_search_time = run_ranger_trial({
            'var_brittleness': var_brittleness
        })
        
        # 4. Random Baseline
        random_success, random_steps, random_stats, random_search_time = run_ranger_trial({})

        # Print 10 sample to monitor progress/differences (not even 10 solved in the end smh)
        if complete_steps != 10000 and count < 10:
            print(f"[{i}] Steps -> Complete: {complete_steps}, Core: {core_steps}, Var: {var_steps}, Random: {random_steps}")
            count += 1
            
        records.append({
            'id': i,
            'n_vars': n_vars,

            'complete_time': latency + complete_search_time,
            'complete_success': complete_success,
            'complete_steps': complete_steps,
            'complete_active_clause_length': np.mean(complete_stats['avg_lengths']) if complete_stats['avg_lengths'] else 0,

            'core_time': latency + core_search_time,
            'core_success': core_success,
            'core_steps': core_steps,
            'core_active_clause_length': np.mean(core_stats['avg_lengths']) if core_stats['avg_lengths'] else 0,

            'var_time': latency + var_search_time,
            'var_success': var_success,
            'var_steps': var_steps,
            'var_active_clause_length': np.mean(var_stats['avg_lengths']) if var_stats['avg_lengths'] else 0,

            'random_time': random_search_time, # Random has 0 latency
            'random_success': random_success,
            'random_steps': random_steps,
            'random_active_clause_length': np.mean(random_stats['avg_lengths']) if random_stats['avg_lengths'] else 0
        })

        if (i+1) % 100 == 0:
            print(f"Processed {i+1} / {len(data)}...")

    # Print Final Results
    df = pd.DataFrame(records)
    
    if df.empty:
        print("No instances to analyze.")
        return

    solved_by_all = df[df['complete_success'] & df['random_success'] & df['core_success'] & df['var_success']]
    total_unfiltered = len(df)
    total_all_solved = len(solved_by_all)
    
    print("\n" + "="*60)
    print("      ABLATION STUDY: GNN HEURISTICS VS RANDOM BASELINE")
    print("="*60)
    print(f"Total UNSAT Instances Tested: {total_unfiltered}")
    print(f"Instances Solved by All Methods: {total_all_solved}")
    
    methods = ['random', 'core', 'var', 'complete']

    print("\n[1] SOLVABILITY (SUCCEEDED / TOTAL)")
    for m in methods:
        success_rate = df[f'{m}_success'].mean() * 100
        total_solved = df[f'{m}_success'].sum()
        print(f"{m.upper():<10}: {success_rate:>5.1f}% ({total_solved}/{total_unfiltered})")

    print("\n[2] SEARCH EFFICIENCY (MEDIAN STEPS TO SOLVE)")
    for m in methods:
        success_df = df[df[f'{m}_success']]
        if not success_df.empty:
            median_steps = success_df[f'{m}_steps'].median()
            print(f"{m.upper():<10}: {median_steps:>7.1f} steps")
        else:
            print(f"{m.upper():<10}: No successful solves")

    print("\n[3] SHIFTED GEOMETRIC MEAN (SGM-10) SPEEDUP")
    shift = 10
    sgm = {}
    for m in methods:
        sgm[m] = gmean(df[f'{m}_steps'] + shift) - shift
    
    print(f"SGM (Random):   {sgm['random']:.2f}")
    for m in ['core', 'var', 'complete']:
        speedup = sgm['random'] / sgm[m] if sgm[m] > 0 else 0
        print(f"SGM ({m.capitalize() + '):':<9} {sgm[m]:>7.2f}  | Speedup: {speedup:.2f}x")

    print("\n[4] WALL-CLOCK PERFORMANCE (SEARCH + INFERENCE)")
    for m in methods:
        success_df = df[df[f'{m}_success']]
        if not success_df.empty:
            avg_time = success_df[f'{m}_time'].mean()
            total_time = success_df[f'{m}_time'].sum()
            throughput = (len(success_df) * 1000) / total_time if total_time > 0 else 0
            print(f"{m.upper():<10}: {avg_time:>6.2f} ms/sample | Throughput: {throughput:>6.2f} solved/sec")
        else:
             print(f"{m.upper():<10}: No successful solves")

    print("\n[5] LOGICAL DENSITY (MEAN ACTIVE CLAUSE LENGTH)")
    for m in methods:
        avg_len = df[f'{m}_active_clause_length'].mean()
        print(f"{m.upper():<10}: {avg_len:.3f} literals")

    print("\n" + "="*60)

    # Plotting (no reason to plot since only <10 solved but here for reference)
    # results_to_plot = {
    #     "Random Baseline": df['random_steps'].values,
    #     "Unsatisfiable Core Start": df['core_steps'].values,
    #     "Weighted Variable Resolution": df['var_steps'].values,
    #     "Both Heuristics": df['complete_steps'].values,        
    # }
    
    # # plot_cactus(**results_to_plot)
    # # plot_heavy_tail(**results_to_plot)


# ranger_test()
# ranger_experiment()