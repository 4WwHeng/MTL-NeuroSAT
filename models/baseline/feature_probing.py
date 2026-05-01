import numpy as np
import random
from itertools import accumulate

MAX_NODES = 500
MAX_STEPS = 20


# DPLL

def data_index(d):
    if d > 64:
        return 4
    elif d > 16:
        return 3
    elif d > 4:
        return 2
    elif d > 1:
        return 1
    else:
        return 0


def propagate_units(clauses, assignment):
    queue = []
    for c in clauses:
        if len(c) == 1:
            queue.append(c[0])

    prop_count = 0
    active_clauses = clauses

    while queue:
        lit = queue.pop()
        var = abs(lit)
        val = 1 if lit > 0 else -1

        if assignment[var] != 0:
            if assignment[var] != val:
                return None, prop_count, False
            continue

        assignment[var] = val
        prop_count += 1

        new_clauses = []
        for clause in active_clauses:
            if lit in clause:
                continue
            if -lit in clause:
                nc = tuple(x for x in clause if x != -lit)
                if not nc:
                    return None, prop_count, False
                if len(nc) == 1:
                    queue.append(nc[0])
                new_clauses.append(nc)
            else:
                new_clauses.append(clause)

        active_clauses = tuple(new_clauses)

    return active_clauses, prop_count, True


def dpll(data, clauses, assignment, depth, nodes, max_nodes):
    if nodes[0] >= max_nodes:
        return None

    nodes[0] += 1

    clauses, pc, ok = propagate_units(clauses, assignment)
    data[data_index(depth)] += pc

    if not ok:
        return False
    if not clauses:
        return True

    lit = clauses[0][0]
    var = abs(lit)

    for val in (lit, -lit):
        assignment_copy = assignment.copy()
        if dpll(data, clauses + ((val,),), assignment_copy, depth + 1, nodes, max_nodes):
            return True

    return False


def dpll_contradict(clauses, assignment, depth, nodes, max_nodes):
    if nodes[0] >= max_nodes:
        return depth, 0

    nodes[0] += 1

    clauses, pc, ok = propagate_units(clauses, assignment)
    if not ok:
        return depth, pc
    if not clauses:
        return 0, pc

    lit = clauses[0][0]
    assignment_copy = assignment.copy()
    d, p = dpll_contradict(clauses + ((lit,),), assignment_copy, depth + 1, nodes, max_nodes)
    return d, pc + p


def random_literal(clauses):
    c = random.choice(clauses)
    return random.choice(c)


def dpll_features(clauses, n_var, n_probes=10, max_nodes=MAX_NODES):
    clauses = tuple(tuple(c) for c in clauses)
    data = [0, 0, 0, 0, 0]
    assignment = np.zeros(n_var + 1, dtype=np.int8)

    nodes = [0]
    dpll(data, clauses, assignment, 0, nodes, max_nodes)
    prop = list(accumulate(data))

    stat = {}
    stat["prop_1"] = prop[0] / (prop[1] / 4 + 1e-6)
    stat["prop_4"] = (prop[1] / 4) / (prop[2] / 16 + 1e-6)
    stat["prop_16"] = (prop[2] / 16) / (prop[3] / 64 + 1e-6)
    stat["prop_64"] = (prop[3] / 64) / (prop[4] / 256 + 1e-6)
    stat["prop_256"] = (prop[4] / 256) / (2 * n_var)

    total_depth = 0
    total_prop = 0

    for _ in range(n_probes):
        assignment = np.zeros(n_var + 1, dtype=np.int8)
        lit = random_literal(clauses)
        nodes = [0]
        d, p = dpll_contradict(clauses + ((lit,),), assignment, 0, nodes, max_nodes / 8)
        total_depth += d
        total_prop += p

    stat["mean_depth_conflict"] = total_depth / n_probes
    stat["log_nodes"] = 0.0 if prop[4] == 0 else np.log2(total_prop / n_probes)

    return stat


# SAPS

def satisfied(clause, assignment):
    for l in clause:
        if assignment[abs(l)] == (1 if l > 0 else -1):
            return True
    return False


def count_unsat(clauses, assignment):
    return sum(not satisfied(c, assignment) for c in clauses)


def build_var_clause_map(clauses, n_var):
    vc = [[] for _ in range(n_var + 1)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            vc[abs(lit)].append(i)
    return vc


def weighted_step_incremental(
    clauses, assignment, weights, cost, unsat, var_clauses
):
    best_cost = cost
    best_vars = []

    for v in range(1, len(var_clauses)):
        delta = 0
        assignment[v] *= -1

        for ci in var_clauses[v]:
            was_unsat = ci in unsat
            now_unsat = not satisfied(clauses[ci], assignment)

            if was_unsat and not now_unsat:
                delta -= weights[ci]
            elif not was_unsat and now_unsat:
                delta += weights[ci]

        assignment[v] *= -1

        new_cost = cost + delta
        if new_cost < best_cost:
            best_cost = new_cost
            best_vars = [v]
        elif new_cost == best_cost:
            best_vars.append(v)

    if not best_vars:
        return False, cost

    v = random.choice(best_vars)
    assignment[v] *= -1

    # Update unsat set incrementally
    for ci in var_clauses[v]:
        if not satisfied(clauses[ci], assignment):
            unsat.add(ci)
        else:
            unsat.discard(ci)

    return True, best_cost


def weighted_step(clauses, n_var, assignment, weights, cost):
    best_cost = cost
    best_vars = []

    for v in range(1, n_var + 1):
        assignment[v] *= -1
        new_cost = sum(
            weights[i]
            for i, c in enumerate(clauses)
            if not satisfied(c, assignment)
        )
        assignment[v] *= -1

        if new_cost < best_cost:
            best_cost = new_cost
            best_vars = [v]
        elif new_cost == best_cost:
            best_vars.append(v)

    if not best_vars:
        return False, cost

    v = random.choice(best_vars)
    assignment[v] *= -1
    return True, best_cost


def saps(
    clauses, n_var,
    scaling=1.3,
    smoothing=0.8,
    random_walk_prob=0.01,
    smoothing_prob=0.05,
    max_steps=MAX_STEPS
):
    # Initial random assignment
    assignment = np.random.choice([-1, 1], size=n_var + 1)
    assignment[0] = 0

    weights = np.ones(len(clauses))
    var_clauses = build_var_clause_map(clauses, n_var)

    # Track unsatisfied clauses explicitly
    unsat = set(
        i for i, c in enumerate(clauses)
        if not satisfied(c, assignment)
    )

    cost = sum(weights[i] for i in unsat)

    best_unsat = len(unsat)
    init_unsat = best_unsat
    steps_to_best = 0
    total_improve = 0

    first_min = -1
    minima = []

    for step in range(1, max_steps + 1):

        improved, new_cost = weighted_step_incremental(
            clauses, assignment, weights, cost, unsat, var_clauses
        )

        unsat_count = len(unsat)

        if unsat_count < best_unsat:
            total_improve += best_unsat - unsat_count
            best_unsat = unsat_count
            steps_to_best = step

        if unsat_count == 0:
            break

        if not improved:
            if first_min < 0:
                first_min = unsat_count
            minima.append(unsat_count)

            if random.random() < random_walk_prob:
                v = random.randint(1, n_var)
                assignment[v] *= -1

                # update unsat set
                for ci in var_clauses[v]:
                    if not satisfied(clauses[ci], assignment):
                        unsat.add(ci)
                    else:
                        unsat.discard(ci)
            else:
                # weight update only on unsatisfied clauses
                for ci in unsat:
                    weights[ci] *= scaling

                if random.random() < smoothing_prob:
                    w_mean = np.mean(weights)
                    weights[:] = smoothing * weights + (1 - smoothing) * w_mean

            cost = sum(weights[i] for i in unsat)
        else:
            cost = new_cost

    # Feature calculations 
    frac_first_min = (
        0.0 if init_unsat == best_unsat else
        max(0, init_unsat - first_min) / (init_unsat - best_unsat)
    )

    varcoef = 0.0 if not minima else np.std(minima) / np.mean(minima)

    return {
        "steps_to_best": steps_to_best,
        "avg_improvement": total_improve / step,
        "frac_first_min": frac_first_min,
        "n_unsat_clause": varcoef
    }


def saps_features(clauses, n_var, n_probes=10):
    runs = [saps(clauses, n_var) for _ in range(n_probes)]

    return {
        "mean_steps_best": np.mean([r["steps_to_best"] for r in runs]),
        "median_steps_best": np.median([r["steps_to_best"] for r in runs]),
        "tenth_steps_best": np.percentile([r["steps_to_best"] for r in runs], 10),
        "ninetieth_steps_best": np.percentile([r["steps_to_best"] for r in runs], 90),
        "avg_improvements_mean": np.mean([r["avg_improvement"] for r in runs]),
        "frac_first_min_mean": np.mean([r["frac_first_min"] for r in runs]),
        "n_unsat_clause_mean": np.mean([r["n_unsat_clause"] for r in runs])
    }


