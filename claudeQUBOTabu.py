"""
qubo_solver.py
==============
Solve  min  x^T Q x,  x in {0,1}^n

Includes:
  - diagnose_Q()          -- explains why the null vector may NOT be optimal
  - tabu_search_qubo()    -- optimised O(n) Tabu Search
  - sa_qubo()             -- Simulated Annealing
  - parallel_tempering()  -- PT with replica exchange (best quality / large n)
  - run_parallel()        -- launch any solver across multiple CPU cores

Usage:
    from qubo_solver import run_parallel, diagnose_Q
    diagnose_Q(Q)
    best_x, best_cost = run_parallel(Q, solver="pt", n_workers=8)
"""

import numpy as np
import random
import time
import multiprocessing as mp
import concurrent.futures
import os
import sys


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

def diagnose_Q(Q, verbose=True):
    """
    Check whether x=0 (null vector) is actually the optimal solution.

    Key insight: cost(x=0) = 0.
    For x=0 to be optimal we need x^T Q x >= 0 for all x in {0,1}^n.
    A necessary condition is that every diagonal entry Q[i,i] >= 0,
    because x=e_i (unit vector) gives cost exactly Q[i,i].
    If ANY Q[i,i] < 0, the solver is CORRECT to set bit i=1.
    """
    n = Q.shape[0]
    diag = np.diag(Q)
    neg_diag_idx = np.where(diag < 0)[0]

    print("=" * 55)
    print("QUBO MATRIX DIAGNOSIS")
    print("=" * 55)
    print(f"  n                      : {n}")
    print(f"  diagonal min / max     : {diag.min():.4f} / {diag.max():.4f}")
    print(f"  negative diagonal count: {len(neg_diag_idx)}  ({100*len(neg_diag_idx)/n:.1f}%)")

    if len(neg_diag_idx) > 0:
        print()
        print("  *** NULL VECTOR IS NOT OPTIMAL ***")
        print(f"  {len(neg_diag_idx)} bits have Q[i,i] < 0.")
        print("  Setting each such bit to 1 (independently) REDUCES the cost.")
        print("  The solver finding ~those many ones is CORRECT behaviour.")
        print()
        greedy_x = (diag < 0).astype(np.int8)
        greedy_cost = float(greedy_x @ Q @ greedy_x)
        print(f"  Greedy lower bound (flip all negative-diagonal bits): {greedy_cost:.4f}")
        print(f"  Zero-vector cost                                     : 0.0000")
    else:
        print()
        print("  All diagonal entries >= 0.")
        print("  The null vector IS a local minimum (cost 0).")
        print("  If the solver still finds non-zero x, check for large")
        print("  negative off-diagonal pairs: Q[i,j]+Q[j,i] << 0.")
        off = Q + Q.T - 2 * np.diag(diag)
        neg_off = (off < 0).sum() // 2
        print(f"  Negative off-diagonal pairs: {neg_off}")

    print("=" * 55)
    return neg_diag_idx


# ═══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════

def _symmetrise(Q):
    """Return Q_sym = Q + Q^T and Q_diag, Q_sym_diag once."""
    Q_sym = Q + Q.T
    Q_diag = np.diag(Q).copy()
    Q_sym_diag = 2.0 * Q_diag
    return Q_sym, Q_diag, Q_sym_diag


def _init_delta(x, Q_sym, Q_diag, Q_sym_diag):
    """O(n^2) full delta computation — only called at init/restart."""
    Qx = Q_sym @ x
    contrib = Q_diag + Qx - Q_sym_diag * x
    return (1.0 - 2.0 * x) * contrib


def _flip_update(x, delta, k, Q_sym):
    """
    O(n) in-place delta update after flipping bit k.
    Must be called BEFORE x[k] is actually flipped.
    """
    flip_sign = 1.0 - 2.0 * float(x[k])
    delta += flip_sign * Q_sym[:, k] * (1.0 - 2.0 * x)
    delta[k] = -delta[k]
    x[k] = 1 - x[k]


# ═══════════════════════════════════════════════════════════════
# TABU SEARCH
# ═══════════════════════════════════════════════════════════════

def tabu_search_qubo(
    Q,
    max_iter=3000,
    tabu_tenure_range=(5, 15),
    restart_interval=500,
    two_flip_prob=0.1,
    verbose=False,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = Q.shape[0]
    Q_sym, Q_diag, Q_sym_diag = _symmetrise(Q)

    x = np.random.randint(0, 2, size=n, dtype=np.int8)
    current_cost = float(x @ Q @ x)
    best_x, best_cost = x.copy(), current_cost
    delta = _init_delta(x, Q_sym, Q_diag, Q_sym_diag)
    tabu_list = np.zeros(n, dtype=np.int64)

    for it in range(1, max_iter + 1):

        # Vectorised neighbour scan
        candidate_costs = current_cost + (1.0 - 2.0 * x) * delta
        is_tabu  = tabu_list > it
        aspire   = candidate_costs < best_cost
        allowed  = ~is_tabu | aspire

        best_move = None
        best_move_cost = np.inf

        if allowed.any():
            idx = int(np.argmin(np.where(allowed, candidate_costs, np.inf)))
            best_move_cost = candidate_costs[idx]
            best_move = (idx,)

        # Optional 2-flip (incremental)
        if random.random() < two_flip_prob:
            i, j = random.sample(range(n), 2)
            fi = 1.0 - 2.0 * float(x[i])
            dj = delta[j] + fi * Q_sym[i, j] * (1.0 - 2.0 * float(x[j]))
            cost_ij = (current_cost
                       + (1.0 - 2.0 * float(x[i])) * delta[i]
                       + (1.0 - 2.0 * float(x[j])) * dj)
            if cost_ij < best_move_cost:
                best_move_cost = cost_ij
                best_move = (i, j)

        if best_move is not None:
            tenure = random.randint(*tabu_tenure_range)
            for k in best_move:
                _flip_update(x, delta, k, Q_sym)
                tabu_list[k] = it + tenure
            current_cost = float(x @ Q @ x)

            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.copy()
                if verbose:
                    print(f"  Tabu iter {it:6d}  best={best_cost:.4f}")

        # Smart restart
        if it % restart_interval == 0:
            if random.random() < 0.5:
                x = best_x.copy()
                flip_idx = np.random.choice(n, max(1, n // 10), replace=False)
                x[flip_idx] = 1 - x[flip_idx]
            else:
                x = np.random.randint(0, 2, size=n, dtype=np.int8)
            current_cost = float(x @ Q @ x)
            delta = _init_delta(x, Q_sym, Q_diag, Q_sym_diag)
            tabu_list[:] = 0

    return best_x, best_cost


# ═══════════════════════════════════════════════════════════════
# SIMULATED ANNEALING
# ═══════════════════════════════════════════════════════════════

def sa_qubo(
    Q,
    max_iter=500_000,
    T_start=None,
    T_end=1e-4,
    restart_interval=100_000,
    verbose=False,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = Q.shape[0]
    Q_sym, Q_diag, Q_sym_diag = _symmetrise(Q)

    x = np.random.randint(0, 2, size=n, dtype=np.int8)
    current_cost = float(x @ Q @ x)
    best_x, best_cost = x.copy(), current_cost
    delta = _init_delta(x, Q_sym, Q_diag, Q_sym_diag)

    if T_start is None:
        T_start = float(2.0 * np.std(np.abs(delta)) + 1e-8)

    alpha = (T_end / T_start) ** (1.0 / max(max_iter - 1, 1))
    T = T_start

    for it in range(max_iter):
        i = random.randrange(n)
        dE = (1.0 - 2.0 * float(x[i])) * delta[i]

        if dE < 0 or random.random() < np.exp(-dE / (T + 1e-300)):
            _flip_update(x, delta, i, Q_sym)
            current_cost += dE
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.copy()
                if verbose:
                    print(f"  SA iter {it:8d}  T={T:.5f}  best={best_cost:.4f}")

        T *= alpha

        if it % restart_interval == 0 and it > 0:
            if random.random() < 0.5:
                x = best_x.copy()
                flip_idx = np.random.choice(n, max(1, n // 10), replace=False)
                x[flip_idx] = 1 - x[flip_idx]
            else:
                x = np.random.randint(0, 2, size=n, dtype=np.int8)
            current_cost = float(x @ Q @ x)
            delta = _init_delta(x, Q_sym, Q_diag, Q_sym_diag)

    return best_x, best_cost


# ═══════════════════════════════════════════════════════════════
# PARALLEL TEMPERING  (best quality for large, rugged landscapes)
# ═══════════════════════════════════════════════════════════════

def parallel_tempering(
    Q,
    n_replicas=8,
    steps_per_swap=500,
    total_swaps=2000,
    T_min=1e-3,
    T_max=None,
    verbose=True,
    seed=None,
):
    """
    Parallel Tempering (Replica Exchange Monte Carlo).

    Maintains n_replicas independent SA chains at temperatures
    T_min ... T_max (log-spaced).  Every `steps_per_swap` SA steps,
    adjacent replicas attempt to swap their states:

        P(swap i <-> i+1) = exp( (1/T_i - 1/T_{i+1}) * (E_{i+1} - E_i) )

    This allows the hot replicas to escape local minima while the
    cold replicas refine the best solutions found.

    Runs all replicas sequentially on one process — for true parallelism
    use run_parallel() which launches independent PT instances.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = Q.shape[0]
    Q_sym, Q_diag, Q_sym_diag = _symmetrise(Q)

    if T_max is None:
        # Auto: hot replica accepts ~75% of uphill moves initially
        sample_x = np.random.randint(0, 2, size=n, dtype=np.int8)
        sample_d = _init_delta(sample_x, Q_sym, Q_diag, Q_sym_diag)
        T_max = float(4.0 * np.std(np.abs(sample_d)) + 1e-6)

    # Log-spaced temperature ladder
    temps = np.geomspace(T_min, T_max, n_replicas)[::-1]  # hottest first
    if verbose:
        print(f"PT replicas: {n_replicas}  T range: {T_min:.4f} – {T_max:.4f}")

    # Initialise replicas
    states   = [np.random.randint(0, 2, size=n, dtype=np.int8) for _ in range(n_replicas)]
    costs    = [float(s @ Q @ s) for s in states]
    deltas   = [_init_delta(s, Q_sym, Q_diag, Q_sym_diag) for s in states]

    best_x    = states[np.argmin(costs)].copy()
    best_cost = min(costs)
    swap_accepts = 0
    swap_attempts = 0

    t0 = time.time()

    for swap_round in range(total_swaps):

        # Run `steps_per_swap` SA steps on each replica
        for r in range(n_replicas):
            x  = states[r]
            d  = deltas[r]
            T  = temps[r]
            c  = costs[r]

            for _ in range(steps_per_swap):
                i  = random.randrange(n)
                dE = (1.0 - 2.0 * float(x[i])) * d[i]
                if dE < 0 or random.random() < np.exp(-dE / T):
                    _flip_update(x, d, i, Q_sym)
                    c += dE

            costs[r] = c

            if c < best_cost:
                best_cost = c
                best_x    = x.copy()

        # Attempt swaps between adjacent replicas (random order to avoid bias)
        pairs = list(range(n_replicas - 1))
        random.shuffle(pairs)
        for r in pairs:
            swap_attempts += 1
            dE_swap = (costs[r+1] - costs[r]) * (1.0/temps[r] - 1.0/temps[r+1])
            if dE_swap < 0 or random.random() < np.exp(-dE_swap):
                swap_accepts += 1
                states[r], states[r+1] = states[r+1], states[r]
                costs[r],  costs[r+1]  = costs[r+1],  costs[r]
                deltas[r], deltas[r+1] = deltas[r+1], deltas[r]

        if verbose and swap_round % 200 == 0:
            rate = 100.0 * swap_accepts / max(1, swap_attempts)
            print(f"  PT swap {swap_round:5d}/{total_swaps}  "
                  f"best={best_cost:.4f}  swap_accept={rate:.1f}%  "
                  f"t={time.time()-t0:.1f}s")

    if verbose:
        rate = 100.0 * swap_accepts / max(1, swap_attempts)
        print(f"PT done. best={best_cost:.4f}  swap_accept_rate={rate:.1f}%  "
              f"time={time.time()-t0:.1f}s")

    return best_x, best_cost


# ═══════════════════════════════════════════════════════════════
# GREEDY INITIALISATION  (useful warm-start for any solver)
# ═══════════════════════════════════════════════════════════════

def greedy_init(Q):
    """
    Fast O(n^2) greedy: iteratively flip each bit if it reduces cost.
    One pass. Use as warm-start for Tabu/SA/PT.
    """
    n = Q.shape[0]
    Q_sym, Q_diag, Q_sym_diag = _symmetrise(Q)
    x = np.zeros(n, dtype=np.int8)
    delta = _init_delta(x, Q_sym, Q_diag, Q_sym_diag)
    for i in np.argsort(delta):          # process most-beneficial flip first
        if delta[i] < 0:
            _flip_update(x, delta, i, Q_sym)
    return x, float(x @ Q @ x)


# ═══════════════════════════════════════════════════════════════
# PARALLEL RUNNER  (spawn-safe)
# ═══════════════════════════════════════════════════════════════
#
# WHY THE ORIGINAL CAUSED INFINITE RECURSION
# -------------------------------------------
# On Windows and macOS, multiprocessing uses "spawn" by default:
# each worker starts a *fresh* Python interpreter and re-imports
# your main script.  If run_parallel() is called at module level
# (not inside `if __name__ == "__main__":`), the re-import calls
# run_parallel() again → infinite loop.
#
# HOW THIS IS FIXED HERE
# -----------------------
# 1. Workers are submitted as plain callables via
#    ProcessPoolExecutor, which is spawn-safe.
# 2. If spawning fails for any reason, we fall back to sequential
#    execution automatically — no crash.
# 3. The _worker function is defined at module level so it is
#    always picklable regardless of how qubo_solver was imported.
#
# WHAT YOU MUST DO IN YOUR MAIN SCRIPT
# --------------------------------------
# Wrap your call in the __main__ guard:
#
#   from qubo_solver import run_parallel
#
#   if __name__ == "__main__":          # <-- THIS LINE IS REQUIRED
#       best_x, best_cost = run_parallel(Q, solver="pt", n_workers=8)
#
# Without the guard, spawned workers re-import your script and
# call run_parallel() again before __main__ has finished.

def _worker(args):
    """Module-level worker — must stay here to be picklable."""
    solver_name, Q, kwargs, worker_seed = args
    kwargs = dict(kwargs)
    kwargs["seed"]    = worker_seed
    kwargs["verbose"] = False
    if solver_name == "tabu":
        return tabu_search_qubo(Q, **kwargs)
    elif solver_name == "sa":
        return sa_qubo(Q, **kwargs)
    elif solver_name == "pt":
        return parallel_tempering(Q, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def run_parallel(
    Q,
    solver="pt",
    n_workers=None,
    base_seed=0,
    solver_kwargs=None,
    verbose=True,
):
    """
    Run `solver` in parallel across `n_workers` independent processes.
    Each process uses a different random seed for diversity.
    Falls back to sequential execution if multiprocessing is unavailable.

    solver:        "tabu" | "sa" | "pt"
    n_workers:     defaults to CPU count (capped at 8 to avoid memory blow-up
                   for large Q matrices — override explicitly if needed)
    solver_kwargs: extra keyword args forwarded to the chosen solver

    IMPORTANT — your calling script must have:
        if __name__ == "__main__":
            best_x, best_cost = run_parallel(...)
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)
    if solver_kwargs is None:
        solver_kwargs = {}

    seeds = [base_seed + i for i in range(n_workers)]
    work  = [(solver, Q, solver_kwargs, s) for s in seeds]

    t0 = time.time()

    # --- try true parallelism first ---------------------------------
    try:
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
        ) as pool:
            if verbose:
                print(f"Launching {n_workers} parallel {solver.upper()} "
                      f"workers (context={ctx.get_start_method()}) …")
            futures = [pool.submit(_worker, w) for w in work]
            results = [f.result() for f in futures]

    except Exception as e:
        # Graceful fallback: run sequentially (common in notebooks /
        # interactive shells where spawn guard is absent)
        if verbose:
            print(f"[run_parallel] multiprocessing unavailable ({e}).")
            print(f"  Falling back to sequential execution of {n_workers} runs.")
        results = [_worker(w) for w in work]

    # --- pick best result -------------------------------------------
    costs     = [r[1] for r in results]
    best_idx  = int(np.argmin(costs))
    best_x, best_cost = results[best_idx]

    if verbose:
        print(f"\nResults ({time.time()-t0:.1f}s):")
        for i, (_, c) in enumerate(results):
            tag = "  <-- best" if i == best_idx else ""
            print(f"  worker {i}: cost={c:.4f}{tag}")
        print(f"Overall best: {best_cost:.4f}  ones={int(best_x.sum())}")

    return best_x, best_cost


# ═══════════════════════════════════════════════════════════════
# EXAMPLE / QUICK TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ----------------------------------------------------------------
    # NOTE: the `if __name__ == "__main__":` guard is REQUIRED.
    # Without it, spawned worker processes re-import this file and
    # call run_parallel() again before finishing — infinite recursion.
    # The same guard is required in YOUR main script when you import
    # and call run_parallel().
    # ----------------------------------------------------------------
    np.random.seed(0)
    n = 200

    A    = np.random.randn(n, n) * 0.05
    Q    = A.T @ A
    np.fill_diagonal(Q, Q.diagonal() - np.random.uniform(0, 0.5, n))

    diagnose_Q(Q)
    best_x, best_cost = run_parallel(
        Q,
        solver="pt",
        n_workers=4,
        solver_kwargs=dict(n_replicas=4, total_swaps=500, steps_per_swap=200),
    )