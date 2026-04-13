import numpy as np
import random


def tabu_search_qubo(Q, max_iter=2000, tabu_tenure_range=(5, 15),
                     restart_interval=500, two_flip_prob=0.1):
    """
    Solve QUBO: minimize x^T Q x where x ∈ {0,1}^n
    """

    n = Q.shape[0]

    # --- Initial solution ---
    x = np.random.randint(0, 2, size=n)
    
    def compute_cost(x):
        return x @ Q @ x

    current_cost = compute_cost(x)
    best_x = x.copy()
    best_cost = current_cost

    # Tabu list: stores last iteration when a variable was flipped
    tabu_list = np.zeros(n, dtype=int)

    # Precompute contribution vector for delta updates
    def compute_delta(x):
        delta = np.zeros(n)
        for i in range(n):
            delta[i] = Q[i, i] + 2 * np.sum(Q[i, :] * x) - 2 * Q[i, i] * x[i]
        return delta

    delta = compute_delta(x)

    for it in range(1, max_iter + 1):

        best_move = None
        best_move_value = float('inf')

        # --- Explore neighborhood ---
        for i in range(n):
            new_cost = current_cost + (1 - 2 * x[i]) * delta[i]

            is_tabu = tabu_list[i] > it

            # Aspiration: allow tabu if improves best
            if is_tabu and new_cost >= best_cost:
                continue

            if new_cost < best_move_value:
                best_move_value = new_cost
                best_move = (i,)

        # --- Optional 2-bit flip ---
        if random.random() < two_flip_prob:
            i, j = random.sample(range(n), 2)
            new_cost = compute_cost(x ^ np.eye(n, dtype=int)[i] ^ np.eye(n, dtype=int)[j])

            if new_cost < best_move_value:
                best_move_value = new_cost
                best_move = (i, j)

        # --- Apply best move ---
        if best_move is None:
            continue

        tenure = random.randint(*tabu_tenure_range)

        for i in best_move:
            x[i] = 1 - x[i]
            tabu_list[i] = it + tenure

        current_cost = compute_cost(x)

        # Update best solution
        if current_cost < best_cost:
            best_cost = current_cost
            best_x = x.copy()

        # Update delta after move
        delta = compute_delta(x)

        # --- Diversification (restart) ---
        if it % restart_interval == 0:
            x = np.random.randint(0, 2, size=n)
            current_cost = compute_cost(x)
            delta = compute_delta(x)
        print(it)
    return best_x, best_cost