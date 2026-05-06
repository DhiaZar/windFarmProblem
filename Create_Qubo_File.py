import numpy as np
def matrix_to_qubo(Q, filename="output.qubo", topology="0"):
    Q = np.array(Q)
    n = Q.shape[0]
    couplers = [(i, j, Q[i, j]) for i in range(n) for j in range(i+1, n) if Q[i, j] != 0]
    n_couplers = len(couplers)

    with open(filename, "w") as f:
        # Comments
        f.write("c Generated QUBO file\n")
        f.write(f"c Matrix size: {n}\n")

        # Problem line

        f.write("c Program Line\n")
        f.write(f"p qubo {topology} {n} {n} {n_couplers}\n")
        f.write("c -------- Diagonal Terms --------\n")
        # Node weights (diagonal)
        for i in range(n):
            if Q[i, i] != 0:
                f.write(f"{i} {i} {Q[i, i]}\n")
        f.write("c -------- Coupler Terms --------\n")
        # Couplers (upper triangle only)
        for i, j, val in couplers:
            f.write(f"{i} {j} {val}\n")

    print(f"QUBO file written to {filename}")