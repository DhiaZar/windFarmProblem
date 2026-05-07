from math import floor
import numpy as np
from turbines import dist
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from claudeQUBOTabu import *
from dwave_qbsolv import QBSolv
import neal
import csv
from Create_Qubo_File import matrix_to_qubo
from create_Q import create_Q

def yToX(t,n):
    i = floor(t/n) % n + 1
    j = t % n + 1
    k = floor(t/(n**2)) + 1
    return (i,j,k)

def xToY(i,j,k,n):
    return (i-1)*n + (j-1) + (k-1)*(n**2)




def solve(N,L):

    Q = create_Q(N,L)

    # print("number of zeros in diagonal")
    sum_diag = 0
    for i in range(len(Q)):
        sum_diag += 1 if Q[i,i] != 0 else 0
    # print(f"there is {sum_diag} non-zero values in diagonal")
    # print("bch nebdew fi normalization ")
    Q_dict = {}

    n = Q.shape[0]

    for i in range(n):
        for j in range(n):
            if Q[i, j] != 0:
                Q_dict[(i, j)] = float(Q[i, j])
    vals = list(Q_dict.values())
    max_abs = max(abs(v) for v in Q_dict.values())
    Q_norm = {
        k: v / max_abs
        for k, v in Q_dict.items()
    }
    Q = Q/np.max(Q)
    vals = list(Q_norm.values())
    matrix_to_qubo(Q)
    # print("min:", min(vals))
    # print("max:", max(vals))
    # print("nonzero count:", sum(v != 0 for v in vals))

    subqubo_size = 50
    sampler = SimulatedAnnealingSampler()
    response = QBSolv().sample_qubo(
        Q_norm,
        solver=sampler,
        solver_limit=50,
        num_repeats=20   # VERY important
    )
    # print("best energy:", response.first.energy)
    # print("sample size:", len(response.first.sample))
    sample = response.first.sample  
    with open("solution_labeled.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "value"])

        for i in sorted(sample.keys()):
            writer.writerow([i, sample[i]])
    return sample