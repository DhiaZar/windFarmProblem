from math import floor
import numpy as np
from turbines import dist
from dwave.samplers import SimulatedAnnealingSampler
from claudeQUBOTabu import *
from dwave_qbsolv import QBSolv
import csv
from Create_Qubo_File import matrix_to_qubo
from solving import solve
# from solving import N
from visualizer import visualize

def yToX(t,n):
    i = floor(t/n) % n + 1
    j = t % n + 1
    k = floor(t/(n**2)) + 1
    return (i,j,k)

def xToY(i,j,k,n):
    return (i-1)*n + (j-1) + (k-1)*(n**2)

def penalty_check(sample, N):
    """
    Return (is_feasible, violation_score)
    Customize this to your constraints.
    """
    sum_weighted_cables = 0
    sum_weighted_cables_score = 0
    sum_connections_gen_1 = 0
    sum_connections_gen1_score = 0
    sum_connections_gen_2 = 0
    sum_connections_gen2_score = 0
    connected=set()

    not_connect_score = 0


    for t in range(len(sample)):
        i,j,k = yToX(t,N)
        if sample[t] == 1 :
            connected.add(i)
            connected.add(j)
        #total number of weighted cables
            sum_weighted_cables += 1
        #total number of connections to gen 2
            if j == N :
                sum_connections_gen_2 += k
        #total number of connections to gen 1
            if j == N-1:
                sum_connections_gen_1 += k
            if j == i :
                print(f"Unverified : Self Loop {i}")
    for i in range(1,N+1):
        if i not in connected :
            not_connect_score += 1
            # print(f"Unverified : Node not connected {i}")
    not_connect_score = not_connect_score**2  * N**2
    if sum_weighted_cables != N-2 :
        # print(f"Unverified : sum_weighted_cables = {sum_weighted_cables}")
        sum_weighted_cables_score = 0.7 if sum_weighted_cables > (N-2) else 2
    if sum_connections_gen_1 != (N-2)/2:
        print(f"Unverified : sum_connections_gen_1 = {sum_connections_gen_1}")
        sum_connections_gen1_score = 0.7 if sum_connections_gen_1 > (N-2)/2 else 2
        # print(sum_connections_gen1_score)
    if sum_connections_gen_2 != (N-2)/2:
        # print(f"Unverified : sum_connections_gen_2 = {sum_connections_gen_2}")
        sum_connections_gen2_score = 0.7 if sum_connections_gen_2 > (N-2)/2 else 2

    return sum_connections_gen1_score,sum_connections_gen2_score,sum_weighted_cables_score, not_connect_score

#Initializing
L = [1 , 1, 1, 1, 100, 150000]
n_iterations = 100
N = 6
for __ in range(n_iterations):
    sample = solve(N , L)
    l1,l2,l3,l4 = penalty_check(sample , N)
    print(l1,l2,l3)
    L[0] = L[0] * l1
    L[1] = L[1] * l2
    L[2] = L[2] * l3
    # L[3] = L[3] + l4
    # print(f"Intermediarie penalties {L}")
visualize(N)