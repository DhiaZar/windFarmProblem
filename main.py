from math import floor
import numpy as np
from turbines import dist
from dwave.samplers import SimulatedAnnealingSampler
from claudeQUBOTabu import *
from dwave_qbsolv import QBSolv
import csv
from Create_Qubo_File import matrix_to_qubo
from matrixFill import Solve
from matrixFill import N

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
    sum_connections_gen_1 = 0
    sum_connections_gen_2 = 0
    connected=set()
    for t in range(len(sample)):
        i,j,k = yToX(t,N)
        if sample[t] == 1 :
            connected.add(i)
            connected.add(j)
        #total number of weighted cables
            sum_weighted_cables += 1
        #total number of connections to gen 2
            if j == N or i == N:
                sum_connections_gen_2 += 1
        #total number of connections to gen 1
            if j == N-1 or i == N-1:
                sum_connections_gen_1 += 1
            if j == i :
                print(f"Unverified : Self Loop {i}")
    for i in range(1,N+1):
        if i not in connected :
            print(f"Unverified : Node not connected {i}")

    if sum_weighted_cables != N-2 :
        print(f"Unverified : sum_weighted_cables = {sum_weighted_cables}")
    if sum_connections_gen_1 != (N-2)/2:
        print(f"Unverified : sum_connections_gen_1 = {sum_connections_gen_1}")
    if sum_connections_gen_2 != (N-2)/2:
        print(f"Unverified : sum_connections_gen_2 = {sum_connections_gen_2}")

L = [1200 , 1200, 5000, 5000, 100, 150000]
sample = Solve(N , L)
penalty_check(sample , N)