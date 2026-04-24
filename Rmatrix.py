from math import floor
import numpy as np
from turbines import dist
#import dimod
#from dwave.samplers import TabuSampler
from chatQUBOTabu import tabu_search_qubo
import torch
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig, ClassicalConfig
from qubosolver.solver import QuboSolver
import neal



def yToX(t,n):
    i = floor(t/n) % n + 1
    j = t % n + 1
    k = floor(t/(n**2)) + 1
    return (i,j,k)

def xToY(i,j,k,n):
    return (i-1)*n + (j-1) + (k-1)*(n**2)

N = 6
L = 1
Q_mat = np.zeros((N*N*4,N*N*4))
D = np.zeros((N,N))
Q_qubo_linear = {}
Q={}
distance_dict = {}
with open("Coordinates.csv",'r') as f:
    for line in f:
        tuple_elem = line.strip().split(',')
        if(tuple_elem[0] == 'i'):
            continue
        distance_dict[int(tuple_elem[0])] = (float(tuple_elem[1]),float(tuple_elem[2]))

for i in range(N):
    for j in range(N):
        D[i,j] = dist(distance_dict[i][0],distance_dict[i][1],distance_dict[j][0],distance_dict[j][1])


cost = {1:500,2:700,3:900,4:1100}
print("bch nebdew n3amrou fel dict ")

for i in range(1,N+1):
    for j in range(1,N+1):
        for k in range(1,5):
            t=xToY(i,j,k,N)
            if (t,t) not in Q.keys():
                Q[t,t] = 0  
            Q[t,t] += cost[k]*D[i-1,j-1]


for i in range(1,N-1):
    for k in range(1,5):
        for j in range(1,N+1):
            t=xToY(i,j,k,N)
            for j1 in range(1,j):
                v=xToY(i,j1,k,N)
                if (t,v) not in Q.keys():
                    Q[t,v] = 0  
                Q[t,v] +=  L*2*(k**2)
            if (t,t) not in Q.keys():
                Q[t,t] = 0  
            Q[t,t] += L*(k**2)


for i in range(1,N-1):
    for k in range(1,5):
        for j in range(1,N+1):
            t=xToY(j,i,k,N)
            for j1 in range(1,j):
                v=xToY(j1,i,k,N)
                if (t,v) not in Q.keys():
                    Q[t,v] = 0  
                Q[t,v] +=  L*2*(k**2)
            if (t,t) not in Q.keys():
                Q[t,t] = 0  
            Q[t,t] += L*(k**2)


for i in range(1,N-1):
    for j in range(1,N+1):
        for j1 in range(1,N+1):
            for k in range(1,5):
                t=xToY(i,j,k,N)
                for k1 in range(1,k):
                    v=xToY(i,j1,k1,N)
                    if (t,v) not in Q.keys():
                        Q[t,v] = 0  
                    Q[t,v] += L*2*k*k1


for i in range(1,N-1):
    for j in range(1,N+1):
        for j1 in range(1,N+1):
            for k in range(1,5):
                t=xToY(j,i,k,N)
                for k1 in range(1,k):
                    v=xToY(j1,i,k1,N)
                    if (t,v) not in Q.keys():
                        Q[t,v] = 0  
                    Q[t,v] += L*2*k*k1


for i in range(1,N-1):
    for j in range(1,N+1):
        for j1 in range(1,N+1):
            for k in range(1,5):
                t=xToY(i,j,k,N)
                for k1 in range(1,5):
                    v=xToY(i,j1,k1,N)
                    if (t,v) not in Q.keys():
                        Q[t,v] = 0  
                    Q[t,v] -= L*2*k*k1


for i in range (1,N-1):
    for j in range(1,N+1):
        for k in range(1,5):
            t=xToY(i,j,k,N)
            v=xToY(j,i,k,N)
            if (t,t) not in Q.keys():
                Q[t,t] = 0  
            Q[t,t] -= L*2*k
            if (v,v) not in Q.keys():
                Q[v,v] = 0  
            Q[v,v] +=L*2*k

for i in range(N-1,N+1):
    for k in range(1,5):
        for j in range(1,N):
            t=xToY(i,j,k,N)
            for j1 in range(1,j):
                v=xToY(i,j1,k,N)
                if (t,v) not in Q.keys():
                    Q[t,v] = 0  
                Q[t,v] +=  L*2*(k**2)
            if (t,t) not in Q.keys():
                Q[t,t] = 0  
            Q[t,t] += L*(k**2)
for i in range(N-1,N+1):
    for j in range(1,N):
        for j1 in range(1,N+1):
            for k in range(1,5):
                t=xToY(i,j,k,N)
                for k1 in range(1,k):
                    v=xToY(i,j1,k1,N)
                    if (t,v) not in Q.keys():
                        Q[t,v] = 0  
                    Q[t,v] += L*2*k*k1
for i in range(N-1,N+1):
    for k in range(1,5):
        for j in range(1,N):
            t=xToY(i,j,k,N)
            if (t,t) not in Q.keys():
                Q[t,t] = 0  
            Q[t,t] -= L*k
sampler = neal.SimulatedAnnealingSampler()

# Convert defaultdict back to a standard dictionary
# num_reads and num_sweeps should be scaled with N
sampleset = sampler.sample_qubo(dict(Q), num_reads=100, num_sweeps=1000)

# Output Results
best_sample = sampleset.first.sample
best_energy = sampleset.first.energy

print(f"Result: {best_sample}")