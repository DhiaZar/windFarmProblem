from math import floor
import numpy as np
from turbines import dist

def yToX(t,n):
    i = floor(t/n) % n + 1
    j = t % n + 1
    k = floor(t/(n**2)) + 1
    return (i,j,k)

def xToY(i,j,k,n):
    return (i-1)*n + (j-1) + (k-1)*(n**2)

N = 48
L = 5
Q = np.zeros((N*N*4,N*N*4))
D = np.zeros((N,N))
Q_qubo = {}
distance_dict = {}
with open("Coordinates.csv",'r') as f:
    for line in f:
        tuple_elem = line.strip().split(',')
        if(tuple_elem[0] == 'i'):
            continue
        distance_dict[int(tuple_elem[0])] = (float(tuple_elem[1]),float(tuple_elem[2]))

for i in range(len(distance_dict)):
    for j in range(len(distance_dict)):
        D[i,j] = dist(distance_dict[i][0],distance_dict[i][1],distance_dict[j][0],distance_dict[j][1])


cost = {1:500,2:700,3:900,4:1100}

for i in range(1,N+1):
    for j in range(1,N+1):
        for k in range(1,5):
            t = xToY(i,j,k,N)
            Q[t,t] += cost[k] * D[i-1,j-1] + L * (k**2) - 2*L*k
            Q_qubo[(t,t)] += Q[t,t]
            v = xToY(j,i,k,N)
            Q[v,v] += L * (k**2) + 2*L*k
            Q_qubo[(v,v)] += Q[v,v]

for i in range(1,N+1):
    for k in range(1,5):
        for j in range(1,N+1):
            for j2 in range(1,j):
                t = xToY(i,j,k,N)
                v = xToY(i,j2,k,N)
                Q[t,v] += 2*L*(k**2)
                Q_qubo[(t,v)] += Q[t,v] 
                t = xToY(j,i,k,N)
                v = xToY(j2,i,k,N)
                Q[t,v] += 2*L*(k**2)
                Q_qubo[(t,v)] += Q[t,v]

for i in range(1,N+1):
    for j in range(1,N+1):
        for j2 in range(1,N+1):
            for k in range(1,5):
                for k2 in range(1,k):
                    t = xToY(i,j,k,N)
                    v = xToY(i,j2,k2,N)
                    Q[t,v] += 2* L * k*k2
                    Q_qubo[(t,v)] += Q[t,v]
                    t = xToY(j,i,k,N)
                    v = xToY(j2,i,k,N)
                    Q[t,v] += 2*L*(k*k2)
                    Q_qubo[(t,v)] += Q[t,v]

for i in range(1,N+1):
    for j in range(1,N+1):
        for j2 in range(1,N+1):
            for k in range(1,5):
                for k2 in range(1,5):
                    t = xToY(i,j,k,N)
                    v = xToY(i,j2,k2,N)
                    Q[t,v] += 2 * L * k * k2
                    Q_qubo[(t,v)] += Q[t,v]

for i in range(N-1,N+1):
    for k in range(1,5):
        for j in range(1,N-1):
            t = xToY(i,j,k,N)
            Q[t,t] += k**2 - 46*k
            Q_qubo[(t,t)] += Q[t,t]
            for j2 in range(1,j):
                v = xToY(i,j2,k,N)
                Q[t,v] += 2*(k**2)
                Q_qubo[(t,v)] += Q[t,v]
            for j2 in range(1,N-1):
                for k2 in range(1,k):
                    v = xToY(i,j2,k2,N)
                    Q[t,v] += 2*k*k2
                    Q_qubo[(t,v)] = Q[t,v]


