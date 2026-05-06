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

def yToX(t,n):
    i = floor(t/n) % n + 1
    j = t % n + 1
    k = floor(t/(n**2)) + 1
    return (i,j,k)

def xToY(i,j,k,n):
    return (i-1)*n + (j-1) + (k-1)*(n**2)




if __name__ == "__main__":
    N = 10
    L = [1200 , 1200 , 1700, 5000, 100, 1500000] # Each element corresponds to a penalty function
    '''
    L[0] corresponds to first collector, 
    L[1] to the second  
    L[2] to inter-turbine connections
    L[3] for the sum of connections 
    L[4] for no-loop to self
    L[5] for one cable per pair
    ''' 
    Q = np.zeros((N*N*4,N*N*4))
    D = np.zeros((N,N))
    Q_qubo_linear = {}
    Q_qubo_quad = {}

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
    for i in range(N):
        D[i,i] = N**2 * np.max(D)
    cost = {1:500,2:700,3:900,4:1100}
    # L[4] = cost[4] * np.max(D) * 2
    print(f"no self loop lambda : {L[4]}")
    L1 = L[0]
    L2 = L[1]
    L3 = L[2]
    L4 = L[3]
    print("bch nebdew n3amrou fel dict ")
    for i in range(1,N+1):
        for j in range(1,N+1):
            for k in range(1,5):
                t = xToY(i,j,k,N)
                Q[t,t] += cost[k] * D[i-1,j-1]
                if((i != N-1 and i != N) and (j != N-1 and j!= N)):
                    Q[t,t] += int(L3) * (k**2) - 2*L3*k
                # if not(t in Q_qubo_linear.keys()):
                #     Q_qubo_linear[t] = 0
                # Q_qubo_linear[t] += cost[k] * D[i-1,j-1] + int(L3) * (k**2) - 2*L3*k
                v = xToY(j,i,k,N)
                if((i != N-1 and i != N) and (j != N-1 and j!= N)):
                    Q[v,v] += int(L3) * (k**2) + 2*int(L3)*k
                # if not(v in Q_qubo_linear.keys()):
                #     Q_qubo_linear[v] = 0
                # Q_qubo_linear[v] += int(L3) * (k**2) + 2*int(L3)*k

    for i in range(1,N+1):
        for k in range(1,5):
            for j in range(1,N+1):
                for j2 in range(1,j):
                    t = xToY(i,j,k,N)
                    v = xToY(i,j2,k,N)
                    #if((i != N-1 and i != N) and (j != N-1 and j!= N)):
                    Q[t,v] += 2*int(L3)*(k**2)
                    # if not( (t,v) in Q_qubo_quad.keys()):
                    #     Q_qubo_quad[(t,v)] = 0
                    # Q_qubo_quad[(t,v)] += 2*int(L3)*(k**2)
                    t = xToY(j,i,k,N)
                    v = xToY(j2,i,k,N)
                    #if((i != N-1 and i != N) and (j != N-1 and j!= N)):
                    Q[t,v] += 2*L3*(k**2)
                    # if not( (t,v) in Q_qubo_quad.keys()):
                    #     Q_qubo_quad[(t,v)] = 0
                    # Q_qubo_quad[(t,v)] += 2*int(L3)*(k**2)

    for i in range(1,N+1):
        for j in range(1,N+1):
            for j2 in range(1,N+1):
                for k in range(1,5):
                    for k2 in range(1,k):
                        t = xToY(i,j,k,N)
                        v = xToY(i,j2,k2,N)
                        Q[t,v] += 2* int(L3) * k*k2
                        # if not( (t,v) in Q_qubo_quad.keys()):
                        #     Q_qubo_quad[(t,v)] = 0
                        # Q_qubo_quad[(t,v)] += 2* int(L3) * k*k2
                        t = xToY(j,i,k,N)
                        v = xToY(j2,i,k,N)
                        Q[t,v] += 2*int(L3)*(k*k2)
                        # if not( (t,v) in Q_qubo_quad.keys()):
                        #     Q_qubo_quad[(t,v)] = 0
                        # Q_qubo_quad[(t,v)] += 2*int(L3)*(k*k2)

    for i in range(1,N+1):
        for j in range(1,N+1):
            for j2 in range(1,N+1):
                for k in range(1,5):
                    for k2 in range(1,5):
                        t = xToY(i,j,k,N)
                        v = xToY(i,j2,k2,N)
                        Q[t,v] -= 2 * int(L3) * k * k2
                        # if not( (t,v) in Q_qubo_quad.keys()):
                        #     Q_qubo_quad[(t,v)] = 0
                        # Q_qubo_quad[(t,v)] -= 2 * int(L3) * k * k2

    for i in range(N-1,N+1):
        c = 0 if i == N -1 else 1
        for k in range(1,5):
            for j in range(1,N-1):
                t = xToY(i,j,k,N)
                Q[t,t] += (k**2 - 46*k) * int(L[c])
                # if not( (t,v) in Q_qubo_quad.keys()):
                #     Q_qubo_quad[(t,v)] = 0
                # Q_qubo_quad[(t,t)] += int(L[c]) * (k**2 - 46*k)
                for j2 in range(1,j):
                    v = xToY(i,j2,k,N)
                    Q[t,v] += 2*(k**2) * int(L[c])
                    # if not( (t,v) in Q_qubo_quad.keys()):
                    #     Q_qubo_quad[(t,v)] = 0
                    # Q_qubo_quad[(t,v)] += 2*(k**2) * int(L[c])
                for j2 in range(1,N-1):
                    for k2 in range(1,k):
                        v = xToY(i,j2,k2,N)
                        Q[t,v] += 2*k*k2 * int(L[c])
                        # if not( (t,v) in Q_qubo_quad.keys()):
                        #     Q_qubo_quad[(t,v)] = 0
                        # Q_qubo_quad[(t,v)] = 2*k*k2 * int(L[c])

    #The fourth penalty (sum_i,j,k k * x_ijk = 46)
    for i in range(1,N+1):
        for j in range(1,N+1):
            for k in range(1,5):
                t = xToY(i,j,k,N)
                # if not( (t,t) in Q_qubo_quad.keys()):
                #     Q_qubo_quad[(t,t)] = 0
                # Q_qubo_quad[(t,t)] += L4 * ((k ** 2) - 2*(N-2)*k)  
                Q[t,t] += L4 * (1 - 2*(N-2))

    for i in range(1,N+1):
        for j in range(1,N+1):
            for k in range(1,5):
                for k2 in range(1,k):
                    t = xToY(i,j,k,N)
                    v = xToY(i,j,k2,N)
                    # if not( (t,v) in Q_qubo_quad.keys()):
                    #         Q_qubo_quad[(t,v)] = 0
                    # Q_qubo_quad[(t,v)] += 2*k*k2*L4    
                    Q[t,v] += 2*L4
    
    for i in range(1,N+1):
        for j in range(1,N+1):
            for j2 in range(1,j):
                for k in range(1,5):
                    for k2 in range(1,5):
                        t = xToY(i,j,k,N)
                        v = xToY(i,j2,k2,N)
                        # if not( (t,v) in Q_qubo_quad.keys()):
                        #     Q_qubo_quad[(t,v)] = 0
                        # Q_qubo_quad[(t,v)] += k*k2*L4
                        Q[t,v] += L4
    for i in range(1,N+1):
        for i2 in range(1,i):
            for j in range(1,N+1):
                for j2 in range(1,N+1):
                    for k in range(1,5):
                        for k2 in range(1,5):
                            t = xToY(i,j,k,N)
                            v = xToY(i2,j2,k2,N)
                            # if not( (t,v) in Q_qubo_quad.keys()):
                            #     Q_qubo_quad[(t,v)] = 0  
                            # Q_qubo_quad[(t,v)] += 2 * k * k2 * L4
                            Q[t,v] += 2 * L4


    #Fifth penalty x_ii = 0
    for i in range(1,N+1):
        for k in range(1,5):
            t = xToY(i,i,k,N)
            Q[t,t] += L[4]

    #Sixth penalty (at most one cable per pair)
    for i in range(1,N+1):
        for j in range(1,N+1):
            for k in range(1,5):
                for k2 in range(1,k):
                    t = xToY(i,j,k,N)
                    v = xToY(i,j,k2,N)
                    Q[t,v] += L[5]



    # tranform Q into upper triangular
    Q_new = np.triu(Q) + np.diag(np.diag(Q)) + np.triu(np.transpose(Q))
    # print(np.diag(np.diag(Q)))
    Q = Q_new
    # print(Q)
    print("number of zeros in diagonal")
    sum_diag = 0
    for i in range(len(Q)):
        sum_diag += 1 if Q[i,i] == 1 else 0
    print(f"there is {sum_diag} non-zero values in diagonal")
    print("bch nebdew fi normalization ")
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
    print("min:", min(vals))
    print("max:", max(vals))
    print("nonzero count:", sum(v != 0 for v in vals))
    print("bch nebdew fel Dwave")
    subqubo_size = 50
    # sampler = neal.SimulatedAnnealingSampler()
    sampler = SimulatedAnnealingSampler()
    response = QBSolv().sample_qubo(
        Q_norm,
        solver=sampler,
        solver_limit=50,
        num_repeats=20   # VERY important
    )
    print("best energy:", response.first.energy)
    print("sample size:", len(response.first.sample))
    sample = response.first.sample  
    with open("solution_labeled.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "value"])

        for i in sorted(sample.keys()):
            writer.writerow([i, sample[i]])
    # sampler = TabuSampler()

    # results = sampler.sample(bqm, label="wind farm ")
    # #Returns Dictionary index by index
    # smbl = results.first.sample
    # Print results
    # with open("FinalResult.txt","w") as file:
    #     file.write(str(smbl))
    # print("kamalna")