from math import floor
import numpy as np
from turbines import dist
import dimod
from dwave.samplers import TabuSampler
from claudeQUBOTabu import *

def yToX(t,n):
    i = floor(t/n) % n + 1
    j = t % n + 1
    k = floor(t/(n**2)) + 1
    return (i,j,k)

def xToY(i,j,k,n):
    return (i-1)*n + (j-1) + (k-1)*(n**2)




if __name__ == "__main__":
    N = 48
    L = 0
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

    for i in range(len(distance_dict)):
        for j in range(len(distance_dict)):
            D[i,j] = dist(distance_dict[i][0],distance_dict[i][1],distance_dict[j][0],distance_dict[j][1])


    cost = {1:500,2:700,3:900,4:1100}
    print("bch nebdew n3amrou fel dict ")
    for i in range(1,N+1):
        for j in range(1,N+1):
            for k in range(1,5):
                t = xToY(i,j,k,N)
                Q[t,t] += cost[k] * D[i-1,j-1] + L * (k**2) - 2*L*k
                # if not(t in Q_qubo_linear.keys()):
                #     Q_qubo_linear[t] = 0
                # Q_qubo_linear[t] += cost[k] * D[i-1,j-1] + L * (k**2) - 2*L*k
                v = xToY(j,i,k,N)
                Q[v,v] += L * (k**2) + 2*L*k
                # if not(v in Q_qubo_linear.keys()):
                #     Q_qubo_linear[v] = 0
                # Q_qubo_linear[v] += L * (k**2) + 2*L*k

    for i in range(1,N+1):
        for k in range(1,5):
            for j in range(1,N+1):
                for j2 in range(1,j):
                    t = xToY(i,j,k,N)
                    v = xToY(i,j2,k,N)
                    Q[t,v] += 2*L*(k**2)
                    # if not( (t,v) in Q_qubo_quad.keys()):
                    #     Q_qubo_quad[(t,v)] = 0
                    # Q_qubo_quad[(t,v)] += 2*L*(k**2)
                    t = xToY(j,i,k,N)
                    v = xToY(j2,i,k,N)
                    Q[t,v] += 2*L*(k**2)
                    # if not( (t,v) in Q_qubo_quad.keys()):
                    #     Q_qubo_quad[(t,v)] = 0
                    # Q_qubo_quad[(t,v)] += 2*L*(k**2)

    for i in range(1,N+1):
        for j in range(1,N+1):
            for j2 in range(1,N+1):
                for k in range(1,5):
                    for k2 in range(1,k):
                        t = xToY(i,j,k,N)
                        v = xToY(i,j2,k2,N)
                        Q[t,v] += 2* L * k*k2
                        # if not( (t,v) in Q_qubo_quad.keys()):
                        #     Q_qubo_quad[(t,v)] = 0
                        # Q_qubo_quad[(t,v)] += 2* L * k*k2
                        t = xToY(j,i,k,N)
                        v = xToY(j2,i,k,N)
                        Q[t,v] += 2*L*(k*k2)
                        # if not( (t,v) in Q_qubo_quad.keys()):
                        #     Q_qubo_quad[(t,v)] = 0
                        # Q_qubo_quad[(t,v)] += 2*L*(k*k2)

    for i in range(1,N+1):
        for j in range(1,N+1):
            for j2 in range(1,N+1):
                for k in range(1,5):
                    for k2 in range(1,5):
                        t = xToY(i,j,k,N)
                        v = xToY(i,j2,k2,N)
                        Q[t,v] += 2 * L * k * k2
                        # if not( (t,v) in Q_qubo_quad.keys()):
                        #     Q_qubo_quad[(t,v)] = 0
                        # Q_qubo_quad[(t,v)] += 2 * L * k * k2

    for i in range(N-1,N+1):
        for k in range(1,5):
            for j in range(1,N-1):
                t = xToY(i,j,k,N)
                Q[t,t] += k**2 - 46*k
                # if not( (t,v) in Q_qubo_quad.keys()):
                #     Q_qubo_quad[(t,v)] = 0
                # Q_qubo_quad[(t,t)] += k**2 - 46*k
                for j2 in range(1,j):
                    v = xToY(i,j2,k,N)
                    Q[t,v] += 2*(k**2)
                    # if not( (t,v) in Q_qubo_quad.keys()):
                    #     Q_qubo_quad[(t,v)] = 0
                    # Q_qubo_quad[(t,v)] += 2*(k**2)
                for j2 in range(1,N-1):
                    for k2 in range(1,k):
                        v = xToY(i,j2,k2,N)
                        Q[t,v] += 2*k*k2
                        # if not( (t,v) in Q_qubo_quad.keys()):
                        #     Q_qubo_quad[(t,v)] = 0
                        # Q_qubo_quad[(t,v)] = 2*k*k2

    print("bch nebdew fi d-wave ")


    diagnose_Q(Q)
    solution, cost = run_parallel(Q,solver='pt',n_workers=4,
        solver_kwargs=dict(
            n_replicas=8,
            total_swaps=3000,
            steps_per_swap=500,
        ))
    solution_list = list(solution)
    with open("FinalResult.txt","w") as file:
        for t in solution_list:
            file.write(str(int(t)))

    print(solution)

    # bqm = dimod.BinaryQuadraticModel(Q_qubo_linear,Q_qubo_quad,1200,dimod.BINARY)

    # sampler = TabuSampler()

    # results = sampler.sample(bqm, label="wind farm ")
    # #Returns Dictionary index by index
    # smbl = results.first.sample
    # Print results
    # with open("FinalResult.txt","w") as file:
    #     file.write(str(smbl))
    # print("kamalna")