import numpy as np
from turbines import dist
from math import floor

def yToX(t,n):
    i = floor(t/n) % n + 1
    j = t % n + 1
    k = floor(t/(n**2)) + 1
    return (i,j,k)

def xToY(i,j,k,n):
    return (i-1)*n + (j-1) + (k-1)*(n**2)



def create_Q(N,L):
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
    
    for i in range(1,N+1):
        for j in range(1,N+1):
            for k in range(1,5):
                t = xToY(i,j,k,N)
                Q[t,t] += cost[k] * D[i-1,j-1]
                if i != N-1 and i != N:
                    Q[t,t] += L[2] * (k**2) - 2*L[2]*k
               
                v = xToY(j,i,k,N)
                if i != N-1 and i != N:
                    Q[v,v] += L[2] * (k**2) + 2*L[2]*k
                

    for i in range(1,N+1):
        for k in range(1,5):
            for j in range(1,N+1):
                for j2 in range(1,j):
                    t = xToY(i,j,k,N)
                    v = xToY(i,j2,k,N)
                    Q[t,v] += 2*L[2]*(k**2)
                    
                    t = xToY(j,i,k,N)
                    v = xToY(j2,i,k,N)
                    Q[t,v] += 2*L[2]*(k**2)
                    

    for i in range(1,N+1):
        for j in range(1,N+1):
            for j2 in range(1,N+1):
                for k in range(1,5):
                    for k2 in range(1,k):
                        t = xToY(i,j,k,N)
                        v = xToY(i,j2,k2,N)
                        Q[t,v] += 2* L[2] * k*k2
                        
                        t = xToY(j,i,k,N)
                        v = xToY(j2,i,k,N)
                        Q[t,v] += 2*L[2]*(k*k2)

    for i in range(1,N+1):
        for j in range(1,N+1):
            for j2 in range(1,N+1):
                for k in range(1,5):
                    for k2 in range(1,5):
                        t = xToY(i,j,k,N)
                        v = xToY(i,j2,k2,N)
                        Q[t,v] -= 2 * L[2] * k * k2
                    
    # the two collectors penalties
    for i in range(N-1,N+1):
        c = 0 if i == N -1 else 1
        for k in range(1,5):
            for j in range(1,N-1):
            
                t = xToY(j,i,k,N)
                Q[t,t] += (k**2 - 46*k) * int(L[c])
                
                for j2 in range(1,j):
                    v = xToY(j2,i,k,N)
                    Q[t,v] += 2*(k**2) * int(L[c])
                    
                for j2 in range(1,N-1):
                    for k2 in range(1,k):
                        v = xToY(j2,i,k2,N)
                        Q[t,v] += 2*k*k2 * int(L[c])
                        

    #The fourth penalty (sum_i,j,k k * x_ijk = 46)
    for i in range(1,N+1):
        for j in range(1,N+1):
            for k in range(1,5):
                t = xToY(i,j,k,N)
                
                Q[t,t] += L[3] * (1 - 2*(N-2))

    for i in range(1,N+1):
        for j in range(1,N+1):
            for k in range(1,5):
                for k2 in range(1,k):
                    t = xToY(i,j,k,N)
                    v = xToY(i,j,k2,N)
                       
                    Q[t,v] += 2*L[3]
    
    for i in range(1,N+1):
        for j in range(1,N+1):
            for j2 in range(1,j):
                for k in range(1,5):
                    for k2 in range(1,5):
                        t = xToY(i,j,k,N)
                        v = xToY(i,j2,k2,N)
                        
                        Q[t,v] += L[3]
    for i in range(1,N+1):
        for i2 in range(1,i):
            for j in range(1,N+1):
                for j2 in range(1,N+1):
                    for k in range(1,5):
                        for k2 in range(1,5):
                            t = xToY(i,j,k,N)
                            v = xToY(i2,j2,k2,N)
                            
                            Q[t,v] += 2 * L[3]


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
    Q_new = Q_new / np.max(Q_new)
    return Q_new