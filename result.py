
import ast
import numpy as np
from math import floor
n=6
def yToX(t,n):
    i = floor(t/n) % n + 1
    j = t % n + 1
    k = floor(t/(n**2)) + 1
    return (i,j,k)
R = np.zeros((n,n))
with open("FinalResult.txt","r") as f:
    a = f.read()
    for i in range(len(a)):
        if a[i]=="1":
            (i1,j1,k1)=yToX(i,n)
            R[i1-1,j1-1]+=k1
print(R)



