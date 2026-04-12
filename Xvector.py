from math import floor
from collections import Counter

#DEBUGGING
turb = 46
gene = 2
n = turb + gene
X_size = 4 * (n**2)
bigX = []
with open("XvectorIndices.txt","w") as file:
    for t in range(X_size):
        i=(floor(t/n))%(n)+1
        j=(t%n)+1
        k=floor(t/(n**2))+1
        bigX.append(f"{i}-{j}-{k}")
        file.write(f"i: {i} , j:{j} , k:{k}\n")

##FUNCTION
def yToX(t,n):
    i=(floor(t/n))%(n)+1
    j=(t%n)+1
    k=floor(t/(n**2))+1
    return(i,j,k)
