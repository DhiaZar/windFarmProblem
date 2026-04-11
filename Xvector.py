from math import floor
from collections import Counter

turb = 46
gene = 2
n = turb + gene

X_size = 4 * (n+1) * (n)

bigX = []


with open("XvectorIndices.txt","w") as file:
    for t in range(X_size):
        i = (floor(t/4) % n) + 1
        j = (floor(t/(4*(n+1)))) + 1
        k = t % 4 +1
        bigX.append(f"{i}-{j}-{k}")
        file.write(f"i: {i} , j:{j} , k:{k}\n")

counts = Counter(bigX)
duplicates = list(filter(lambda x: counts[x] > 1, counts))
with open("XvectorDuplicates.txt","w") as file:
    for i in duplicates:
        file.write(i + "\n")