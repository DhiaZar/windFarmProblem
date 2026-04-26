import neal
from dwave_qbsolv import QBSolv
import csv

N = 100  # number of variables
target_sum = 46

Q = {}

# Linear terms
for i in range(N):
    Q[(i, i)] = 1 - 2 * target_sum  # = -91

# Quadratic terms
for i in range(N):
    for j in range(i+1, N):
        Q[(i, j)] = 2

sampler = neal.SimulatedAnnealingSampler()

response = QBSolv().sample_qubo(
    Q,
    solver=sampler,
    num_repeats=10
)

sample = response.first.sample
print("Sum:", sum(sample.values()))
print("Energy:", response.first.energy)
with open("solution_labeled.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "value"])

    for i in sorted(sample.keys()):
        writer.writerow([i, sample[i]])