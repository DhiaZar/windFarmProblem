import torch
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig, ClassicalConfig
from qubosolver.solver import QuboSolver

# Define a simple 2×2 QUBO matrix (identity)
matrix = torch.eye(2)
instance = QUBOInstance(coefficients=matrix)

# Prepare solver configuration
cplex = ClassicalConfig(
    classical_solver_type="cplex",
    cplex_maxtime=120.0,
    cplex_log_path="cplex_run.log",
)
config = SolverConfig(
    classical=cplex,
    use_quantum=False
)

# Directly obtain solution via dispatcher
classical_solver = QuboSolver(instance, config)
solution = classical_solver.solve()

print("Bitstrings:", solution.bitstrings)
print("Costs:", solution.costs)