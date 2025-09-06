import cvxpy as cp
import numpy as np

# --------------------------
# Parameters
# --------------------------
np.random.seed(42)
n = 2         # Number of assets
N = 1000      # Number of scenarios
alpha = 0.95  # Inner CVaR level
gamma = 0.99  # Outer CVaR level

# Simulated scenario losses
losses = np.random.normal(loc=3.0, scale=1.0, size=(N, n))

# --------------------------
# Variables
# --------------------------
x = cp.Variable(n)
eta_alpha = cp.Variable()
eta_gamma = cp.Variable()
z_alpha = cp.Variable(N)
z_gamma = cp.Variable(N)

# --------------------------
# Constraints
# --------------------------
constraints = [
    cp.sum(x) == 1,
    x >= 0,
    z_alpha >= losses @ x - eta_alpha,
    z_alpha >= 0,
    z_gamma >= eta_alpha + (1 / ((1 - alpha) * N)) * cp.sum(z_alpha) - eta_gamma,
    z_gamma >= 0
]

# --------------------------
# Nested CVaR Objective
# --------------------------
objective = cp.Minimize(
    eta_gamma + (1 / ((1 - gamma) * N)) * cp.sum(z_gamma)
)

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GUROBI)

# --------------------------
# Output Results
# --------------------------
print("--- Nested CVaR Results ---")
print("x:", np.round(x.value, 4))
print("eta_alpha:", np.round(eta_alpha.value, 4))
print("eta_gamma:", np.round(eta_gamma.value, 4))
print("Optimal Value:", np.round(prob.value, 4))
