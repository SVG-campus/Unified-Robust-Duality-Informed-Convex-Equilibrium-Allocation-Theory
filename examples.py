import numpy as np
from urdi_module import robust_equilibrium_solve, objective_value

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 5
    A = rng.normal(0, 0.05, size=(N, N))
    Sigma = A.T @ A + 1e-3 * np.eye(N)
    mu = rng.normal(0.01, 0.005, size=N)

    w, hist = robust_equilibrium_solve(Sigma, mu, epsilon=0.1, lam=0.0, eta=0.2, max_iter=2000, record_history=True)
    print("w:", w, "sum:", w.sum())
    print("final objective:", objective_value(w, Sigma, mu, epsilon=0.1, lam=0.0))
