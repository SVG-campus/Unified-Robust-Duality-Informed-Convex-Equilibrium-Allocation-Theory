import numpy as np
from urdi_module import projected_simplex, robust_equilibrium_solve, objective_value

def test_projection_invariants():
    rng = np.random.default_rng(0)
    v = rng.normal(size=7)
    w = projected_simplex(v)
    assert np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0)

def test_solver_invariants_and_monotonicity():
    rng = np.random.default_rng(1)
    N = 5
    A = rng.normal(0, 0.05, size=(N, N))
    Sigma = A.T @ A + 1e-3 * np.eye(N)
    mu = rng.normal(0.01, 0.005, size=N)

    w, hist = robust_equilibrium_solve(Sigma, mu, epsilon=0.1, lam=0.0, eta=0.2, max_iter=1000, record_history=True)
    assert np.all(w >= 0) and np.isclose(w.sum(), 1.0)
    # objective should not blow up; last value should be finite and <= first few iterations
    assert np.isfinite(hist[-1])
    assert hist[0] >= hist[-1] - 1e-8

def test_higher_epsilon_more_uniform():
    rng = np.random.default_rng(2)
    N = 6
    A = rng.normal(0, 0.05, size=(N, N))
    Sigma = A.T @ A + 1e-3 * np.eye(N)
    mu = rng.normal(0.01, 0.01, size=N)
    w_low, _ = robust_equilibrium_solve(Sigma, mu, epsilon=0.01, lam=0.0, eta=0.2, max_iter=1500, record_history=True)
    w_high, _ = robust_equilibrium_solve(Sigma, mu, epsilon=0.3,  lam=0.0, eta=0.2, max_iter=1500, record_history=True)
    # variance across weights should shrink as epsilon increases
    assert w_high.var() <= w_low.var() + 1e-9
