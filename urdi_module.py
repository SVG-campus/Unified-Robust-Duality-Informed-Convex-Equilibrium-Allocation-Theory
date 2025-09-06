"""
Unified Robust Duality-Informed Convex Equilibrium Allocation (reference implementation).

Objective:
    J(w) = 0.5 * w^T Σ w  -  μ^T w  +  ε * ||w||_2  +  0.5 * λ * ||w||_2^2
subject to:
    w >= 0,  sum(w) = 1

We solve with projected gradient descent on the simplex.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-12

def projected_simplex(v: np.ndarray, s: float = 1.0) -> np.ndarray:
    """Project v onto {w | w >= 0, sum w = s}. Algorithm from Duchi et al. (2008)."""
    v = np.asarray(v, dtype=float)
    if s <= 0:
        raise ValueError("s must be > 0")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    sw = w.sum()
    if sw <= 0 or not np.isfinite(sw):
        raise ValueError("projection failed")
    return w / sw

def objective_value(w: np.ndarray, Sigma: np.ndarray, mu: np.ndarray, epsilon: float = 0.0, lam: float = 0.0) -> float:
    w = np.asarray(w, dtype=float)
    S = np.asarray(Sigma, dtype=float)
    m = np.asarray(mu, dtype=float).ravel()
    quad = 0.5 * float(w @ S @ w)
    lin  = - float(m @ w)
    l2   = float(epsilon * np.linalg.norm(w))
    ridge= 0.5 * float(lam * (w @ w))
    return quad + lin + l2 + ridge

def _grad(w: np.ndarray, Sigma: np.ndarray, mu: np.ndarray, epsilon: float = 0.0, lam: float = 0.0) -> np.ndarray:
    S = np.asarray(Sigma, dtype=float)
    m = np.asarray(mu, dtype=float).ravel()
    # Subgradient of epsilon * ||w||_2 is epsilon * w / ||w||_2 (use _EPS for stability)
    norm_w = np.linalg.norm(w) + _EPS
    return S @ w - m + lam * w + (epsilon * w) / norm_w

def robust_equilibrium_solve(
    Sigma: np.ndarray,
    mu: np.ndarray,
    epsilon: float = 0.1,
    lam: float = 0.0,
    eta: float = 0.2,
    max_iter: int = 2000,
    tol: float = 1e-8,
    w0: np.ndarray | None = None,
    record_history: bool = False
):
    """
    Solve min J(w) s.t. w in simplex using projected gradient descent.
    Returns (w, history) if record_history else (w, None).
    """
    m = np.asarray(mu, dtype=float).ravel()
    n = m.size
    if w0 is None:
        w = np.ones(n) / n
    else:
        w = projected_simplex(np.asarray(w0, dtype=float), 1.0)

    history = [] if record_history else None
    last = objective_value(w, Sigma, mu, epsilon, lam)

    for t in range(1, max_iter + 1):
        g = _grad(w, Sigma, mu, epsilon, lam)
        w = projected_simplex(w - eta * g, 1.0)
        if record_history:
            val = objective_value(w, Sigma, mu, epsilon, lam)
            history.append(val)
            if abs(last - val) < tol:
                break
            last = val

    return (w, history) if record_history else (w, None)
