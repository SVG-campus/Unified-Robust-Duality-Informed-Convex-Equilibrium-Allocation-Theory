# Unified Robust Duality‑Informed Convex Equilibrium Allocation Theory (Paper 0)

This repository provides a practical, reproducible implementation that translates
the **duality‑informed convex equilibrium** allocation idea into a robust solver.

We solve for portfolio weights \(w\) on the simplex (non‑negative, sums to 1) that minimize a convex,
robust objective combining risk, return, and distributional robustness:

\[
J(w) = \tfrac{1}{2} w^\top \Sigma w \, - \, \mu^\top w \,+\, \epsilon\,\lVert w\rVert_2 \,+\, \tfrac{\lambda}{2}\,\lVert w\rVert_2^2
\]

- \(\Sigma\): covariance estimate (symmetric PSD)  
- \(\mu\): expected return vector  
- \(\epsilon\): robustness radius (duality‑linked worst‑case mean shift)  
- \(\lambda\): optional Tikhonov (ridge) regularization

A projected‑gradient solver enforces the simplex constraint at every step.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python examples.py
```

## Usage
```python
import numpy as np
from urdi_module import (
    projected_simplex, robust_equilibrium_solve, objective_value
)

# toy instance
rng = np.random.default_rng(0)
N = 4
A = rng.normal(0, 0.05, size=(N, N))
Sigma = A.T @ A + 1e-3 * np.eye(N)       # PSD covariance
mu = rng.normal(0.01, 0.005, size=N)     # expected returns

w, hist = robust_equilibrium_solve(
    Sigma=Sigma, mu=mu, epsilon=0.1, lam=0.0, eta=0.2, max_iter=2000, record_history=True
)

print("weights:", w, "sum:", w.sum(), "objective:", objective_value(w, Sigma, mu, epsilon=0.1, lam=0.0))
```

**Notes**
- Increasing **epsilon** typically yields **more uniform** allocations (higher entropy / lower \(\ell_2\) norm), reflecting worst‑case mean pessimism.
- Set **lam** > 0 to add extra convex curvature and improve numerical stability on noisy \(\Sigma\).

## Files included
- `urdi_module.py` — simplex projection, objective + gradient, and robust projected‑gradient solver.
- `tests/test_urdi_module.py` — invariants and robustness behavior tests.
- `tests/test_artifacts_exist.py` — checks presence of the paper PDF and Archive.zip (skips gracefully if missing).
- `.github/workflows/ci.yml` — run tests on push/PR.
- `.github/workflows/release.yml` — GitHub Release on tags (for Zenodo integration).
- `CITATION.cff` — includes your ORCID iD.
- `.zenodo.json` — Zenodo deposition metadata.
- `requirements.txt`, `examples.py`, `CHANGELOG.md`, `LICENSE-CODE`, `LICENSE-DOCS`, `.gitignore`.

## ORCID & Zenodo
- Author ORCID: **https://orcid.org/0009-0004-9601-5617**.
- With GitHub↔Zenodo connected, pushing a tag (e.g., `v0.1.0`) will mint a DOI.

**Publish checklist**
1. Commit code, paper, and tests.
2. Update versions in `CHANGELOG.md` and `CITATION.cff`.
3. Tag a release: `git tag v0.1.0 && git push --tags`.
4. When the DOI appears on Zenodo, add the badge here and add to `CITATION.cff -> identifiers`.
5. Ensure the DOI shows in your ORCID Works (add manually if needed).

## Citing
See `CITATION.cff`. Replace the DOI badge after the first release.
