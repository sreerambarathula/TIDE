"""Direct double-zero-eigenvalue solver for genuine Bogdanov-Takens points --
supersedes the "fold curve + coalescence crosses zero" method used to find
Points B/C originally (bt_point_data.py's NSUB_BT/NPCH_BT, PROJECT_LOG.md
Sec. 37/40).

An external audit (documented in PROJECT_LOG.md, post-manuscript-draft
review) found that those coordinates are NOT actually double-zero points:
direct eigenvalue evaluation at (NSUB_BT, NPCH_BT) gives two DISTINCT real
eigenvalues (+0.0439, -0.0451 for Point B), not a repeated zero. Root
cause: the original method fixed Npch = Eq.-26-fold(Nsub) as a CONSTRAINT
(assuming the fold curve passes through the true double-zero point) and
searched only along that 1D curve. It does not, even in the Lambda->0
limit -- confirmed directly here, the gap between the true double-zero
Npch and the Eq. 26 fold value does NOT shrink to zero as Lambda->0
(~1.6e-3 to ~1.8e-3 residual gap, stable). This is the same class of
subtlety already documented elsewhere in this project (Sec. 16): the
Eq. 26 fold is a purely algebraic/kinematic condition (Eu's Npch-extremum
from the steady-state relation) and is not guaranteed to coincide exactly
with a genuine dynamical-Jacobian eigenvalue condition.

This module instead solves DIRECTLY for the double-zero condition as an
honest 2D root-find: find (Nsub, Npch) such that the "coalescing" pair of
eigenvalues has BOTH sum=0 and product=0 (equivalent to both eigenvalues
individually zero). Sum and product of a pair are used, not the raw
eigenvalues themselves, because they remain smooth/analytic through the
real-to-complex-conjugate branch point where the raw eigenvalues are not
differentiable -- solving on the raw eigenvalues directly was tried first
and diverges (confirmed: Newton on raw (lambda_a, lambda_b) blows up to
NaN within a few steps, exactly at the ill-conditioning this reformulation
avoids).

The "coalescing pair" at each trial point is identified as the two
eigenvalues of SMALLEST COMPLEX MAGNITUDE (not smallest |imaginary part|,
which incorrectly picks up other, unrelated real eigenvalues from the
node-discretization spectrum with imag=0 but large |real| -- confirmed
directly: at N1=16 there are 4 purely-real eigenvalues at the reference
point, two around -100/-200 from node damping and two near zero; smallest
magnitude unambiguously isolates the near-zero pair and continues to do so
robustly across the branch point into the complex regime)."""

import numpy as np

from tide.continuation.pseudo_arclength import _eigvals_jit


def _coalescing_pair(nsub, npch, Fr, Lam, ki, ke, N1):
    eigs = np.array(_eigvals_jit(nsub, npch, Fr, Lam, ki, ke, N1))
    idx = np.argsort(np.abs(eigs))[:2]
    return eigs[idx]


def _sum_product_residual(x, Fr, Lam, ki, ke, N1):
    nsub, npch = x
    pair = _coalescing_pair(nsub, npch, Fr, Lam, ki, ke, N1)
    return np.array([(pair[0] + pair[1]).real, (pair[0] * pair[1]).real])


def find_double_zero_point(x0, Fr, Lam, ki, ke, N1, h=1e-6, tol=1e-11,
                            max_iter=60, damping=0.7):
    """2D Newton solve for a genuine double-zero (Bogdanov-Takens) point,
    starting from an initial guess x0=(nsub, npch). Returns
    (nsub, npch, converged, residual, n_iter)."""
    x = np.array(x0, dtype=float)
    for it in range(max_iter):
        f = _sum_product_residual(x, Fr, Lam, ki, ke, N1)
        if np.max(np.abs(f)) < tol:
            return x[0], x[1], True, f, it
        J = np.zeros((2, 2))
        for j in range(2):
            xp = x.copy()
            xp[j] += h
            fp = _sum_product_residual(xp, Fr, Lam, ki, ke, N1)
            J[:, j] = (fp - f) / h
        dx = np.linalg.solve(J, -f)
        x = x + damping * dx
    f = _sum_product_residual(x, Fr, Lam, ki, ke, N1)
    return x[0], x[1], np.max(np.abs(f)) < 1e-6, f, max_iter
