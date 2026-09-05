"""General augmented Newton systems for locating fold and Hopf bifurcation points,
following the standard construction (e.g. Kuznetsov, "Elements of Applied
Bifurcation Theory"): solve for the equilibrium, the degenerate eigenvector(s), and
the bifurcation parameter simultaneously, rather than sweeping the parameter and
inspecting eigenvalues of an independently-known equilibrium.

FINAL GUIDANCE, after investigation (PROJECT_LOG.md Sec. 16 has the full story):

- FOLD: do NOT use this module. `fold_residual`/`solve_fold` are kept only to
  document a real finding, not as a recommended tool. With the state x left free,
  the raw N1=2 system (Eq. 42) admits an extraneous equilibrium branch that
  violates the boiling-boundary definition h(lambda)=0 (confirmed directly: an
  8.9% violation on the branch Newton converges to from a perturbed seed near the
  known fold). An attempted fix (substituting lambda=Nsub/Npch as an explicit
  constraint row) made the Newton iteration numerically unstable (diverges to NaN)
  -- and chasing that fix led to a more fundamental conclusion anyway: once
  lambda=Nsub/Npch is enforced, l1 and u_i are immediately pinned too by the node
  equations, leaving x with *zero* remaining degrees of freedom. The classical
  Ledinegg fold is a non-invertibility of the scalar map Npch->Eu (Eq. 26), not a
  degeneracy of a genuinely 4-dimensional dynamical state. Use
  `tide.continuation.curves.fold_npch` (the closed-form approach) instead -- it is
  not a convenient approximation, it is the objectively correct method for this
  model's structure.
- HOPF: the genuinely correct approach (full dynamical Jacobian, evaluated at a
  known-valid closed-form equilibrium) is already implemented and validated in
  `tide.continuation.curves.hopf_npch` / `_leading_complex_real_part`. This
  module's `hopf_residual`/`solve_hopf` let Npch float freely the same way
  `fold_residual` does and have NOT been checked for the same extraneous-branch
  vulnerability -- do not trust them without first repeating the same h(lambda)
  check done for fold.

Eu (the external pressure/pump characteristic) is treated as the fixed control
parameter, and Npch as an unknown solved for jointly with the state and the
degenerate eigenvector -- this is what makes a fold detectable as a genuine
Jacobian degeneracy in the first place (see PROJECT_LOG.md Sec. 15 for why sweeping
Npch directly, with Eu recomputed from it each time, cannot see it at all).
"""

import jax
import jax.numpy as jnp

from tide.physics.clausse_lahey import state_derivative_n1_2

N_STATE = 4  # (l1, lambda, m, u_i) for the N1=2 model


def _f(x, npch, Nsub, Fr, Lam, ki, ke, Eu):
    params = dict(Nsub=Nsub, Npch=npch, Fr=Fr, Lam=Lam, ki=ki, ke=ke, Eu=Eu)
    return state_derivative_n1_2(x, params)


def _f_constrained(x, npch, Nsub, Fr, Lam, ki, ke, Eu):
    """f(x; npch) with the boiling-boundary definition h(lambda)=0 (i.e.
    lambda = Nsub/Npch) substituted in for the lambda-equation.

    WHY THIS IS NEEDED (see PROJECT_LOG.md Sec. 16): the raw node-position ODEs
    (Eq. 42's first line) only force l1=lambda/2 and u_i=lambda at steady state --
    they do NOT by themselves force lambda=Nsub/Npch. That relation is exactly the
    definition of the boiling boundary (h(lambda)=0, fluid reaches saturation) and
    was used *during* the derivation of the node equations (via the fixed-enthalpy
    node definition, Eq. 28) but does not survive as an independent, explicit
    equation in the final assembled system. An unconstrained Newton solve on the
    raw system can therefore converge to an extraneous root where lambda != Nsub/
    Npch -- confirmed directly: at one such root, h(lambda) = lambda - Nsub/Npch
    was -0.062 (an 8.9% violation), i.e. the fluid is still measurably subcooled at
    the point the solution calls "the boiling boundary." That is not a different
    physical regime; it fails the basic definition of the quantity it claims to be.

    Fix: replace the (physically incomplete) lambda-ODE component of f with the
    explicit, physically non-negotiable definition lambda - Nsub/Npch = 0."""
    fx = _f(x, npch, Nsub, Fr, Lam, ki, ke, Eu)
    lam = x[1]
    boiling_boundary_defn = lam - Nsub / npch
    return fx.at[1].set(boiling_boundary_defn)


def fold_residual(X, Nsub, Fr, Lam, ki, ke, Eu):
    """X = [x(4), v(4), npch] -- 9 unknowns, 9 equations.
    f(x; npch) = 0, J(x; npch) v = 0, v.v - 1 = 0.

    Deliberately uses the RAW (unconstrained) `_f`, matching the residual that
    originally surfaced the extraneous-branch finding in PROJECT_LOG.md Sec. 16 --
    this exists to document that finding (see test_augmented_systems.py), not as a
    recommended fold-finder. Use `tide.continuation.curves.fold_npch` instead."""
    x, v, npch = X[:4], X[4:8], X[8]

    def f_of_x(xx):
        return _f(xx, npch, Nsub, Fr, Lam, ki, ke, Eu)

    fx = f_of_x(x)
    J = jax.jacfwd(f_of_x)(x)
    Jv = J @ v
    norm_eq = jnp.dot(v, v) - 1.0
    return jnp.concatenate([fx, Jv, jnp.array([norm_eq])])


def fold_residual_constrained(X, Nsub, Fr, Lam, ki, ke, Eu):
    """Same as fold_residual, but using _f_constrained (lambda=Nsub/Npch enforced
    explicitly). Attempted fix, NOT working -- diverges to NaN from a perturbed
    seed (PROJECT_LOG.md Sec. 16). Kept only as a record of the attempt; do not
    build on this without first understanding why the Newton iteration is unstable
    with this residual (suspected: the algebraic constraint row changes the
    conditioning of the augmented Jacobian near the solution in a way plain,
    undamped Newton doesn't handle well)."""
    x, v, npch = X[:4], X[4:8], X[8]

    def f_of_x(xx):
        return _f_constrained(xx, npch, Nsub, Fr, Lam, ki, ke, Eu)

    fx = f_of_x(x)
    J = jax.jacfwd(f_of_x)(x)
    Jv = J @ v
    norm_eq = jnp.dot(v, v) - 1.0
    return jnp.concatenate([fx, Jv, jnp.array([norm_eq])])


def hopf_residual(X, Nsub, Fr, Lam, ki, ke, Eu, q):
    """X = [x(4), v_real(4), v_imag(4), omega, npch] -- 14 unknowns, 14 equations.
    f(x; npch) = 0, J vr + omega vi = 0, J vi - omega vr = 0,
    q.vr - 1 = 0, q.vi = 0  (q pins scale and rotational freedom of the eigenvector).

    Uses the RAW (unconstrained) `_f`, same caveat as fold_residual: with npch
    free, this has NOT been checked for the same extraneous-branch vulnerability
    found for fold (PROJECT_LOG.md Sec. 16). Prefer
    `tide.continuation.curves.hopf_npch`, which is validated and always evaluates
    the Jacobian at a known-valid closed-form equilibrium rather than letting x
    float freely."""
    x, vr, vi, omega, npch = X[:4], X[4:8], X[8:12], X[12], X[13]

    def f_of_x(xx):
        return _f(xx, npch, Nsub, Fr, Lam, ki, ke, Eu)

    fx = f_of_x(x)
    J = jax.jacfwd(f_of_x)(x)
    eq_real = J @ vr + omega * vi
    eq_imag = J @ vi - omega * vr
    pin1 = jnp.dot(q, vr) - 1.0
    pin2 = jnp.dot(q, vi)
    return jnp.concatenate([fx, eq_real, eq_imag, jnp.array([pin1, pin2])])


def newton_solve(residual_fn, X0, max_iter=100, tol=1e-12, damping=1.0):
    """Plain Newton's method (JAX-autodiff Jacobian, direct dense solve) on a
    generic residual_fn(X) -> vector. Returns (X_solution, residual_history)."""
    X = X0
    history = []
    J_fn = jax.jacobian(residual_fn)
    for _ in range(max_iter):
        r = residual_fn(X)
        res_norm = float(jnp.max(jnp.abs(r)))
        history.append(res_norm)
        if res_norm < tol:
            break
        Jm = J_fn(X)
        dX = jnp.linalg.solve(Jm, -r)
        X = X + damping * dX
    return X, history


def solve_fold(x0, v0, npch0, Nsub, Fr, Lam, ki, ke, Eu, **kwargs):
    X0 = jnp.concatenate([x0, v0, jnp.array([npch0])])
    residual_fn = lambda X: fold_residual(X, Nsub, Fr, Lam, ki, ke, Eu)
    X, history = newton_solve(residual_fn, X0, **kwargs)
    return dict(x=X[:4], v=X[4:8], npch=float(X[8]), residual_history=history)


def solve_hopf(x0, vr0, vi0, omega0, npch0, Nsub, Fr, Lam, ki, ke, Eu, q=None, **kwargs):
    if q is None:
        q = jnp.array([0.0, 0.0, 0.0, 1.0])  # pin on the u_i component by default
    X0 = jnp.concatenate([x0, vr0, vi0, jnp.array([omega0, npch0])])
    residual_fn = lambda X: hopf_residual(X, Nsub, Fr, Lam, ki, ke, Eu, q)
    X, history = newton_solve(residual_fn, X0, **kwargs)
    return dict(
        x=X[:4], v_real=X[4:8], v_imag=X[8:12],
        omega=float(X[12]), npch=float(X[13]), residual_history=history,
    )
