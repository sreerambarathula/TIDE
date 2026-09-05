"""The Phase 3 payoff measurement (PROJECT_LOG.md Sec. 31-32): does the
standard MLP surrogate's low, uniform field error (Sec. 30) coexist with a
boundary error that spikes near the codim-2 point? Pass/fail criterion
pre-registered in Sec. 31 BEFORE the first analysis was run, per Sec. 6's own
rule.

Sec. 32 rewrote this module after a user-requested critique surfaced several
real methodological problems in the first version (all listed and fixed
there, not just noted): boundary extraction upgraded from grid+interpolation
to Newton/bisection precision matching the ground-truth tools' own standard;
a local-training-data-density feature added for a second mechanism
hypothesis (Sec. 32's gradient-based mechanism test came back with the wrong
sign); statistics moved to eval/stats.py for proper seed-level (not
per-Nsub-point) replication, since treating every (seed, Nsub) evaluation as
an independent sample was pseudo-replication.

Two tests:
  1. Primary (decoupling): boundary error near the codim-2 point vs. far from
     it, compared against field error near vs. far, over the same windows.
  2. Secondary (mechanism): does local Hopf boundary error correlate with
     1/||grad g|| (the classical ill-conditioning hypothesis) or with local
     training-data sparsity (the alternative, ML-standard hypothesis)?
"""

import jax
import numpy as np

from tide.physics.ledinegg_curve import euler_number
from tide.surrogates.data_generation import _boundaries_at, _leading_real_part


def _extract_surrogate_fold(surrogate, nsub, npch_grid, tol=1e-6, max_iter=60):
    """Fold = argmax of predicted Eu over Npch, located by bisecting
    d(Eu_pred)/d(Npch)=0 (via jax.grad on the surrogate's differentiable
    scalar predictor), seeded from a coarse grid argmax for the bracket --
    the same "grid-seed, then Newton/bisect" pattern curves.fold_npch uses
    for the ground truth (PROJECT_LOG.md Sec. 15's fix), applied here to the
    surrogate so both sides of the comparison are extracted to comparable
    precision (Sec. 32; the first version used only a 3-point parabolic
    fit, adequate but less precise than the ground truth's own standard)."""
    nsub_arr = np.full_like(npch_grid, nsub)
    eu_pred, _ = surrogate.predict(nsub_arr, npch_grid)
    eu_pred = np.array(eu_pred)
    imax = int(np.argmax(eu_pred))
    if imax == 0 or imax == len(npch_grid) - 1:
        return None

    deu = jax.grad(lambda npch: surrogate.predict_scalar(nsub, npch)[0])
    lo, hi = npch_grid[imax - 1], npch_grid[imax + 1]
    f_lo, f_hi = float(deu(lo)), float(deu(hi))
    if f_lo * f_hi > 0:
        return float(npch_grid[imax])  # grid resolution already tight enough
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        f_mid = float(deu(mid))
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _extract_surrogate_hopf(surrogate, nsub, npch_grid, tol=1e-6, max_iter=60):
    """Hopf = zero-crossing of predicted g over Npch, located by bisection
    (not linear interpolation, for the same precision-matching reason as
    _extract_surrogate_fold).

    npch_grid MUST already be restricted to the narrow band around the true
    Hopf point (same (fold*0.85, fold*1.4) convention as
    data_generation._boundaries_at) -- confirmed directly (not assumed) that
    the surrogate correctly reproduced the SAME two real sign changes the
    ground truth has (Sec. 27's "excursive mode eigenvalue reshuffling"
    crossing near fold*0.8, and the true Hopf crossing near fold*1.1): an
    earlier version searched from nsub*1.02 and consistently locked onto the
    first (wrong, excursive-related) crossing, producing a spurious ~12-unit
    "boundary error" that was actually a methodology mismatch, not a
    surrogate failure."""
    nsub_arr = np.full_like(npch_grid, nsub)
    _, g_pred = surrogate.predict(nsub_arr, npch_grid)
    g_pred = np.array(g_pred)
    bracket = None
    for i in range(len(npch_grid) - 1):
        if g_pred[i] * g_pred[i + 1] < 0:
            bracket = (npch_grid[i], npch_grid[i + 1])
            break
    if bracket is None:
        return None

    def g_scalar(npch):
        return float(surrogate.predict_scalar(nsub, npch)[1])

    lo, hi = bracket
    f_lo, f_hi = g_scalar(lo), g_scalar(hi)
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        f_mid = g_scalar(mid)
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _true_gradient_norm_at_hopf(nsub, npch_hopf, Fr, Lam, ki, ke, N1, h=1e-3):
    """||grad g|| at the true Hopf point via central finite differences --
    not autodiff, deliberately: this project's own memory notes eigenvalue-
    derivative blowup near codim-2 points, and this is a diagnostic quantity,
    not a training signal, so there is no need to risk that here."""

    def g(n, p):
        val = _leading_real_part(n, p, Fr, Lam, ki, ke, N1)
        return val if val is not None else np.nan

    dg_dnsub = (g(nsub + h, npch_hopf) - g(nsub - h, npch_hopf)) / (2 * h)
    dg_dnpch = (g(nsub, npch_hopf + h) - g(nsub, npch_hopf - h)) / (2 * h)
    return float(np.hypot(dg_dnsub, dg_dnpch))


def _nearest_train_distance(nsub, npch, train_data, nsub_scale, npch_scale):
    """Distance (in scale-normalized units, so Nsub and Npch contribute
    comparably despite different ranges) from (nsub, npch) to the nearest
    training point -- the standard ML-native explanation for surrogate error
    (sparser local coverage -> worse fit), tested here as an alternative to
    the gradient-based mechanism hypothesis (Sec. 32, since Sec. 31's
    gradient test came back with the wrong sign)."""
    d_nsub = (train_data["nsub"] - nsub) / nsub_scale
    d_npch = (train_data["npch"] - npch) / npch_scale
    return float(np.sqrt(np.min(d_nsub**2 + d_npch**2)))


def run_decoupling_analysis(surrogate, Fr, Lam, ki, ke, N1, nsub_star,
                             nsub_grid, train_data=None,
                             npch_grid_points=2000, npch_mult=3.0):
    """Runs the full pre-registered analysis (PROJECT_LOG.md Sec. 31-32) over
    an independent Nsub grid. Returns a dict of per-Nsub arrays plus the
    near/far aggregate comparison. If train_data is given, also computes the
    nearest-training-point distance at each evaluation point (Sec. 32's
    alternative mechanism hypothesis)."""
    if train_data is not None:
        nsub_scale = float(np.std(train_data["nsub"]))
        npch_scale = float(np.std(train_data["npch"]))

    rows = []
    for nsub in nsub_grid:
        true_fold, true_hopf = _boundaries_at(nsub, Fr, Lam, ki, ke, N1)
        if true_fold is None or true_hopf is None:
            continue
        npch_grid = np.linspace(nsub * 1.02, nsub * npch_mult, npch_grid_points)
        # Hopf extraction uses the SAME narrow band as the ground-truth tool
        # -- a wide grid picks up a second, real-but-different sign change
        # (Sec. 27's excursive-mode eigenvalue reshuffling).
        npch_grid_hopf = np.linspace(true_fold * 0.85, true_fold * 1.4, npch_grid_points)

        surr_fold = _extract_surrogate_fold(surrogate, nsub, npch_grid)
        surr_hopf = _extract_surrogate_hopf(surrogate, nsub, npch_grid_hopf)
        if surr_fold is None or surr_hopf is None:
            continue

        # Field error over a LOCAL NEIGHBORHOOD (not just the single exact
        # true_hopf point): evaluating field error only AT the root itself
        # conflates it with the very quantity boundary error measures.
        npch_local = np.linspace(true_fold * 0.85, true_fold * 1.4, 25)
        eu_true_local = np.array([euler_number(p, nsub, Fr, Lam, ki, ke) for p in npch_local])
        g_true_local = [_leading_real_part(nsub, p, Fr, Lam, ki, ke, N1) for p in npch_local]
        nsub_local = np.full_like(npch_local, nsub)
        eu_pred_local, g_pred_local = surrogate.predict(nsub_local, npch_local)
        eu_pred_local, g_pred_local = np.asarray(eu_pred_local), np.asarray(g_pred_local)

        eu_field_err = float(np.sqrt(np.mean((eu_pred_local - eu_true_local) ** 2)))
        g_true_arr = np.array([v if v is not None else np.nan for v in g_true_local])
        g_field_err = float(np.sqrt(np.nanmean((g_pred_local - g_true_arr) ** 2))) if np.any(np.isfinite(g_true_arr)) else np.nan

        grad_norm = _true_gradient_norm_at_hopf(nsub, true_hopf, Fr, Lam, ki, ke, N1)

        row = dict(
            nsub=nsub, true_fold=true_fold, true_hopf=true_hopf,
            surr_fold=surr_fold, surr_hopf=surr_hopf,
            fold_error=abs(surr_fold - true_fold), hopf_error=abs(surr_hopf - true_hopf),
            eu_field_err=eu_field_err, g_field_err=g_field_err,
            grad_norm_at_hopf=grad_norm,
        )
        if train_data is not None:
            row["train_dist_at_hopf"] = _nearest_train_distance(
                nsub, true_hopf, train_data, nsub_scale, npch_scale)
        rows.append(row)

    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}

    near = np.abs(data["nsub"] - nsub_star) <= 1.0
    far = np.abs(data["nsub"] - nsub_star) >= 4.0

    summary = dict(
        n_near=int(near.sum()), n_far=int(far.sum()),
        fold_error_near=float(np.mean(data["fold_error"][near])) if near.any() else np.nan,
        fold_error_far=float(np.mean(data["fold_error"][far])) if far.any() else np.nan,
        hopf_error_near=float(np.mean(data["hopf_error"][near])) if near.any() else np.nan,
        hopf_error_far=float(np.mean(data["hopf_error"][far])) if far.any() else np.nan,
        eu_field_err_near=float(np.mean(data["eu_field_err"][near])) if near.any() else np.nan,
        eu_field_err_far=float(np.mean(data["eu_field_err"][far])) if far.any() else np.nan,
        g_field_err_near=float(np.nanmean(data["g_field_err"][near])) if near.any() else np.nan,
        g_field_err_far=float(np.nanmean(data["g_field_err"][far])) if far.any() else np.nan,
    )
    return data, summary
