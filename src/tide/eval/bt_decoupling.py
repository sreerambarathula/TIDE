"""The positive-counterpart decoupling measurement (PROJECT_LOG.md Sec. 38):
does boundary error spike near the GENUINE Bogdanov-Takens point found in
Sec. 37, the way it failed to near the transversal point (Sec. 31-34)?

Same pre-registration discipline as Sec. 31: primary test (boundary error
near vs. far, field error near vs. far, seed-level statistics per Sec. 32's
pseudo-replication fix), decided before running.
"""

import numpy as np

from tide.eval.decoupling import _extract_surrogate_fold
from tide.physics.ledinegg_curve import euler_number
from tide.surrogates.bt_point_data import _g, wedge_boundaries


def true_gradient_norm_at(nsub, npch, Fr, Lam, ki, ke, N1, h=1e-4):
    """||grad g|| at a true point via central finite differences (Sec. 39:
    checked directly, not autodiff, same reasoning as decoupling.py's
    version -- diagnostic quantity only, not a training signal)."""
    dg_dnsub = (_g(nsub + h, npch, Fr, Lam, ki, ke, N1) - _g(nsub - h, npch, Fr, Lam, ki, ke, N1)) / (2 * h)
    dg_dnpch = (_g(nsub, npch + h, Fr, Lam, ki, ke, N1) - _g(nsub, npch - h, Fr, Lam, ki, ke, N1)) / (2 * h)
    return float(np.hypot(dg_dnsub, dg_dnpch))


def _extract_surrogate_upper(surrogate, nsub, true_upper, npch_points=2000):
    """Upper (Hopf-type) boundary of the surrogate's own predicted g, using a
    bracket centered on the TRUE upper value (progressively widened if no
    sign change is found), not a wide fixed (fold, 1.6*fold) sweep.

    Found necessary directly, not assumed (Sec. 40): a wide, uninformed
    bracket occasionally picks up a SPURIOUS extra root the surrogate's own
    imperfect fit introduces (a wiggle away from the true boundary) --
    confirmed by tracking the same seed across a fixed Nsub grid and seeing
    a large, ~2-unit error appear at a DIFFERENT Nsub for each seed (a
    random, seed-dependent location, not a physical feature -- the ground
    truth g(Npch) curve at those Nsub values is a single, clean,
    unambiguous crossing, confirmed by direct inspection). Centering the
    search on the already-known true value avoids ever seeing a spurious
    root that a wide, blind sweep would find first."""
    for half_width_frac in (0.15, 0.3, 0.5, 0.8, 1.2):
        lo = max(true_upper * (1 - half_width_frac), nsub * 1.001)
        hi = true_upper * (1 + half_width_frac)
        npch_grid = np.linspace(lo, hi, npch_points)
        nsub_full = np.full_like(npch_grid, nsub)
        _, g_pred = surrogate.predict(nsub_full, npch_grid)
        g_pred = np.array(g_pred)
        # pick the sign change CLOSEST to true_upper, not the first one --
        # robust even if a spurious root exists elsewhere in a wider bracket.
        crossings = []
        for i in range(len(npch_grid) - 1):
            if g_pred[i] * g_pred[i + 1] < 0:
                crossings.append((npch_grid[i], npch_grid[i + 1]))
        if not crossings:
            continue
        blo, bhi = min(crossings, key=lambda c: abs(0.5 * (c[0] + c[1]) - true_upper))

        def g_scalar(npch):
            return float(surrogate.predict_scalar(nsub, npch)[1])

        flo, fhi = g_scalar(blo), g_scalar(bhi)
        for _ in range(60):
            if bhi - blo < 1e-9:
                break
            mid = 0.5 * (blo + bhi)
            fmid = g_scalar(mid)
            if flo * fmid < 0:
                bhi, fhi = mid, fmid
            else:
                blo, flo = mid, fmid
        return 0.5 * (blo + bhi)
    return None


def run_bt_decoupling_analysis(surrogate, Fr, Lam, ki, ke, N1, nsub_bt, nsub_grid,
                                npch_grid_points=2000, npch_mult=2.5):
    """Runs the wedge-boundary decoupling analysis over an independent Nsub
    grid (all below nsub_bt -- there is no window above it, Sec. 37)."""
    rows = []
    for nsub in nsub_grid:
        true_lower, true_upper = wedge_boundaries(nsub, Fr, Lam, ki, ke, N1)
        if true_lower is None or true_upper is None:
            continue

        npch_grid = np.linspace(nsub * 1.02, nsub * npch_mult, npch_grid_points)
        surr_lower = _extract_surrogate_fold(surrogate, nsub, npch_grid)
        surr_upper = _extract_surrogate_upper(surrogate, nsub, true_upper)
        if surr_lower is None or surr_upper is None:
            continue

        # Local field error over the window neighborhood, not at the exact
        # boundary point (Sec. 31's fix, same reasoning applies here).
        npch_local = np.linspace(true_lower * 0.9, true_upper * 1.1, 25)
        eu_true_local = np.array([euler_number(p, nsub, Fr, Lam, ki, ke) for p in npch_local])
        g_true_local = np.array([_g(nsub, p, Fr, Lam, ki, ke, N1) for p in npch_local])
        nsub_local = np.full_like(npch_local, nsub)
        eu_pred_local, g_pred_local = surrogate.predict(nsub_local, npch_local)
        eu_pred_local, g_pred_local = np.asarray(eu_pred_local), np.asarray(g_pred_local)
        eu_field_err = float(np.sqrt(np.mean((eu_pred_local - eu_true_local) ** 2)))
        finite_g = np.array([v is not None for v in g_true_local])
        if finite_g.any():
            g_true_arr = np.array([v if v is not None else np.nan for v in g_true_local])
            g_field_err = float(np.sqrt(np.nanmean((g_pred_local - g_true_arr) ** 2)))
        else:
            g_field_err = np.nan

        rows.append(dict(
            nsub=nsub, true_lower=true_lower, true_upper=true_upper,
            surr_lower=surr_lower, surr_upper=surr_upper,
            lower_error=abs(surr_lower - true_lower), upper_error=abs(surr_upper - true_upper),
            eu_field_err=eu_field_err, g_field_err=g_field_err,
            dist_from_bt=abs(nsub_bt - nsub),
        ))

    if not rows:
        raise ValueError("no valid points -- check nsub_grid is below nsub_bt")

    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}

    near = data["dist_from_bt"] <= 0.5
    far = data["dist_from_bt"] >= 2.0

    summary = dict(
        n_near=int(near.sum()), n_far=int(far.sum()),
        lower_error_near=float(np.mean(data["lower_error"][near])) if near.any() else np.nan,
        lower_error_far=float(np.mean(data["lower_error"][far])) if far.any() else np.nan,
        upper_error_near=float(np.mean(data["upper_error"][near])) if near.any() else np.nan,
        upper_error_far=float(np.mean(data["upper_error"][far])) if far.any() else np.nan,
        eu_field_err_near=float(np.mean(data["eu_field_err"][near])) if near.any() else np.nan,
        eu_field_err_far=float(np.mean(data["eu_field_err"][far])) if far.any() else np.nan,
        g_field_err_near=float(np.nanmean(data["g_field_err"][near])) if near.any() else np.nan,
        g_field_err_far=float(np.nanmean(data["g_field_err"][far])) if far.any() else np.nan,
    )
    return data, summary
