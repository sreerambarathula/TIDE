"""Section 5.3's mechanism test: does Point B's boundary-error elevation
correlate with the classical vanishing-gradient hypothesis (1/||grad g||)
or with the representability hypothesis (1/window-width)? Per-seed Spearman
correlation, not pooled (same pseudo-replication discipline as everywhere
else in this project).

Every function called here is pre-existing, already-real code:
tide.surrogates.bt_point_data.generate_bt_dataset / wedge_boundaries_true,
tide.eval.bt_decoupling.run_bt_decoupling_analysis / true_gradient_norm_at,
tide.surrogates.mlp.train_surrogate / train_test_split. This script only
orchestrates them and adds the two correlation summaries -- no new physics.

Usage:
    python scripts/run_mechanism_test.py [--n-seeds 20]
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "generated"
THIS_FILE = Path(__file__).resolve()

POINT_B = dict(Fr=0.5, Lam=0.001, ki=11.0, ke=3.0, Nsub_bt=14.142794816,
               nsub_range=(11.0, 14.10))


def _provenance():
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=THIS_FILE.parents[1],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "not-a-git-repo"
    return dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                python_version=platform.python_version(),
                platform=platform.platform(), git_commit=git_hash)


def _run_single_worker(seed):
    """One seed's mechanism data: trains a baseline surrogate, runs the
    decoupling analysis, and computes both candidate mechanism quantities
    (gradient norm, window width) at every surviving Nsub point -- using
    the CORRECTED wedge_boundaries_true throughout (Supplementary S4.5),
    not the superseded wedge_boundaries."""
    from tide.eval.bt_decoupling import run_bt_decoupling_analysis, true_gradient_norm_at
    from tide.surrogates.bt_point_data import generate_bt_dataset, wedge_boundaries_true
    from tide.surrogates.mlp import train_surrogate, train_test_split

    p = POINT_B
    data = generate_bt_dataset(p["Fr"], p["Lam"], p["ki"], p["ke"], N1=16,
                                nsub_range=p["nsub_range"], n_nsub=60)
    train, _ = train_test_split(data, seed=seed)
    surrogate, _ = train_surrogate(train, hidden_sizes=(64, 64), n_epochs=10000, seed=seed)

    nsub_grid = np.linspace(p["nsub_range"][0], p["nsub_range"][1], 60)
    per_point, _ = run_bt_decoupling_analysis(
        surrogate, p["Fr"], p["Lam"], p["ki"], p["ke"], N1=16,
        nsub_bt=p["Nsub_bt"], nsub_grid=nsub_grid)

    grad_norms, window_widths = [], []
    for i, nsub in enumerate(per_point["nsub"]):
        true_upper = per_point["true_upper"][i]
        true_lower = per_point["true_lower"][i]
        grad_norms.append(true_gradient_norm_at(nsub, true_upper, p["Fr"], p["Lam"],
                                                  p["ki"], p["ke"], N1=16))
        # Corrected window width -- true_upper/true_lower already come from
        # wedge_boundaries_true inside run_bt_decoupling_analysis, so this
        # is the corrected width, not the pre-audit ~58%-overstated one.
        window_widths.append(true_upper - true_lower)

    result = dict(
        seed=seed,
        nsub=[float(v) for v in per_point["nsub"]],
        upper_error=[float(v) for v in per_point["upper_error"]],
        grad_norm=[float(v) for v in grad_norms],
        window_width=[float(v) for v in window_widths],
    )
    print("RESULT_JSON:" + json.dumps(result))


def _run_seed_subprocess(seed):
    proc = subprocess.run(
        [sys.executable, str(THIS_FILE), "--_worker", "--seed", str(seed)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"  seed {seed}: SUBPROCESS FAILED (exit {proc.returncode})", flush=True)
        print(proc.stderr[-4000:], flush=True)
        raise RuntimeError(f"worker subprocess failed for seed {seed}")
    result_line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:"))
    return json.loads(result_line[len("RESULT_JSON:"):])


def _summarize(rhos, ps):
    rhos, ps = np.array(rhos), np.array(ps)
    return dict(
        n_seeds=len(rhos), median_rho=float(np.median(rhos)) if len(rhos) else None,
        frac_positive_significant=float(np.mean((rhos > 0) & (ps < 0.05))) if len(rhos) else None,
        frac_negative_significant=float(np.mean((rhos < 0) & (ps < 0.05))) if len(rhos) else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        _run_single_worker(args.seed)
        return

    t0 = time.time()
    per_seed_results = []
    grad_rhos, grad_ps = [], []
    width_rhos, width_ps = [], []
    for seed in range(args.n_seeds):
        r = _run_seed_subprocess(seed)
        per_seed_results.append(r)
        upper_error = np.array(r["upper_error"])
        grad_norm = np.array(r["grad_norm"])
        window_width = np.array(r["window_width"])
        finite = np.isfinite(upper_error) & np.isfinite(grad_norm) & (grad_norm > 0)
        if finite.sum() > 5:
            rho, p = sstats.spearmanr(upper_error[finite], 1.0 / grad_norm[finite])
            grad_rhos.append(rho)
            grad_ps.append(p)
        finite2 = np.isfinite(upper_error) & np.isfinite(window_width) & (window_width > 0)
        if finite2.sum() > 5:
            rho2, p2 = sstats.spearmanr(upper_error[finite2], 1.0 / window_width[finite2])
            width_rhos.append(rho2)
            width_ps.append(p2)
        print(f"  seed {seed}: grad_rho={grad_rhos[-1]:.3f} width_rho={width_rhos[-1]:.3f}",
              flush=True)

    output = dict(
        provenance=_provenance(),
        n_seeds=args.n_seeds,
        gradient_mechanism=_summarize(grad_rhos, grad_ps),
        window_width_mechanism=_summarize(width_rhos, width_ps),
        per_seed=dict(grad_rhos=grad_rhos, grad_ps=grad_ps,
                       width_rhos=width_rhos, width_ps=width_ps),
        elapsed_seconds=round(time.time() - t0, 1),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"mechanism_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nGradient mechanism: median rho={output['gradient_mechanism']['median_rho']:.3f} "
          f"(manuscript reports -0.675, wrong-signed relative to classical hypothesis)")
    print(f"Window-width mechanism: median rho={output['window_width_mechanism']['median_rho']:.3f} "
          f"(manuscript reports +0.668, correctly-signed for representability limit)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
