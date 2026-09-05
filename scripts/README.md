# Reproducing this project's results

`run_phase4_experiments.py` reproduces every row of the manuscript's Table 3
(Point B: baseline, three capacity architectures, and the combined fix) and
Table 4 (Point C: baseline and the combined fix) from a clean checkout.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-lock.txt   # exact versions this project was run with
```

## Running

```bash
# Full run (20 seeds for Point B baseline/capacity/combined-fix, 6 seeds
# for Point C -- matches the manuscript exactly). Takes ~2.5-3 hours on a
# single laptop CPU; no GPU or HPC needed or used anywhere in this project.
python scripts/run_phase4_experiments.py --point both

# Quick smoke-check (6 seeds everywhere) that the pipeline runs end to end
# without waiting for the full statistical power -- NOT the numbers the
# manuscript reports, just a fast sanity check. Takes ~10 minutes.
python scripts/run_phase4_experiments.py --point both --quick

# Either point alone:
python scripts/run_phase4_experiments.py --point B
python scripts/run_phase4_experiments.py --point C
```

## Output

Each run writes a single timestamped JSON file to `data/generated/`,
containing:
- `provenance`: timestamp, Python version, platform, git commit (if run
  inside a git checkout), and a pointer to the lockfile used.
- Per-configuration results: every individual seed's near/far boundary
  error (not just the aggregate), plus the mean, ratio, and Wilcoxon
  p-value the manuscript reports.

This means every number in Table 3/4 traces back to a specific, committed,
re-runnable artifact — not only to prose in `PROJECT_LOG.md` (the gap an
external audit flagged, PROJECT_LOG.md Sec. 44, finding 5).

## Reproducibility notes

- All physics ground-truth tools (`double_zero.py`, `wedge_boundaries_true`
  in `src/tide/surrogates/bt_point_data.py`) use the corrected
  post-audit coordinates and boundary computation (Sec. 44) — confirmed
  there that this does not change any of the numbers this script
  reproduces, since Table 3/4 only ever use the upper boundary, which was
  unaffected by the correction.
- Seed-to-seed variance at low seed counts (e.g. the Point C `--quick`
  6-seed check) is expected and has been a recurring, explicitly-discussed
  finding throughout this project (PROJECT_LOG.md, multiple sections) --
  do not treat a single 6-seed run as final; the manuscript's Point C
  numbers use the same 6-seed count deliberately (matching what the
  original investigation used), and are reported with that seed count
  explicitly stated rather than implied to have the same power as Point
  B's 20-seed numbers.
