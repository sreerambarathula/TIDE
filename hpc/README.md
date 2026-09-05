# TIDE: one-click full reproducibility run

Everything in `REPRODUCIBILITY.md`'s pipeline, executed fresh on the
cluster, with no manual steps between submission and the final report.

## What this runs

| # | Stream | What | Parallelism |
|---|---|---|---|
| 01 | `01_tide_phase4_rerun` | Table 3/4 (Point B/C surrogate training + stats) | 22 seed-batch tasks |
| 02 | `02_point_verification` | Table 2 (locate/verify Points A, B, C from scratch) | 1 job (fast, deterministic) |
| 03 | `03_point_a_analysis` | Section 5.1 (3 sampling schemes + 2 architectures at Point A) | 5 config tasks |
| 04 | `04_mechanism_test` | Section 5.3 (gradient vs. window-width mechanism, Point B) | 20 per-seed tasks |

Total concurrent cores if the scheduler runs everything at once: 22+2+5+20 = 49,
comfortably under a 60-core budget. Every task is pinned to 1 core (except
02's single job at 2) to avoid thread oversubscription across ~49
concurrently-running single-threaded JAX processes -- matching the thread
hygiene used throughout this pipeline (`OMP_NUM_THREADS`/`OPENBLAS`/`MKL`/
`NUMEXPR` all pinned, `JAX_PLATFORM_NAME=cpu`).

None of the four streams' own scripts (`scripts/run_phase4_experiments.py`,
`scripts/find_and_verify_points.py`, `scripts/run_point_a_experiments.py`,
`scripts/run_mechanism_test.py`) are modified by anything in `hpc/` -- every
`Code_2_*` runner just calls that script's existing `--_worker` mode (or,
for 02, the plain top-level script) in a fresh subprocess per task, and
every `Code_3_Merge_Results.py` recombines the pieces into exactly the
statistics an unsplit run would produce.

## One-time setup (run once)

```bash
ssh <cluster>
mkdir -p /home/barathula.sreeram/Python_Stuff/Fresh_TIDE
cd /home/barathula.sreeram/Python_Stuff/Fresh_TIDE
git clone git@github.com:sreerambarathula/Fresh_TIDE.git repo
cd repo/hpc/01_tide_phase4_rerun
bash Code_1_Setup.sh
cat /home/barathula.sreeram/Python_Stuff/Fresh_TIDE/results/01_tide_phase4_rerun/SETUP_THIS.txt
```

Must say `STATUS: PASS` (clones the repo, builds the shared venv at
`$TIDE_ROOT/envs/tide_env`, confirms the post-audit tools import cleanly,
and runs the full test suite). This setup is shared by all 4 streams --
only needs to happen once.

**Alternative: run setup as a batch job instead of interactively.** The
dependency-install step is silent for several minutes (installing JAX/
SciPy/etc.), which can look hung on an interactive terminal; running it as
a PBS job instead survives disconnects and doesn't risk an impatient
`Ctrl-C`. A batch job has no terminal to type your SSH key's passphrase
into, so this needs HTTPS + a fine-grained PAT instead of SSH:

```bash
export GIT_REPO_URL="https://<token>@github.com/sreerambarathula/Fresh_TIDE.git"
cd /home/barathula.sreeram/Python_Stuff/Fresh_TIDE/repo/hpc/01_tide_phase4_rerun
qsub -v GIT_REPO_URL submit_setup.pbs
# then, once it finishes:
cat /home/barathula.sreeram/Python_Stuff/Fresh_TIDE/results/01_tide_phase4_rerun/SETUP_THIS.txt
```
`-v GIT_REPO_URL` passes that one variable from your shell into the job --
the token is never written into any file or committed anywhere.

**Before submitting anything**, check the `#PBS -P` project code (currently
`as_mae_jyho`, your C2PD-HPC code) in every `submit*.pbs` file under `hpc/`
is correct for this allocation, and add a `-q <queue>` line if your
scheduler requires one for jobs this size (none of these need more than a
few GB of memory or more than an hour of walltime per task).

## The one click

```bash
cd /home/barathula.sreeram/Python_Stuff/Fresh_TIDE/repo/hpc
bash submit_everything.sh
```

This submits all 4 streams' array/single jobs, each stream's own merge job
(dependent on that stream's array finishing), and one final master-report
job dependent on **all four** streams succeeding
(`-W depend=afterok:<merge1>:<job2>:<merge3>:<merge4>`). Nothing else to
run by hand.

## Watching it

```bash
bash Code_Watch_All.sh              # one-shot snapshot of all 4 streams
watch -n 30 bash Code_Watch_All.sh  # auto-refreshing dashboard
qstat -t <any_array_job_id>         # raw PBS state for one stream
```

Each stream also has its own per-task logs at
`$TIDE_ROOT/results/<stream>/task_<i>/console.log` (and `error.log`,
`UPLOAD_THIS.txt`, `PASS`/`FAIL`) if something needs closer inspection.

## What lands where, when it's all done

- **Per-stream reports**: `$TIDE_ROOT/results/<stream>/FINAL_REPORT.txt`
  (or, for 02, `UPLOAD_THIS.txt`) -- each stream's own PASS/FAIL and
  numbers next to the manuscript's currently-printed values.
- **Per-stream CSVs**: `<repo>/data/generated/*_per_seed_*.csv` (Point A,
  mechanism test) and the per-seed arrays inside each stream's merged JSON
  (Table 3/4, point verification).
- **One master report**: `$TIDE_ROOT/results/MASTER_REPORT.txt` -- every
  quantitative result across all 4 streams, next to the manuscript's value,
  with a per-stream and one overall PASS/FAIL.
- **One master CSV**: `$TIDE_ROOT/results/MASTER_SUMMARY.csv` -- the same
  comparisons as rows, for spreadsheet use.
- **Merged JSONs** (full per-seed data, not just summaries):
  `<repo>/data/generated/phase4_results_full_*.json`,
  `point_verification_*.json`, `point_a_results_full_*.json`,
  `mechanism_test_full_*.json`.

## Bringing results back

Everything under `<repo>/data/generated/` and `$TIDE_ROOT/results/` is what
you need to `scp` back (or commit `data/generated/*.json` and `*.csv` to
`Fresh_TIDE`) for comparing against what's currently in the manuscript,
updating figures/tables, and only then drafting.

## If something fails

Each task's `PASS`/`FAIL` and logs are independent. A single failed array
task can be resubmitted alone (`qsub -J <i> submit_array.pbs` inside that
stream's directory) without rerunning the rest of that stream, then that
stream's own merge job rerun, then (if it was the last one needed)
`submit_final_report.pbs` rerun by hand.
