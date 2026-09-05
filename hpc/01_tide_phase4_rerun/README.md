# TIDE Phase-4 rerun (HPC)

Reruns every configuration in the manuscript's Table 3 (Point B: baseline,
3 capacity architectures, combined fix) and Table 4 (Point C: baseline,
combined fix), from scratch, on the cluster -- replacing
`data/generated/phase4_results_full_20260831_161035.json`, which mixed
freshly-computed configs with 3 rows hand-copied from an earlier killed
run's log (`scripts/run_remaining_configs.py`, `ALREADY_DONE_B`) and which
does not actually reproduce Table 3/4 when independently checked.

## Why 22 tasks, not 7

The 7 (point, config) pairs are each fully independent, but so is every
*seed* within them -- `run_phase4_experiments.py` just happens to loop
over seeds sequentially inside one process. With 60 free cores and only 7
independent units of work, most of that allocation would sit idle. Instead,
each Point-B config's 20 seeds are split into 4 batches of 5
(`Code_2b_run_seed_batch.py` runs an arbitrary explicit seed list, not
just `range(n_seeds)`), and Point C's 6-seed configs are left whole. That's
22 independent, 1-core tasks -- enough parallelism to matter, comfortably
under the 60-core budget, and each task still cheap enough that giving it
more than 1 core would only add thread contention, not speed.
`Code_3_Merge_Results.py` recombines the seed-batches back into exactly
the statistics an unsplit run would have produced (same mean/ratio/
Wilcoxon procedure as `run_phase4_experiments.py`'s own
`_run_single_worker`).

Neither `run_phase4_experiments.py` nor its config tables (`CONFIGS_B`,
`CONFIGS_C`, `POINT_B`, `POINT_C`) are modified -- `Code_2b` imports them
directly, so the HPC path and the original reproducibility script can
never silently drift apart.

## One-time setup

```bash
ssh <cluster>
mkdir -p /home/barathula.sreeram/Python_Stuff/tide
export TIDE_ROOT=/home/barathula.sreeram/Python_Stuff/tide
export GITHUB_TOKEN=<a fine-grained PAT scoped to read-only access on Fresh_TIDE>
cd "$TIDE_ROOT"
# copy Code_1_Setup.sh here first (scp it, or clone Fresh_TIDE once
# manually to get this hpc/ folder, then run it from inside the checkout)
bash Code_1_Setup.sh
cat "$TIDE_ROOT/results/01_tide_phase4_rerun/SETUP_THIS.txt"   # must say STATUS: PASS
```

`Code_1_Setup.sh` clones `github.com/sreerambarathula/Fresh_TIDE` (private
-- needs `GITHUB_TOKEN`), builds a venv under `$TIDE_ROOT/envs/tide_env`,
installs from `requirements-lock.txt`, confirms the post-audit tools
(`wedge_boundaries_true`, `solve_double_zero`) import cleanly, and runs the
full test suite before declaring PASS. If it reports FAIL, paste
`SETUP_THIS.txt` back and don't proceed further.

**Before submitting anything**, check `submit_array.pbs` and
`submit_merge.pbs`: both currently use `-P as_mae_jyho` (your C2PD-HPC
project code) and no explicit `-q` (matching C2PD-HPC's small/short jobs,
which used the default queue -- `qamd_wfly` was only used there for large
16-48h runs). Confirm this project code is correct for this allocation
too, and add a `-q` line if your scheduler requires one for 1-cpu/30-min
jobs.

## Running

```bash
cd "$TIDE_ROOT/repo/hpc/01_tide_phase4_rerun"
bash submit_all.sh
```

This submits the 22-task array job, then a merge job that only runs once
every array task has succeeded (`-W depend=afterokarray:<jobid>[]`) --
nothing to babysit or trigger by hand.

Monitor with `qstat -t <array_job_id>` (printed by `submit_all.sh`).

## Output

- Per-task logs/status: `$TIDE_ROOT/results/01_tide_phase4_rerun/task_<i>/`
  (`console.log`, `error.log`, `result.json`, `PASS` or `FAIL`,
  `UPLOAD_THIS.txt`).
- **Final report** (what you asked for -- the plain-text PASS/FAIL after
  everything finishes): `$TIDE_ROOT/results/01_tide_phase4_rerun/FINAL_REPORT.txt`.
  States `STATUS: PASS` only if all 7 (point, config) rows got their full,
  complete, expected seed count with no missing or failed tasks, and shows
  every row's near/far/ratio/p next to the manuscript's currently-printed
  Table 3/4 numbers for direct comparison.
- Merged result JSON, in the same schema `run_phase4_experiments.py`'s own
  `main()` produces: `<repo>/data/generated/phase4_results_full_<timestamp>.json`.

## If something needs to be redone

Every task's PASS/FAIL and logs are independent and kept per-index, so a
single failed task can be resubmitted alone
(`qsub -J <i> submit_array.pbs`) without rerunning the other 21, then
`Code_3_Merge_Results.py` rerun once all 22 show PASS.

## Bringing results back

The merged JSON lands inside the cluster-side checkout at
`$TIDE_ROOT/repo/data/generated/`. Copy it back (`scp`) into your local
`Tide_Tutor/data/generated/` and/or commit it to `Fresh_TIDE` so the
manuscript's Table 3/4 (and Figure 4, which is generated from this same
data) can be checked against it.
