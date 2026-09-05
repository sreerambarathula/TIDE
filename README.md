# TIDE

Two-phase Instability boundary via Decoupled-Error estimation.

Full project history, decisions, and reasoning: [`PROJECT_LOG.md`](PROJECT_LOG.md).
That file is the source of truth — read it before making changes to scope or method.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Pure Python/JAX, CPU-only, no HPC required — the full data-generation pipeline runs
in under an hour on a laptop.

## Layout

```
src/tide/physics/       lumped moving-boundary channel ODE model (JAX)
src/tide/continuation/  Newton + pseudo-arclength continuation for fold/Hopf ground truth
src/tide/surrogates/    learned surrogate models and training objectives
src/tide/eval/          boundary-distance metrics, decoupling diagnostics
tests/
data/generated/         generated datasets (gitignored)
```
