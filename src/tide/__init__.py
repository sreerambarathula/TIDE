"""TIDE: Two-phase Instability boundary via Decoupled-Error estimation.

Enables float64 globally at import time. Newton continuation for fold/Hopf
conditions (Phase 2) needs double precision to converge to machine-precision
residuals; JAX defaults to float32, which would silently cap continuation
accuracy well before it becomes the bottleneck.
"""

import jax

jax.config.update("jax_enable_x64", True)
