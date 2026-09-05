"""Boundary-targeted loss: Phase 4's fourth fix candidate, and the one this
project's own mlp.py docstring names as "the contribution being argued for"
-- unlike every previous attempt (capacity scaling, Fourier features,
log-distance features), which all kept the SAME uniform field-error MSE
objective and only changed architecture/input representation, this changes
the TRAINING OBJECTIVE ITSELF: up-weight the g-loss for training points
whose TRUE g is close to zero (i.e., close to an actual fold/Hopf
crossing), so the optimizer is directly told that getting g right near its
own zero-crossing matters far more than average accuracy elsewhere in the
domain -- a direct attack on the mechanism (Sec. 39: this was diagnosed as
a representability/resolution problem, and uniform field error treats a
razor-thin, disproportionately important region exactly the same as
everywhere else, giving the optimizer no reason to spend capacity there).

weight_i = 1 / (|g_true_i| + eps_w), normalized to a per-dataset mean of 1
so the overall loss scale matches the unweighted baseline (fair,
comparable convergence behavior at the same learning rate/epoch count).
eps_w controls how sharply weighting concentrates near the crossing;
chosen (like Fourier sigma and the log-distance eps before it) by
held-out FIELD ERROR, never by the decoupling metric itself -- same
pre-registration discipline used throughout this project.

Architecture, optimizer, and prediction interface are otherwise IDENTICAL
to mlp.py's baseline -- this reuses mlp.TrainedSurrogate directly (not a
new class), so it is an isolated test of the loss-reweighting hypothesis
and nothing else."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tide.surrogates.mlp import TrainedSurrogate, _forward, _init_params, _standardize


def train_surrogate_boundary_weighted(train_data, hidden_sizes=(64, 64), eps_w=0.05,
                                       n_epochs=10000, lr=1e-3, seed=0):
    """Train the boundary-weighted MLP surrogate on (Nsub, Npch) -> (Eu, g).

    Only the g-loss is reweighted (Eu carries no boundary-location
    information -- the crossing is defined purely by g=0)."""
    x = np.stack([train_data["nsub"], train_data["npch"]], axis=-1)
    y = np.stack([train_data["Eu"], train_data["g"]], axis=-1)

    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)

    x_norm = jnp.asarray(_standardize(x, x_mean, x_std))
    y_norm = jnp.asarray(_standardize(y, y_mean, y_std))

    g_true = jnp.asarray(train_data["g"])
    weight = 1.0 / (jnp.abs(g_true) + eps_w)
    weight = weight / jnp.mean(weight)

    rng = jax.random.PRNGKey(seed)
    layer_sizes = (2,) + tuple(hidden_sizes) + (2,)
    params = _init_params(rng, layer_sizes)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, x_norm, y_norm, weight):
        pred = _forward(params, x_norm)
        eu_loss = jnp.mean((pred[:, 0] - y_norm[:, 0]) ** 2)
        g_loss = jnp.mean(weight * (pred[:, 1] - y_norm[:, 1]) ** 2)
        return eu_loss + g_loss

    @jax.jit
    def step(params, opt_state, x_norm, y_norm, weight):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_norm, y_norm, weight)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for _ in range(n_epochs):
        params, opt_state, loss = step(params, opt_state, x_norm, y_norm, weight)
        losses.append(float(loss))

    surrogate = TrainedSurrogate(
        params,
        jnp.asarray(x_mean), jnp.asarray(x_std),
        jnp.asarray(y_mean), jnp.asarray(y_std),
    )
    return surrogate, dict(train_loss=np.array(losses))
