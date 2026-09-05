"""Log-distance-from-BT-point feature: Phase 4's third fix candidate.

Motivation, derived from DIRECT empirical measurement, not assumed theory:
the naive expectation (Kuznetsov normal-form tangency) was that the stable
window's width should vanish QUADRATICALLY approaching the BT point. Direct
measurement against this project's own validated ground truth
(bt_point_data.wedge_boundaries, Point B, delta in [2e-3, 2.0]) showed
instead a clean, well-fit (R^2=0.994) LINEAR scaling, width ~ 0.96*delta,
spanning roughly three orders of magnitude in delta. That ruled out a
sqrt-coordinate fix (which would have been built on the wrong exponent) and
pointed at the real problem: not a pathological functional form, but a
multi-decade DYNAMIC RANGE in an otherwise perfectly ordinary linear
feature -- exactly the situation the standard log-transform remedy is for.

Unlike a generic implicit-neural-representation problem (where the
location of fine-scale structure is unknown a priori, motivating
Fourier features / multi-resolution hash grids as location-agnostic
fixes -- both tried already, Phase 4: Fourier features tested and FAILED,
see bt_decoupling results), this problem hands us the singularity's exact
location for free: Nsub_BT is already known to 5 decimals
(bt_point_data.NSUB_BT etc.). This module exploits that directly: augment
the raw (Nsub, Npch) input with an explicit third feature,
log(|Nsub_BT - Nsub| + eps), so the network sees the multi-decade approach
to the BT point on a scale where an ordinary smooth MLP has no trouble
resolving it -- no change to activation functions, capacity, or the
target; the ONLY difference from the plain mlp.py baseline is this one
extra input feature, an isolated test of the log-distance hypothesis."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tide.surrogates.mlp import _init_params, _standardize, _unstandardize


def _augment(nsub, npch, nsub_bt, eps):
    log_dist = jnp.log(jnp.abs(nsub_bt - nsub) + eps)
    return jnp.stack([nsub, npch, log_dist], axis=-1)


def _forward(params, x_norm):
    h = x_norm
    for w, b in params[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params[-1]
    return h @ w + b


@jax.jit
def _predict_normalized(params, x_norm):
    return _forward(params, x_norm)


@jax.jit
def _predict_scalar(params, nsub_bt, eps, x_mean, x_std, y_mean, y_std, nsub, npch):
    x = _augment(nsub, npch, nsub_bt, eps)
    x_norm = _standardize(x, x_mean, x_std)
    y_norm = _forward(params, x_norm)
    y = _unstandardize(y_norm, y_mean, y_std)
    return y[0], y[1]


class TrainedLogDistanceSurrogate:
    """Same public interface as mlp.TrainedSurrogate (.predict / .predict_scalar)
    so it drops into eval/bt_decoupling.py and eval/stats.py unchanged."""

    def __init__(self, params, nsub_bt, eps, x_mean, x_std, y_mean, y_std):
        self.params = params
        self.nsub_bt = nsub_bt
        self.eps = eps
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

    def predict_scalar(self, nsub, npch):
        return _predict_scalar(self.params, self.nsub_bt, self.eps,
                                self.x_mean, self.x_std, self.y_mean, self.y_std,
                                nsub, npch)

    def predict(self, nsub, npch):
        nsub = jnp.atleast_1d(jnp.asarray(nsub, dtype=jnp.float64))
        npch = jnp.atleast_1d(jnp.asarray(npch, dtype=jnp.float64))
        x = _augment(nsub, npch, self.nsub_bt, self.eps)
        x_norm = _standardize(x, self.x_mean, self.x_std)
        y_norm = _predict_normalized(self.params, x_norm)
        y = _unstandardize(y_norm, self.y_mean, self.y_std)
        return y[..., 0], y[..., 1]


def train_log_distance_surrogate(train_data, nsub_bt, hidden_sizes=(64, 64),
                                  eps=1e-6, n_epochs=10000, lr=1e-3, seed=0):
    """Train the log-distance-augmented MLP surrogate on
    (Nsub, Npch, log|Nsub_BT-Nsub|) -> (Eu, g)."""
    nsub_arr, npch_arr = train_data["nsub"], train_data["npch"]
    x = np.stack([nsub_arr, npch_arr, np.log(np.abs(nsub_bt - nsub_arr) + eps)], axis=-1)
    y = np.stack([train_data["Eu"], train_data["g"]], axis=-1)

    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)

    x_norm = jnp.asarray(_standardize(x, x_mean, x_std))
    y_norm = jnp.asarray(_standardize(y, y_mean, y_std))

    rng = jax.random.PRNGKey(seed)
    layer_sizes = (3,) + tuple(hidden_sizes) + (2,)
    params = _init_params(rng, layer_sizes)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, x_norm, y_norm):
        pred = _forward(params, x_norm)
        return jnp.mean((pred - y_norm) ** 2)

    @jax.jit
    def step(params, opt_state, x_norm, y_norm):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_norm, y_norm)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for _ in range(n_epochs):
        params, opt_state, loss = step(params, opt_state, x_norm, y_norm)
        losses.append(float(loss))

    surrogate = TrainedLogDistanceSurrogate(
        params, float(nsub_bt), float(eps),
        jnp.asarray(x_mean), jnp.asarray(x_std),
        jnp.asarray(y_mean), jnp.asarray(y_std),
    )
    return surrogate, dict(train_loss=np.array(losses))
