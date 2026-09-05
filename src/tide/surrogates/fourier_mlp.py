"""Fourier-feature encoding: Phase 4's second fix candidate, targeted at the
diagnosed mechanism rather than generic capacity scaling (PROJECT_LOG.md
Phase 4: plain capacity scaling gave a partial, non-monotonic 20-seed
result -- ratio dropped from 3.54x at (64,64) to ~1.2-1.8x at larger
architectures, but not cleanly or monotonically with size).

Mechanism recap (Sec. 39): BT-point decoupling was diagnosed as a
REPRESENTABILITY limit, not a gradient/conditioning problem -- the stable
Npch window narrows toward zero width approaching the BT point, and a
plain tanh-MLP has a well-documented spectral bias toward low-frequency
functions (Rahaman et al. 2019, "On the Spectral Bias of Neural
Networks"), meaning narrow/localized features are exactly what it
struggles to fit no matter how many parameters it has. Random Fourier
features (Tancik et al. 2020, "Fourier Features Let Networks Learn
High-Frequency Functions in Low-Dimensional Domains") is the standard
targeted remedy: encode the (already-standardized) input through a fixed,
UNTRAINED random sinusoidal basis before the MLP, which removes the
spectral bias and lets ordinary gradient descent fit high-frequency /
narrow features that a raw-coordinate MLP cannot.

gamma(x) = [sin(2*pi*B@x), cos(2*pi*B@x)],  B ~ N(0, sigma^2), sampled once
at init from `seed` and held fixed thereafter (never trained) -- the
standard recipe. sigma controls which frequency band is representable and
is a genuine hyperparameter with no principled default here; it must be
selected by held-out FIELD ERROR (Eu/g RMSE on a test split), never by the
decoupling metric itself -- tuning sigma against the near-vs-far statistic
the paper reports would be data snooping on the primary result, the same
pre-registration discipline bt_decoupling.py already follows for the
boundary-extraction procedure itself.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tide.surrogates.mlp import _init_params, _standardize, _unstandardize


def _fourier_features(x_norm, B):
    proj = 2 * jnp.pi * (x_norm @ B)
    return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)


def _forward(params, x_norm, B):
    h = _fourier_features(x_norm, B)
    for w, b in params[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params[-1]
    return h @ w + b


@jax.jit
def _predict_normalized(params, x_norm, B):
    return _forward(params, x_norm, B)


@jax.jit
def _predict_scalar(params, B, x_mean, x_std, y_mean, y_std, nsub, npch):
    """Differentiable (nsub, npch) -> (Eu, g) for a single point -- same role
    as mlp._predict_scalar, needed so eval/decoupling.py's jax.grad-based
    Newton/bisection extraction works unchanged on this surrogate too."""
    x = jnp.array([nsub, npch])
    x_norm = _standardize(x, x_mean, x_std)
    y_norm = _forward(params, x_norm, B)
    y = _unstandardize(y_norm, y_mean, y_std)
    return y[0], y[1]


class TrainedFourierSurrogate:
    """Same public interface as mlp.TrainedSurrogate (.predict / .predict_scalar)
    so it drops into eval/bt_decoupling.py and eval/stats.py completely
    unchanged -- those modules only ever call the surrogate through this
    duck-typed interface, never touch its internals."""

    def __init__(self, params, B, x_mean, x_std, y_mean, y_std):
        self.params = params
        self.B = B
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

    def predict_scalar(self, nsub, npch):
        return _predict_scalar(self.params, self.B, self.x_mean, self.x_std,
                                self.y_mean, self.y_std, nsub, npch)

    def predict(self, nsub, npch):
        nsub = jnp.atleast_1d(jnp.asarray(nsub, dtype=jnp.float64))
        npch = jnp.atleast_1d(jnp.asarray(npch, dtype=jnp.float64))
        x = jnp.stack([nsub, npch], axis=-1)
        x_norm = _standardize(x, self.x_mean, self.x_std)
        y_norm = _predict_normalized(self.params, x_norm, self.B)
        y = _unstandardize(y_norm, self.y_mean, self.y_std)
        return y[..., 0], y[..., 1]


def train_fourier_surrogate(train_data, hidden_sizes=(64, 64), n_fourier_features=64,
                             sigmas=10.0, n_epochs=10000, lr=1e-3, seed=0):
    """Train the Fourier-feature MLP surrogate on (Nsub, Npch) -> (Eu, g).

    `sigmas` may be a single float (one frequency band, `n_fourier_features`
    total) or a sequence of floats (MULTIPLE bands, `n_fourier_features`
    PER band, concatenated) -- the standard multi-scale recipe (Tancik et
    al. 2020 use several octaves, not one fixed scale) meant to let the
    network use low frequencies for global smoothness and high frequencies
    only where the data actually needs them, rather than forcing one
    global frequency/overfitting trade-off on the whole domain.

    B (shape (2, n_fourier_features * n_bands)) is sampled once from
    N(0, sigma_i^2) per band using `seed` and held fixed for the whole
    training run -- it is NOT an optimized parameter, per the standard
    random-Fourier-features recipe."""
    x = np.stack([train_data["nsub"], train_data["npch"]], axis=-1)
    y = np.stack([train_data["Eu"], train_data["g"]], axis=-1)

    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)

    x_norm = jnp.asarray(_standardize(x, x_mean, x_std))
    y_norm = jnp.asarray(_standardize(y, y_mean, y_std))

    sigma_list = [sigmas] if np.isscalar(sigmas) else list(sigmas)
    rng = jax.random.PRNGKey(seed)
    b_keys = jax.random.split(rng, len(sigma_list) + 1)
    param_key = b_keys[-1]
    B = jnp.concatenate(
        [jax.random.normal(b_keys[i], (2, n_fourier_features)) * s
         for i, s in enumerate(sigma_list)],
        axis=1,
    )
    n_total_features = n_fourier_features * len(sigma_list)

    layer_sizes = (2 * n_total_features,) + tuple(hidden_sizes) + (2,)
    params = _init_params(param_key, layer_sizes)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, x_norm, y_norm, B):
        pred = _forward(params, x_norm, B)
        return jnp.mean((pred - y_norm) ** 2)

    @jax.jit
    def step(params, opt_state, x_norm, y_norm, B):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_norm, y_norm, B)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for _ in range(n_epochs):
        params, opt_state, loss = step(params, opt_state, x_norm, y_norm, B)
        losses.append(float(loss))

    surrogate = TrainedFourierSurrogate(
        params, B,
        jnp.asarray(x_mean), jnp.asarray(x_std),
        jnp.asarray(y_mean), jnp.asarray(y_std),
    )
    return surrogate, dict(train_loss=np.array(losses))
