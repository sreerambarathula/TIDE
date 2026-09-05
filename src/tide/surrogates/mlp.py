"""The "standard surrogate" Phase 3 needs a baseline for (PROJECT_LOG.md Sec.
29-30): a conventional feedforward network trained to minimize plain pointwise
(field) error on (Eu, g) as functions of (Nsub, Npch), with no awareness of
where the fold/Hopf boundaries are. This is deliberately the un-clever
baseline the paper's thesis is about -- Phase 4's boundary-targeted loss is
the contribution being argued for; this module must NOT anticipate it.

Architecture and training choices here are all standard, off-the-shelf
regression practice -- nothing here is boundary-aware:
  - 2 inputs (Nsub, Npch) -> 2 outputs (Eu, g), two hidden layers of 64 units,
    tanh activations (smooth activation for a smooth target field; the
    targets are continuous and differentiable away from the boundary).
  - Inputs and outputs standardized (zero mean, unit variance) using TRAIN-SET
    statistics only -- Eu (range ~36-64) and g (range ~-1.5 to 3) differ by
    ~20x in scale, so without this an unweighted MSE loss would be dominated
    by Eu and barely fit g at all. This is ordinary data hygiene, not a
    boundary-aware trick.
  - Adam optimizer (optax), full-batch gradient descent -- the training set
    (a few thousand points) is small enough that full-batch is both correct
    and fast; no need for mini-batching machinery.
  - A held-out test split (fixed seed, 80/20) so the field-error numbers
    reported are genuine generalization error, not training-set fit quality.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax


def _init_params(rng, layer_sizes):
    params = []
    keys = jax.random.split(rng, len(layer_sizes) - 1)
    for key, n_in, n_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        w_key, b_key = jax.random.split(key)
        scale = jnp.sqrt(2.0 / n_in)
        w = jax.random.normal(w_key, (n_in, n_out)) * scale
        b = jnp.zeros((n_out,))
        params.append((w, b))
    return params


def _forward(params, x):
    h = x
    for w, b in params[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params[-1]
    return h @ w + b


@jax.jit
def _predict_normalized(params, x_norm):
    return _forward(params, x_norm)


def _standardize(x, mean, std):
    return (x - mean) / std


def _unstandardize(x_norm, mean, std):
    return x_norm * std + mean


@jax.jit
def _predict_scalar(params, x_mean, x_std, y_mean, y_std, nsub, npch):
    """Differentiable (nsub, npch) -> (Eu, g) for a single point, pure JAX
    scalars in and out -- needed so eval/decoupling.py can Newton/bisect on
    the surrogate's own predicted Eu and g via jax.grad, for boundary
    extraction precision matching the ground-truth tools' own standard
    (curves.fold_npch, codim2_convergence.hopf_npch_general), rather than the
    coarser grid+interpolation used for a first pass (PROJECT_LOG.md Sec. 32)."""
    x = jnp.array([nsub, npch])
    x_norm = _standardize(x, x_mean, x_std)
    y_norm = _forward(params, x_norm)
    y = _unstandardize(y_norm, y_mean, y_std)
    return y[0], y[1]


class TrainedSurrogate:
    """A trained MLP plus the normalization stats needed to use it -- bundled
    together since predictions are meaningless without both."""

    def __init__(self, params, x_mean, x_std, y_mean, y_std):
        self.params = params
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

    def predict_scalar(self, nsub, npch):
        """Differentiable scalar (Eu, g) prediction -- see _predict_scalar."""
        return _predict_scalar(self.params, self.x_mean, self.x_std,
                                self.y_mean, self.y_std, nsub, npch)

    def predict(self, nsub, npch):
        """Predict (Eu, g) at (nsub, npch), which may be scalars or arrays."""
        nsub = jnp.atleast_1d(jnp.asarray(nsub, dtype=jnp.float64))
        npch = jnp.atleast_1d(jnp.asarray(npch, dtype=jnp.float64))
        x = jnp.stack([nsub, npch], axis=-1)
        x_norm = _standardize(x, self.x_mean, self.x_std)
        y_norm = _predict_normalized(self.params, x_norm)
        y = _unstandardize(y_norm, self.y_mean, self.y_std)
        return y[..., 0], y[..., 1]  # Eu_pred, g_pred


def train_test_split(data, test_frac=0.2, seed=0):
    n = len(data["nsub"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train = {k: v[train_idx] for k, v in data.items()}
    test = {k: v[test_idx] for k, v in data.items()}
    return train, test


def train_surrogate(train_data, hidden_sizes=(64, 64), n_epochs=3000, lr=1e-3, seed=0):
    """Train the standard MLP surrogate on (Nsub, Npch) -> (Eu, g).

    Returns (TrainedSurrogate, history) where history is a dict with a
    single key, `train_loss` (a per-epoch array) -- for diagnosing
    under/overfitting by comparing against a SEPARATE `field_error(surrogate,
    test_data)` call after training, not by passing test data into this
    function itself (found via external audit: this docstring previously,
    incorrectly, promised an optional test-loss array this function has
    never computed or accepted a parameter for)."""
    x = np.stack([train_data["nsub"], train_data["npch"]], axis=-1)
    y = np.stack([train_data["Eu"], train_data["g"]], axis=-1)

    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)

    x_norm = jnp.asarray(_standardize(x, x_mean, x_std))
    y_norm = jnp.asarray(_standardize(y, y_mean, y_std))

    rng = jax.random.PRNGKey(seed)
    layer_sizes = (2,) + tuple(hidden_sizes) + (2,)
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

    surrogate = TrainedSurrogate(
        params,
        jnp.asarray(x_mean), jnp.asarray(x_std),
        jnp.asarray(y_mean), jnp.asarray(y_std),
    )
    return surrogate, dict(train_loss=np.array(losses))


def field_error(surrogate, data):
    """Standard field-error metrics on a dataset (train or test): per-target
    RMSE in the ORIGINAL (unnormalized) units, plus R^2, on Eu and g
    separately -- this is the "field error" half of the decoupling
    comparison; boundary error is measured separately, not here."""
    eu_pred, g_pred = surrogate.predict(data["nsub"], data["npch"])
    eu_pred, g_pred = np.array(eu_pred), np.array(g_pred)

    def rmse_r2(true, pred):
        rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot)
        return rmse, r2

    eu_rmse, eu_r2 = rmse_r2(data["Eu"], eu_pred)
    g_rmse, g_r2 = rmse_r2(data["g"], g_pred)
    return dict(Eu_rmse=eu_rmse, Eu_r2=eu_r2, g_rmse=g_rmse, g_r2=g_r2)
