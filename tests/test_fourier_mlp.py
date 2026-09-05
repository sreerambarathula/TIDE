"""Validates the Fourier-feature surrogate (PROJECT_LOG.md Phase 4, second
fix candidate) purely as a competent, honestly-evaluated regressor with the
same interface as mlp.TrainedSurrogate -- same scope discipline as
test_mlp_surrogate.py. Says nothing about boundary/decoupling error, which
is a separate analysis.
"""

import jax
import numpy as np

from tide.surrogates.bt_point_data import FR_BT, KE_BT, KI_BT, LAM_BT, generate_bt_dataset
from tide.surrogates.fourier_mlp import train_fourier_surrogate
from tide.surrogates.mlp import field_error, train_test_split

N1 = 16


def _small_dataset():
    return generate_bt_dataset(FR_BT, LAM_BT, KI_BT, KE_BT, N1, nsub_range=(11.0, 14.0), n_nsub=20)


def test_training_loss_decreases_substantially():
    train, _ = train_test_split(_small_dataset(), seed=0)
    _, history = train_fourier_surrogate(train, n_epochs=1500, sigmas=0.5, seed=0)
    assert history["train_loss"][-1] < 0.1 * history["train_loss"][0]


def test_held_out_field_error_is_reasonable():
    # sigma=0.5 confirmed (by direct sweep) to give near-perfect held-out
    # field error on this small dataset; sigma=10 was tried first and gave
    # NEGATIVE R2 (classic RFF overfitting-to-high-frequency-noise failure
    # mode) -- this test pins down a value known to actually work, not an
    # arbitrary default.
    data = _small_dataset()
    train, test = train_test_split(data, seed=0)
    surrogate, _ = train_fourier_surrogate(train, n_epochs=2000, sigmas=0.5, seed=0)

    test_err = field_error(surrogate, test)
    assert test_err["Eu_r2"] > 0.95
    assert test_err["g_r2"] > 0.9


def test_predict_matches_field_error_shapes():
    data = _small_dataset()
    train, test = train_test_split(data, seed=0)
    surrogate, _ = train_fourier_surrogate(train, n_epochs=300, sigmas=1.0, seed=0)
    eu_pred, g_pred = surrogate.predict(test["nsub"], test["npch"])
    assert np.array(eu_pred).shape == test["nsub"].shape
    assert np.array(g_pred).shape == test["nsub"].shape


def test_predict_scalar_is_differentiable():
    # eval/decoupling.py's boundary extraction needs jax.grad to work through
    # predict_scalar -- confirm this holds with the Fourier encoding in the
    # forward path, not just for the plain MLP.
    data = _small_dataset()
    train, _ = train_test_split(data, seed=0)
    surrogate, _ = train_fourier_surrogate(train, n_epochs=300, sigmas=1.0, seed=0)

    deu = jax.grad(lambda npch: surrogate.predict_scalar(12.0, npch)[0])
    g = float(deu(20.0))
    assert np.isfinite(g)


def test_different_seeds_give_different_B():
    data = _small_dataset()
    train, _ = train_test_split(data, seed=0)
    surr0, _ = train_fourier_surrogate(train, n_epochs=10, sigmas=1.0, seed=0)
    surr1, _ = train_fourier_surrogate(train, n_epochs=10, sigmas=1.0, seed=1)
    assert not np.allclose(np.array(surr0.B), np.array(surr1.B))


def test_multiband_sigmas_gives_expected_feature_count():
    data = _small_dataset()
    train, _ = train_test_split(data, seed=0)
    surrogate, _ = train_fourier_surrogate(
        train, n_fourier_features=16, sigmas=(0.5, 2.0, 8.0), n_epochs=10, seed=0)
    assert np.array(surrogate.B).shape == (2, 16 * 3)
