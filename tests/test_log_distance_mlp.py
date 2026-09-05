"""Validates the log-distance-feature surrogate (PROJECT_LOG.md Phase 4,
third fix candidate) purely as a competent, honestly-evaluated regressor
with the same interface as mlp.TrainedSurrogate. Says nothing about
boundary/decoupling error, which is a separate analysis.
"""

import jax
import numpy as np

from tide.surrogates.bt_point_data import FR_BT, KE_BT, KI_BT, LAM_BT, NSUB_BT, generate_bt_dataset
from tide.surrogates.log_distance_mlp import train_log_distance_surrogate
from tide.surrogates.mlp import field_error, train_test_split

N1 = 16


def _small_dataset():
    return generate_bt_dataset(FR_BT, LAM_BT, KI_BT, KE_BT, N1, nsub_range=(11.0, 14.0), n_nsub=20)


def test_training_loss_decreases_substantially():
    train, _ = train_test_split(_small_dataset(), seed=0)
    _, history = train_log_distance_surrogate(train, NSUB_BT, n_epochs=1500, seed=0)
    assert history["train_loss"][-1] < 0.1 * history["train_loss"][0]


def test_held_out_field_error_is_low():
    data = _small_dataset()
    train, test = train_test_split(data, seed=0)
    surrogate, _ = train_log_distance_surrogate(train, NSUB_BT, n_epochs=2000, seed=0)

    test_err = field_error(surrogate, test)
    assert test_err["Eu_r2"] > 0.99
    assert test_err["g_r2"] > 0.98


def test_predict_matches_field_error_shapes():
    data = _small_dataset()
    train, test = train_test_split(data, seed=0)
    surrogate, _ = train_log_distance_surrogate(train, NSUB_BT, n_epochs=300, seed=0)
    eu_pred, g_pred = surrogate.predict(test["nsub"], test["npch"])
    assert np.array(eu_pred).shape == test["nsub"].shape
    assert np.array(g_pred).shape == test["nsub"].shape


def test_predict_scalar_is_differentiable_wrt_npch():
    data = _small_dataset()
    train, _ = train_test_split(data, seed=0)
    surrogate, _ = train_log_distance_surrogate(train, NSUB_BT, n_epochs=300, seed=0)

    deu = jax.grad(lambda npch: surrogate.predict_scalar(12.0, npch)[0])
    g = float(deu(20.0))
    assert np.isfinite(g)


def test_handles_nsub_exactly_at_bt_point_without_nan():
    # log(|Nsub_BT - Nsub| + eps) must stay finite even at Nsub == Nsub_BT
    # (eps guards the log(0) singularity) -- exercised directly since this
    # is exactly the point where a careless implementation would blow up.
    data = _small_dataset()
    train, _ = train_test_split(data, seed=0)
    surrogate, _ = train_log_distance_surrogate(train, NSUB_BT, n_epochs=10, seed=0)
    eu, g = surrogate.predict_scalar(NSUB_BT, 20.0)
    assert np.isfinite(float(eu)) and np.isfinite(float(g))
