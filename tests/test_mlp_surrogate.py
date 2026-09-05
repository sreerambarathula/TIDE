"""Validates the standard MLP surrogate (PROJECT_LOG.md Sec. 30) -- the
un-clever, field-error-minimizing baseline the paper's decoupling claim is
measured against. These tests only check it's a competent, honestly-evaluated
regressor; they say nothing about boundary error, which is a separate
analysis by design.
"""

import numpy as np

from tide.surrogates.data_generation import generate_stability_dataset
from tide.surrogates.mlp import field_error, train_surrogate, train_test_split

FR, LAM, KI, KE = 0.035, 5.90, 6.55, 2.03
N1 = 16


def _small_dataset():
    return generate_stability_dataset(FR, LAM, KI, KE, N1, nsub_range=(24.0, 36.0), n_nsub=30)


def test_training_loss_decreases_substantially():
    train, _ = train_test_split(_small_dataset(), seed=0)
    _, history = train_surrogate(train, n_epochs=1500, seed=0)
    assert history["train_loss"][-1] < 0.1 * history["train_loss"][0]


def test_held_out_field_error_is_low_and_matches_train_error():
    # A well-fit, non-overfit baseline: test error should be low AND close to
    # train error, not dramatically worse -- otherwise "field error is low"
    # wouldn't be a fair premise for the later boundary-error comparison.
    data = _small_dataset()
    train, test = train_test_split(data, seed=0)
    surrogate, _ = train_surrogate(train, n_epochs=2500, seed=0)

    train_err = field_error(surrogate, train)
    test_err = field_error(surrogate, test)

    assert test_err["Eu_r2"] > 0.99
    assert test_err["g_r2"] > 0.98
    assert test_err["Eu_rmse"] < 2.0 * train_err["Eu_rmse"] + 1e-6
    assert test_err["g_rmse"] < 2.0 * train_err["g_rmse"] + 1e-6


def test_predict_matches_field_error_shapes():
    data = _small_dataset()
    train, test = train_test_split(data, seed=0)
    surrogate, _ = train_surrogate(train, n_epochs=500, seed=0)
    eu_pred, g_pred = surrogate.predict(test["nsub"], test["npch"])
    assert np.array(eu_pred).shape == test["nsub"].shape
    assert np.array(g_pred).shape == test["nsub"].shape
