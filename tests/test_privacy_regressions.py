"""Regression tests for DP-sensitive outputs, numerical validation, and accounting."""
import math
import random

import pytest

from agents.base import AuditTrail
from diff_privacy_engine.engine import (
    MomentsAccountant,
    PrivacyBudgetCalculator,
    aggregate_gradients,
    clip_gradient,
    dp_sgd_noise,
    dp_sgd_step,
    gaussian_noise_scale,
    laplace_noise_scale,
    mean_sensitivity,
    rdp_composition_gaussian,
    rdp_to_dp,
    run_gaussian_query,
    run_laplace_query,
)


def test_public_queries_do_not_disclose_private_input_or_noise():
    for result in (run_gaussian_query(27.0, 1, 0.5, 1e-5),
                   run_laplace_query(27.0, 1, 0.5)):
        assert "noisy_value" in result
        assert "true_value" not in result
        assert "noise_added" not in result


def test_mechanism_noise_is_not_controlled_by_global_prng_seed():
    random.seed(42)
    first = run_laplace_query(3, 1, 1)["noisy_value"]
    random.seed(42)
    second = run_laplace_query(3, 1, 1)["noisy_value"]
    assert first != second


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_reject_bad_sensitivity(value):
    with pytest.raises(ValueError):
        gaussian_noise_scale(value, 1, 1e-5)
    with pytest.raises(ValueError):
        laplace_noise_scale(value, 1)


def test_gaussian_epsilon_above_one_uses_valid_rdp_calibration():
    delta = 1e-5
    sigma = gaussian_noise_scale(1, 2.0, delta)
    orders = [1 + i / 10 for i in range(1, 1000)]
    rdp = rdp_composition_gaussian(sigma, 1, orders)
    assert rdp_to_dp(rdp, delta) <= 2.0 + 1e-6


def test_mean_sensitivity_fixed_size_replacement_adjacency():
    assert mean_sensitivity(10, 20, 5) == 2.0
    with pytest.raises(ValueError):
        mean_sensitivity(20, 10, 5)


def test_dp_sgd_dimensionless_multiplier_independent_of_clip_norm():
    assert dp_sgd_noise(2, 1, 1e-5, 4) == pytest.approx(
        dp_sgd_noise(1, 1, 1e-5, 4)
    )


def test_dp_sgd_rejects_invalid_batches_and_noise():
    with pytest.raises(ValueError):
        dp_sgd_step([[1, 2], [1]], 1, 1)
    with pytest.raises(ValueError):
        dp_sgd_step([[1, 2]], 1, 0)
    with pytest.raises(ValueError):
        dp_sgd_step([[float("nan"), 1]], 1, 1)


def test_accountant_matches_gaussian_rdp_composition():
    acct = MomentsAccountant(1e-5)
    acct.accumulate(sigma=1.2, num_steps=5)
    acct.accumulate(sigma=1.2, num_steps=7)
    alphas = [1 + i / 10 for i in range(1, 100)]
    expected = rdp_to_dp(rdp_composition_gaussian(1.2, 12, alphas), 1e-5)
    assert acct.get_privacy_spent()[0] == pytest.approx(expected)
    assert acct.steps == 12


def test_accountant_rejects_nonpositive_sigma_and_changed_orders():
    acct = MomentsAccountant(1e-5)
    with pytest.raises(ValueError):
        acct.accumulate(0)
    with pytest.raises(ValueError):
        acct.accumulate(1, -1)
    acct.accumulate(1, 1, [2, 3])
    with pytest.raises(ValueError):
        acct.accumulate(1, 1, [2, 4])
    with pytest.raises(ValueError):
        acct.get_privacy_spent([2, 4])


def test_privacy_budget_rejects_nonfinite_spending():
    budget = PrivacyBudgetCalculator(5, 1e-5)
    with pytest.raises(ValueError):
        budget.spend(float("nan"))
    with pytest.raises(ValueError):
        budget.spend(1, float("inf"))


def test_audit_signature_detects_content_and_signature_tampering():
    trail = AuditTrail(secret_key="test-only-key")
    trail.log("actor", "tier", "event", {"result": "success"})
    assert trail.verify_integrity()
    trail.logs[0]["actor"] = "altered"
    assert not trail.verify_integrity()
    trail.logs[0]["actor"] = "actor"
    assert trail.verify_integrity()
    trail.logs[0]["current_hash"] = "0" * 64
    assert not trail.verify_integrity()
