"""
Real tests for Differential Privacy Calculator Engine.
"""
import math
import random
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diff_privacy_engine.engine import (
    count_sensitivity, sum_sensitivity, mean_sensitivity, histogram_sensitivity,
    gaussian_noise_scale, gaussian_mechanism, gaussian_mechanism_vector,
    laplace_noise_scale, laplace_mechanism, laplace_mechanism_vector,
    basic_composition, advanced_composition,
    rdp_epsilon, rdp_to_dp, rdp_composition_gaussian, find_sigma_for_rdp,
    clip_gradient, clip_gradients, aggregate_gradients,
    dp_sgd_noise, dp_sgd_step,
    MomentsAccountant, PrivacyBudgetCalculator,
    run_gaussian_query, run_laplace_query,
)


class TestSensitivity(unittest.TestCase):

    def test_count_sensitivity(self):
        self.assertEqual(count_sensitivity(), 1.0)

    def test_sum_sensitivity_symmetric(self):
        self.assertEqual(sum_sensitivity(-5, 5), 5.0)

    def test_sum_sensitivity_asymmetric(self):
        self.assertEqual(sum_sensitivity(-10, 3), 10.0)

    def test_sum_sensitivity_positive(self):
        self.assertEqual(sum_sensitivity(0, 100), 100.0)

    def test_mean_sensitivity(self):
        s = mean_sensitivity(0, 10, 100)
        self.assertAlmostEqual(s, 0.1)

    def test_mean_sensitivity_symmetric(self):
        s = mean_sensitivity(-5, 5, 50)
        self.assertAlmostEqual(s, 0.2)

    def test_mean_sensitivity_zero_n_raises(self):
        with self.assertRaises(ValueError):
            mean_sensitivity(0, 1, 0)

    def test_histogram_sensitivity(self):
        self.assertEqual(histogram_sensitivity(), 1.0)


class TestGaussianMechanism(unittest.TestCase):

    def test_noise_scale_formula(self):
        sigma = gaussian_noise_scale(1.0, 1.0, 1e-5)
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.0
        self.assertAlmostEqual(sigma, expected, places=10)

    def test_noise_scale_proportional_to_sensitivity(self):
        s1 = gaussian_noise_scale(1.0, 1.0, 1e-5)
        s2 = gaussian_noise_scale(2.0, 1.0, 1e-5)
        self.assertAlmostEqual(s2, 2.0 * s1, places=10)

    def test_noise_scale_inversely_proportional_to_epsilon(self):
        s1 = gaussian_noise_scale(1.0, 1.0, 1e-5)
        s2 = gaussian_noise_scale(1.0, 2.0, 1e-5)
        self.assertAlmostEqual(s2, s1 / 2.0, places=10)

    def test_noise_scale_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            gaussian_noise_scale(1.0, 0.0, 1e-5)

    def test_noise_scale_invalid_delta_zero(self):
        with self.assertRaises(ValueError):
            gaussian_noise_scale(1.0, 1.0, 0.0)

    def test_noise_scale_invalid_delta_one(self):
        with self.assertRaises(ValueError):
            gaussian_noise_scale(1.0, 1.0, 1.0)

    def test_gaussian_mechanism_returns_float(self):
        random.seed(42)
        result = gaussian_mechanism(10.0, 1.0, 1.0, 1e-5)
        self.assertIsInstance(result, float)

    def test_gaussian_mechanism_vector_length(self):
        random.seed(42)
        result = gaussian_mechanism_vector([1.0, 2.0, 3.0], 1.0, 1.0, 1e-5)
        self.assertEqual(len(result), 3)

    def test_gaussian_mechanism_statistical(self):
        random.seed(42)
        samples = [gaussian_mechanism(100.0, 1.0, 1.0, 1e-5) for _ in range(10000)]
        mean = sum(samples) / len(samples)
        self.assertAlmostEqual(mean, 100.0, delta=1.0)


class TestLaplaceMechanism(unittest.TestCase):

    def test_noise_scale_formula(self):
        b = laplace_noise_scale(1.0, 1.0)
        self.assertAlmostEqual(b, 1.0)

    def test_noise_scale_proportional_to_sensitivity(self):
        b = laplace_noise_scale(5.0, 1.0)
        self.assertAlmostEqual(b, 5.0)

    def test_noise_scale_inversely_proportional_to_epsilon(self):
        b = laplace_noise_scale(1.0, 0.5)
        self.assertAlmostEqual(b, 2.0)

    def test_noise_scale_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            laplace_noise_scale(1.0, 0.0)

    def test_laplace_mechanism_returns_float(self):
        random.seed(42)
        result = laplace_mechanism(10.0, 1.0, 1.0)
        self.assertIsInstance(result, float)

    def test_laplace_mechanism_vector_length(self):
        random.seed(42)
        result = laplace_mechanism_vector([1.0, 2.0, 3.0], 1.0, 1.0)
        self.assertEqual(len(result), 3)

    def test_laplace_mechanism_statistical(self):
        random.seed(42)
        samples = [laplace_mechanism(50.0, 1.0, 1.0) for _ in range(10000)]
        mean = sum(samples) / len(samples)
        self.assertAlmostEqual(mean, 50.0, delta=0.5)


class TestBasicComposition(unittest.TestCase):

    def test_basic_single(self):
        self.assertAlmostEqual(basic_composition(1.0, 1), 1.0)

    def test_basic_multiple(self):
        self.assertAlmostEqual(basic_composition(0.5, 10), 5.0)

    def test_basic_zero(self):
        self.assertAlmostEqual(basic_composition(1.0, 0), 0.0)

    def test_basic_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            basic_composition(-1.0, 5)

    def test_basic_invalid_k(self):
        with self.assertRaises(ValueError):
            basic_composition(1.0, -1)


class TestAdvancedComposition(unittest.TestCase):

    def test_advanced_returns_tuple(self):
        result = advanced_composition(0.1, 1e-6, 10, 1e-5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_advanced_epsilon_positive(self):
        eps, delta = advanced_composition(0.1, 1e-6, 10, 1e-5)
        self.assertGreater(eps, 0)

    def test_advanced_delta_includes_prime(self):
        eps, delta = advanced_composition(0.1, 1e-6, 10, 1e-5)
        self.assertGreaterEqual(delta, 1e-5)

    def test_advanced_better_than_basic_for_small_epsilon(self):
        basic_eps = basic_composition(0.01, 100)
        adv_eps, _ = advanced_composition(0.01, 0.0, 100, 1e-5)
        self.assertLess(adv_eps, basic_eps)

    def test_advanced_invalid_delta_prime(self):
        with self.assertRaises(ValueError):
            advanced_composition(0.1, 1e-6, 10, 0.0)


class TestRDPComposition(unittest.TestCase):

    def test_rdp_epsilon_formula(self):
        eps = rdp_epsilon(2.0, 1.0, 1.0)
        expected = 1.0 / (2.0 * 1.0) * 2.0
        self.assertAlmostEqual(eps, expected)

    def test_rdp_epsilon_proportional_to_alpha(self):
        e1 = rdp_epsilon(2.0, 1.0, 1.0)
        e2 = rdp_epsilon(4.0, 1.0, 1.0)
        self.assertAlmostEqual(e2, 2.0 * e1)

    def test_rdp_epsilon_invalid_alpha(self):
        with self.assertRaises(ValueError):
            rdp_epsilon(1.0, 1.0, 1.0)

    def test_rdp_to_dp_basic(self):
        rdp_eps = {2.0: 0.5, 4.0: 0.3, 8.0: 0.2}
        eps = rdp_to_dp(rdp_eps, 1e-5)
        self.assertGreater(eps, 0)
        self.assertLess(eps, float('inf'))

    def test_rdp_to_dp_invalid_delta(self):
        with self.assertRaises(ValueError):
            rdp_to_dp({2.0: 0.5}, 0.0)

    def test_rdp_composition_linear(self):
        sigma = 1.0
        alphas = [2.0, 4.0, 8.0]
        rdp1 = rdp_composition_gaussian(sigma, 1, alphas)
        rdp5 = rdp_composition_gaussian(sigma, 5, alphas)
        for a in alphas:
            self.assertAlmostEqual(rdp5[a], 5.0 * rdp1[a])

    def test_find_sigma_for_rdp(self):
        sigma = find_sigma_for_rdp(1.0, 1e-5, 100)
        self.assertGreater(sigma, 0)
        alphas = [1.0 + x / 10.0 for x in range(1, 100)]
        rdp_eps = rdp_composition_gaussian(sigma, 100, alphas)
        dp_eps = rdp_to_dp(rdp_eps, 1e-5)
        self.assertLessEqual(dp_eps, 1.0 + 0.2)


class TestGradientClipping(unittest.TestCase):

    def test_clip_no_change_if_within_norm(self):
        grad = [0.1, 0.2, 0.3]
        clipped = clip_gradient(grad, 10.0)
        for a, b in zip(grad, clipped):
            self.assertAlmostEqual(a, b)

    def test_clip_reduces_norm(self):
        grad = [10.0, 10.0, 10.0]
        clipped = clip_gradient(grad, 1.0)
        norm = math.sqrt(sum(g**2 for g in clipped))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_clip_preserves_direction(self):
        grad = [3.0, 4.0]
        clipped = clip_gradient(grad, 1.0)
        self.assertAlmostEqual(clipped[0] / clipped[1], 3.0 / 4.0, places=5)

    def test_clip_invalid_norm(self):
        with self.assertRaises(ValueError):
            clip_gradient([1.0], 0.0)

    def test_clip_gradients_batch(self):
        grads = [[10.0, 0.0], [0.0, 10.0]]
        clipped = clip_gradients(grads, 1.0)
        for g in clipped:
            norm = math.sqrt(sum(x**2 for x in g))
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_aggregate_gradients(self):
        grads = [[1.0, 2.0], [3.0, 4.0]]
        agg = aggregate_gradients(grads)
        self.assertAlmostEqual(agg[0], 2.0)
        self.assertAlmostEqual(agg[1], 3.0)


class TestDPSGD(unittest.TestCase):

    def test_dp_sgd_noise_positive(self):
        noise = dp_sgd_noise(1.0, 1.0, 1e-5)
        self.assertGreater(noise, 0)

    def test_dp_sgd_noise_formula(self):
        noise = dp_sgd_noise(1.0, 1.0, 1e-5)
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5))
        self.assertAlmostEqual(noise, expected, places=10)

    def test_dp_sgd_step_returns_correct_length(self):
        random.seed(42)
        grads = [[random.gauss(0, 1) for _ in range(5)] for _ in range(10)]
        result = dp_sgd_step(grads, 1.0, 1.0)
        self.assertEqual(len(result), 5)

    def test_dp_sgd_step_invalid_clip_norm(self):
        with self.assertRaises(ValueError):
            dp_sgd_noise(0.0, 1.0, 1e-5)


class TestMomentsAccountant(unittest.TestCase):

    def test_initial_privacy_spent(self):
        acct = MomentsAccountant(1e-5)
        eps, delta = acct.get_privacy_spent()
        self.assertAlmostEqual(eps, 0.0)

    def test_accumulate_increases_privacy(self):
        acct = MomentsAccountant(1e-5)
        acct.accumulate(1.0, 10)
        eps, _ = acct.get_privacy_spent()
        self.assertGreater(eps, 0.0)

    def test_more_steps_more_privacy(self):
        acct1 = MomentsAccountant(1e-5)
        acct1.accumulate(1.0, 10)
        acct2 = MomentsAccountant(1e-5)
        acct2.accumulate(1.0, 100)
        eps1, _ = acct1.get_privacy_spent()
        eps2, _ = acct2.get_privacy_spent()
        self.assertGreater(eps2, eps1)

    def test_invalid_delta(self):
        with self.assertRaises(ValueError):
            MomentsAccountant(0.0)


class TestPrivacyBudgetCalculator(unittest.TestCase):

    def test_initial_budget(self):
        calc = PrivacyBudgetCalculator(10.0, 1e-5)
        self.assertAlmostEqual(calc.remaining_epsilon, 10.0)
        self.assertFalse(calc.is_exhausted)

    def test_spend_reduces_budget(self):
        calc = PrivacyBudgetCalculator(10.0, 1e-5)
        ok = calc.spend(3.0, 0.0, "query 1")
        self.assertTrue(ok)
        self.assertAlmostEqual(calc.remaining_epsilon, 7.0)

    def test_spend_exhausted(self):
        calc = PrivacyBudgetCalculator(5.0, 1e-5)
        calc.spend(5.0, 0.0)
        self.assertTrue(calc.is_exhausted)
        ok = calc.spend(1.0, 0.0)
        self.assertFalse(ok)

    def test_spend_delta_budget(self):
        calc = PrivacyBudgetCalculator(10.0, 1e-5)
        ok = calc.spend(1.0, 2e-5)
        self.assertFalse(ok)

    def test_budget_summary(self):
        calc = PrivacyBudgetCalculator(10.0, 1e-5)
        calc.spend(3.0, 1e-6)
        summary = calc.budget_summary()
        self.assertEqual(summary["num_queries"], 1)
        self.assertAlmostEqual(summary["spent_epsilon"], 3.0)

    def test_invalid_total_epsilon(self):
        with self.assertRaises(ValueError):
            PrivacyBudgetCalculator(0.0, 1e-5)

    def test_invalid_total_delta(self):
        with self.assertRaises(ValueError):
            PrivacyBudgetCalculator(1.0, 1.0)


class TestRunQueries(unittest.TestCase):

    def test_run_gaussian_query(self):
        random.seed(42)
        result = run_gaussian_query(100.0, 1.0, 1.0, 1e-5)
        self.assertEqual(result["mechanism"], "gaussian")
        self.assertEqual(result["true_value"], 100.0)
        self.assertIn("noisy_value", result)
        self.assertIn("sigma", result)

    def test_run_laplace_query(self):
        random.seed(42)
        result = run_laplace_query(50.0, 1.0, 1.0)
        self.assertEqual(result["mechanism"], "laplace")
        self.assertEqual(result["true_value"], 50.0)
        self.assertIn("noisy_value", result)
        self.assertIn("scale_b", result)


if __name__ == "__main__":
    unittest.main()
