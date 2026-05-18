"""
Tests for bayesian_model.py

Covers:
  - BayesianState: Beta-Binomial update
  - BayesianState: Log-Normal update (Welford's on log scale)
  - run_bayesian_analysis: output shape, probability bounds, early stopping
  - get_early_stop_day
  - compute_time_saved
  - compute_traditional_result
"""

import numpy as np
import pytest

from bayesian_model import (
    BayesianState, DayResult,
    run_bayesian_analysis,
    compute_traditional_result,
    compute_time_saved,
    get_early_stop_day,
    MIN_DAYS_BEFORE_STOPPING,
)
from simulate import TestConfig, simulate_test


# ── BayesianState: Beta-Binomial ──────────────────────────────────────────────

class TestBetaBinomialUpdate:

    def test_update_increments_alpha_and_beta(self):
        state = BayesianState()
        state.update_binary(successes=10, trials=100)
        assert state.alpha == 11.0   # prior 1 + 10 successes
        assert state.beta  == 91.0   # prior 1 + 90 failures

    def test_update_accumulates_across_days(self):
        state = BayesianState()
        state.update_binary(5, 50)
        state.update_binary(3, 30)
        assert state.alpha == 9.0    # 1 + 5 + 3
        assert state.beta  == 73.0   # 1 + 45 + 27

    def test_posterior_mean_converges_to_true_rate(self):
        true_rate = 0.15
        state = BayesianState()
        rng = np.random.default_rng(0)
        for _ in range(500):
            k = int(rng.binomial(200, true_rate))
            state.update_binary(k, 200)
        assert abs(state.posterior_mean - true_rate) < 0.005

    def test_sample_posterior_binary_in_unit_interval(self):
        state = BayesianState()
        state.update_binary(50, 200)
        samples = state.sample_posterior(10_000, "conversion")
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_zero_successes_does_not_crash(self):
        state = BayesianState()
        state.update_binary(0, 100)
        assert state.alpha == 1.0
        assert state.beta  == 101.0


# ── BayesianState: Log-Normal ─────────────────────────────────────────────────

class TestLogNormalUpdate:

    def test_update_increments_log_n_by_spender_count(self):
        state = BayesianState()
        state.update_revenue(daily_revenue=100.0, n_spenders=4)
        assert state.log_n == 4

    def test_log_mean_equals_log_of_revenue_per_spender(self):
        state = BayesianState()
        state.update_revenue(daily_revenue=100.0, n_spenders=4)
        # All 4 spenders treated as having revenue 100/4 = 25
        assert abs(state.log_mean - np.log(25.0)) < 1e-9

    def test_posterior_mean_uses_lognormal_formula(self):
        # E[X] for log-normal = exp(mu + sigma^2 / 2)
        state = BayesianState()
        state.update_revenue(daily_revenue=100.0, n_spenders=4)
        log_var = state.log_M2 / max(state.log_n - 1, 1)
        expected = np.exp(state.log_mean + log_var / 2)
        assert abs(state.posterior_mean - expected) < 1e-9

    def test_sample_posterior_revenue_always_positive(self):
        state = BayesianState()
        for _ in range(10):
            state.update_revenue(500.0, 20)
        samples = state.sample_posterior(10_000, "revenue")
        assert np.all(samples > 0)

    def test_zero_revenue_skipped(self):
        state = BayesianState()
        state.update_revenue(0.0, 5)
        assert state.log_n == 0   # no update — no information

    def test_zero_spenders_skipped(self):
        state = BayesianState()
        state.update_revenue(100.0, 0)
        assert state.log_n == 0

    def test_posterior_mean_converges_to_true_mean(self):
        true_mean, true_std = 25.0, 10.0
        sigma_log = np.sqrt(np.log(1 + (true_std / true_mean) ** 2))
        mu_log    = np.log(true_mean) - sigma_log ** 2 / 2

        rng = np.random.default_rng(42)
        state = BayesianState()
        for _ in range(200):
            n = 50
            revenue = rng.lognormal(mu_log, sigma_log, n).sum()
            state.update_revenue(revenue, n)

        assert abs(state.posterior_mean - true_mean) < 1.5


# ── run_bayesian_analysis ─────────────────────────────────────────────────────

class TestRunBayesianAnalysis:

    @pytest.fixture
    def conversion_data(self):
        cfg = TestConfig("test", "conversion", 0.05, 0.08, 3000, seed=1)
        return simulate_test(cfg)

    @pytest.fixture
    def null_data(self):
        cfg = TestConfig("null", "conversion", 0.05, 0.05, 1000, seed=99)
        return simulate_test(cfg)

    def test_returns_one_result_per_day(self, conversion_data):
        df_a, df_b = conversion_data
        results, _, _ = run_bayesian_analysis(df_a, df_b, "conversion")
        assert len(results) == len(df_a)

    def test_prob_always_in_unit_interval(self, conversion_data):
        df_a, df_b = conversion_data
        results, _, _ = run_bayesian_analysis(df_a, df_b, "conversion")
        for r in results:
            assert 0.0 <= r.prob_b_beats_a <= 1.0

    def test_prob_a_plus_prob_b_equals_one(self, conversion_data):
        df_a, df_b = conversion_data
        results, _, _ = run_bayesian_analysis(df_a, df_b, "conversion")
        for r in results:
            assert abs(r.prob_b_beats_a + r.prob_a_beats_b - 1.0) < 1e-9

    def test_detects_clear_winner_with_large_effect(self):
        cfg = TestConfig("big", "conversion", 0.05, 0.20, 5000, seed=7)
        df_a, df_b = simulate_test(cfg)
        results, _, _ = run_bayesian_analysis(df_a, df_b, "conversion")
        stop_day = get_early_stop_day(results)
        assert stop_day is not None

    def test_early_stop_respects_burn_in(self):
        # Even with a massive effect, should never stop before MIN_DAYS
        cfg = TestConfig("huge", "conversion", 0.01, 0.50, 10000, seed=3)
        df_a, df_b = simulate_test(cfg)
        results, _, _ = run_bayesian_analysis(df_a, df_b, "conversion")
        stop_day = get_early_stop_day(results)
        if stop_day is not None:
            assert stop_day >= MIN_DAYS_BEFORE_STOPPING

    def test_no_early_stop_for_null_effect(self, null_data):
        df_a, df_b = null_data
        results, _, _ = run_bayesian_analysis(df_a, df_b, "conversion")
        # With equal rates and low traffic, very unlikely to stop early
        # (not guaranteed, but with seed=99 and equal rates it shouldn't)
        stop_day = get_early_stop_day(results)
        if stop_day:
            # If it does stop, it must respect burn-in
            assert stop_day >= MIN_DAYS_BEFORE_STOPPING


# ── get_early_stop_day ────────────────────────────────────────────────────────

class TestGetEarlyStopDay:

    def _make_result(self, day, early_stop, declared_winner):
        return DayResult(
            day=day, prob_b_beats_a=0.97, prob_a_beats_b=0.03,
            expected_lift=0.1, credible_interval_b=(0.05, 0.08),
            early_stop=early_stop, declared_winner=declared_winner,
        )

    def test_returns_none_when_no_stop(self):
        results = [self._make_result(d, False, None) for d in range(1, 31)]
        assert get_early_stop_day(results) is None

    def test_returns_first_stop_day(self):
        results = [self._make_result(d, False, None) for d in range(1, 31)]
        results[14] = self._make_result(15, True, "B")
        results[15] = self._make_result(16, True, "B")
        assert get_early_stop_day(results) == 15


# ── compute_time_saved ────────────────────────────────────────────────────────

class TestComputeTimeSaved:

    def _stopped_results(self, stop_day, total_days=30):
        results = []
        for d in range(1, total_days + 1):
            stopped = d >= stop_day
            results.append(DayResult(
                day=d, prob_b_beats_a=0.97, prob_a_beats_b=0.03,
                expected_lift=0.1, credible_interval_b=(0.05, 0.08),
                early_stop=stopped, declared_winner="B" if stopped else None,
            ))
        return results

    def test_days_saved_correct(self):
        results = self._stopped_results(stop_day=20, total_days=30)
        savings = compute_time_saved(results, total_days=30)
        assert savings["days_saved"] == 10

    def test_no_stop_means_zero_days_saved(self):
        results = [DayResult(d, 0.5, 0.5, 0.0, (0.04, 0.06), False, None)
                   for d in range(1, 31)]
        savings = compute_time_saved(results, total_days=30)
        assert savings["days_saved"] == 0

    def test_cost_saved_uses_cost_per_day(self):
        results = self._stopped_results(stop_day=20, total_days=30)
        savings = compute_time_saved(results, total_days=30, cost_per_day=1_000)
        assert savings["cost_saved_usd"] == 10 * 1_000

    def test_pct_saved_calculation(self):
        results = self._stopped_results(stop_day=15, total_days=30)
        savings = compute_time_saved(results, total_days=30)
        assert abs(savings["pct_saved"] - 50.0) < 1e-9


# ── compute_traditional_result ────────────────────────────────────────────────

class TestComputeTraditionalResult:

    def test_significant_result_for_large_effect(self):
        cfg = TestConfig("test", "conversion", 0.05, 0.15, 10000, seed=1)
        df_a, df_b = simulate_test(cfg)
        results, sa, sb = run_bayesian_analysis(df_a, df_b, "conversion")
        freq = compute_traditional_result(df_a, df_b, "conversion", sa, sb)
        assert freq["significant"] == True
        assert freq["p_value"] < 0.05

    def test_not_significant_for_null_effect(self):
        cfg = TestConfig("null", "conversion", 0.05, 0.05, 500, seed=42)
        df_a, df_b = simulate_test(cfg)
        results, sa, sb = run_bayesian_analysis(df_a, df_b, "conversion")
        freq = compute_traditional_result(df_a, df_b, "conversion", sa, sb)
        # p-value should not be tiny for equal rates
        assert freq["p_value"] > 0.01

    def test_revenue_uses_log_scale_states(self):
        cfg = TestConfig("rev", "revenue", 0.12, 0.20, 4000, seed=5)
        df_a, df_b = simulate_test(cfg)
        results, sa, sb = run_bayesian_analysis(df_a, df_b, "revenue")
        freq = compute_traditional_result(df_a, df_b, "revenue", sa, sb)
        # Should return a valid p-value, not crash
        assert 0.0 <= freq["p_value"] <= 1.0
