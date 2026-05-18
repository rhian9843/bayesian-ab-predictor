"""
Tests for bandit.py

Covers:
  - run_thompson_sampling: output shape, schema, allocation bounds,
    monotonicity of cumulative columns
"""

import numpy as np
import pytest

from bandit import run_thompson_sampling
from simulate import TestConfig


@pytest.fixture
def conversion_cfg():
    return TestConfig("test", "conversion", 0.05, 0.08, 3000, seed=42)

@pytest.fixture
def revenue_cfg():
    return TestConfig("rev", "revenue", 0.12, 0.16, 4000, seed=42)


class TestRunThompsonSampling:

    def test_returns_one_row_per_day(self, conversion_cfg):
        df = run_thompson_sampling(conversion_cfg, total_days=30)
        assert len(df) == 30

    def test_custom_total_days(self, conversion_cfg):
        df = run_thompson_sampling(conversion_cfg, total_days=14)
        assert len(df) == 14

    def test_required_columns_present(self, conversion_cfg):
        df = run_thompson_sampling(conversion_cfg)
        for col in ("day", "alloc_a", "alloc_b", "cum_events_a",
                    "cum_events_b", "cum_baseline_events", "prob_b_beats_a"):
            assert col in df.columns

    def test_allocations_sum_to_one(self, conversion_cfg):
        df = run_thompson_sampling(conversion_cfg)
        assert ((df["alloc_a"] + df["alloc_b"]).round(10) == 1.0).all()

    def test_alloc_b_bounded_between_10_and_90_pct(self, conversion_cfg):
        df = run_thompson_sampling(conversion_cfg)
        assert (df["alloc_b"] >= 0.10).all()
        assert (df["alloc_b"] <= 0.90).all()

    def test_cum_events_non_decreasing(self, conversion_cfg):
        df = run_thompson_sampling(conversion_cfg)
        assert (df["cum_events_a"].diff().dropna() >= 0).all()
        assert (df["cum_events_b"].diff().dropna() >= 0).all()

    def test_cum_baseline_events_non_decreasing(self, conversion_cfg):
        df = run_thompson_sampling(conversion_cfg)
        assert (df["cum_baseline_events"].diff().dropna() >= 0).all()

    def test_prob_b_beats_a_in_unit_interval(self, conversion_cfg):
        df = run_thompson_sampling(conversion_cfg)
        assert (df["prob_b_beats_a"] >= 0.0).all()
        assert (df["prob_b_beats_a"] <= 1.0).all()

    def test_superior_variant_gets_more_traffic(self):
        # With a large effect B >> A, bandit should allocate more to B
        cfg = TestConfig("big", "conversion", 0.02, 0.15, 5000, seed=1)
        df = run_thompson_sampling(cfg, total_days=30)
        # By the end, majority should go to B
        assert df["alloc_b"].iloc[-1] > 0.5

    def test_revenue_metric_runs_without_error(self, revenue_cfg):
        df = run_thompson_sampling(revenue_cfg, total_days=30)
        assert len(df) == 30
        assert (df["alloc_b"] >= 0.10).all()
