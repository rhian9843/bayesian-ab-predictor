"""
Tests for simulate.py

Covers:
  - simulate_test: output shape, schema, monotonicity, rate bounds
  - Revenue metric: log-normal generates positive values
  - n_days field controls experiment length
"""

import numpy as np
import pytest

from simulate import TestConfig, simulate_test


@pytest.fixture
def conversion_cfg():
    return TestConfig("test", "conversion", 0.05, 0.08, 3000, seed=42)

@pytest.fixture
def revenue_cfg():
    return TestConfig("rev", "revenue", 0.12, 0.16, 4000, seed=42)


class TestSimulateTest:

    def test_returns_two_dataframes(self, conversion_cfg):
        df_a, df_b = simulate_test(conversion_cfg)
        assert df_a is not None and df_b is not None

    def test_default_30_days(self, conversion_cfg):
        df_a, df_b = simulate_test(conversion_cfg)
        assert len(df_a) == 30
        assert len(df_b) == 30

    def test_custom_n_days(self):
        cfg = TestConfig("t", "conversion", 0.05, 0.06, 1000, n_days=14, seed=1)
        df_a, df_b = simulate_test(cfg)
        assert len(df_a) == 14
        assert len(df_b) == 14

    def test_day_column_is_sequential(self, conversion_cfg):
        df_a, _ = simulate_test(conversion_cfg)
        assert list(df_a["day"]) == list(range(1, 31))

    def test_required_columns_present(self, conversion_cfg):
        df_a, df_b = simulate_test(conversion_cfg)
        for col in ("day", "users", "events", "rate", "spenders"):
            assert col in df_a.columns
            assert col in df_b.columns

    def test_cumulative_users_monotonically_increasing(self, conversion_cfg):
        df_a, df_b = simulate_test(conversion_cfg)
        assert (df_a["users"].diff().dropna() > 0).all()
        assert (df_b["users"].diff().dropna() > 0).all()

    def test_cumulative_events_non_decreasing(self, conversion_cfg):
        df_a, df_b = simulate_test(conversion_cfg)
        assert (df_a["events"].diff().dropna() >= 0).all()
        assert (df_b["events"].diff().dropna() >= 0).all()

    def test_conversion_rate_in_unit_interval(self, conversion_cfg):
        df_a, df_b = simulate_test(conversion_cfg)
        assert (df_a["rate"] >= 0.0).all() and (df_a["rate"] <= 1.0).all()
        assert (df_b["rate"] >= 0.0).all() and (df_b["rate"] <= 1.0).all()

    def test_spenders_equals_events_for_conversion(self, conversion_cfg):
        df_a, _ = simulate_test(conversion_cfg)
        assert (df_a["spenders"] == df_a["events"]).all()

    def test_revenue_events_non_negative(self, revenue_cfg):
        df_a, df_b = simulate_test(revenue_cfg)
        assert (df_a["events"] >= 0).all()
        assert (df_b["events"] >= 0).all()

    def test_revenue_rate_non_negative(self, revenue_cfg):
        df_a, df_b = simulate_test(revenue_cfg)
        assert (df_a["rate"] >= 0).all()
        assert (df_b["rate"] >= 0).all()

    def test_revenue_spenders_leq_users(self, revenue_cfg):
        # Spenders ≤ users: can't have more buyers than visitors
        df_a, _ = simulate_test(revenue_cfg)
        assert (df_a["spenders"] <= df_a["users"]).all()

    def test_seed_gives_reproducible_results(self):
        cfg = TestConfig("t", "conversion", 0.05, 0.06, 1000, seed=7)
        df1_a, _ = simulate_test(cfg)
        df2_a, _ = simulate_test(cfg)
        assert (df1_a["events"] == df2_a["events"]).all()

    def test_none_seed_gives_different_results(self):
        cfg1 = TestConfig("t", "conversion", 0.05, 0.06, 1000, seed=None)
        cfg2 = TestConfig("t", "conversion", 0.05, 0.06, 1000, seed=None)
        df1_a, _ = simulate_test(cfg1)
        df2_a, _ = simulate_test(cfg2)
        # Extremely unlikely to be identical with seed=None
        assert not (df1_a["events"] == df2_a["events"]).all()
