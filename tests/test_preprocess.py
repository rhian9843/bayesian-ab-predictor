"""
Tests for preprocess.py

Covers:
  - build_engine_df: cumulative columns, rate, day numbering, spenders
  - fix_missing: linear interpolation
  - load_and_preprocess: end-to-end with real Kaggle files
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from preprocess import build_engine_df, fix_missing, load_and_preprocess

KAGGLE_CONTROL = Path.home() / "Downloads/archive/control_group.csv"
KAGGLE_TEST    = Path.home() / "Downloads/archive/test_group.csv"
KAGGLE_AVAILABLE = KAGGLE_CONTROL.exists() and KAGGLE_TEST.exists()


class TestBuildEngineDF:

    @pytest.fixture
    def simple_raw(self):
        return pd.DataFrame({
            "events_col": [10, 20, 15],
            "users_col":  [100, 200, 150],
        })

    def test_cumulative_events(self, simple_raw):
        df = build_engine_df(simple_raw, "events_col", "users_col")
        assert df["events"].iloc[-1] == 45      # 10 + 20 + 15

    def test_cumulative_users(self, simple_raw):
        df = build_engine_df(simple_raw, "events_col", "users_col")
        assert df["users"].iloc[-1] == 450      # 100 + 200 + 150

    def test_rate_is_cumulative_events_over_users(self, simple_raw):
        df = build_engine_df(simple_raw, "events_col", "users_col")
        expected = 45 / 450
        assert abs(df["rate"].iloc[-1] - expected) < 1e-10

    def test_day_column_starts_at_one(self, simple_raw):
        df = build_engine_df(simple_raw, "events_col", "users_col")
        assert df["day"].iloc[0] == 1

    def test_day_column_is_sequential(self, simple_raw):
        df = build_engine_df(simple_raw, "events_col", "users_col")
        assert list(df["day"]) == [1, 2, 3]

    def test_spenders_equals_events(self, simple_raw):
        df = build_engine_df(simple_raw, "events_col", "users_col")
        assert (df["spenders"] == df["events"]).all()

    def test_required_columns_present(self, simple_raw):
        df = build_engine_df(simple_raw, "events_col", "users_col")
        for col in ("day", "users", "events", "rate", "spenders"):
            assert col in df.columns

    def test_output_length_matches_input(self, simple_raw):
        df = build_engine_df(simple_raw, "events_col", "users_col")
        assert len(df) == len(simple_raw)


class TestFixMissing:

    def test_interpolates_single_gap(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        fixed = fix_missing(df)
        assert abs(fixed["a"].iloc[1] - 2.0) < 1e-10

    def test_no_change_when_no_missing(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        fixed = fix_missing(df)
        assert (fixed["a"] == df["a"]).all()

    def test_fills_leading_nan(self):
        df = pd.DataFrame({"a": [np.nan, 2.0, 3.0]})
        fixed = fix_missing(df)
        assert not fixed["a"].isna().any()

    def test_fills_trailing_nan(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan]})
        fixed = fix_missing(df)
        assert not fixed["a"].isna().any()

    def test_non_numeric_columns_unchanged(self):
        df = pd.DataFrame({"label": ["a", "b", "c"], "val": [1.0, np.nan, 3.0]})
        fixed = fix_missing(df)
        assert list(fixed["label"]) == ["a", "b", "c"]


@pytest.mark.skipif(not KAGGLE_AVAILABLE, reason="Kaggle files not present")
class TestLoadAndPreprocess:

    def test_returns_30_days(self):
        df_a, df_b, _ = load_and_preprocess(str(KAGGLE_CONTROL), str(KAGGLE_TEST))
        assert len(df_a) == 30
        assert len(df_b) == 30

    def test_no_missing_values_after_preprocessing(self):
        df_a, df_b, _ = load_and_preprocess(str(KAGGLE_CONTROL), str(KAGGLE_TEST))
        assert not df_a.isna().any().any()
        assert not df_b.isna().any().any()

    def test_meta_contains_rates(self):
        _, _, meta = load_and_preprocess(str(KAGGLE_CONTROL), str(KAGGLE_TEST))
        assert "baseline_rate" in meta
        assert "treatment_rate" in meta
        assert 0 < meta["baseline_rate"] < 1
        assert 0 < meta["treatment_rate"] < 1

    def test_test_group_higher_purchase_rate(self):
        # Known from data: test group outperforms control on purchases/reach
        _, _, meta = load_and_preprocess(str(KAGGLE_CONTROL), str(KAGGLE_TEST))
        assert meta["treatment_rate"] > meta["baseline_rate"]

    def test_cumulative_users_monotonic(self):
        df_a, df_b, _ = load_and_preprocess(str(KAGGLE_CONTROL), str(KAGGLE_TEST))
        assert (df_a["users"].diff().dropna() > 0).all()
        assert (df_b["users"].diff().dropna() > 0).all()
