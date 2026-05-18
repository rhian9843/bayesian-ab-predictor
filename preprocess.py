"""
preprocess.py — Clean and reshape Kaggle A/B campaign CSVs

Input:  semicolon-delimited CSVs with one row per day
Output: two CSVs (control + test) in the format expected by run_bayesian_analysis

Columns in output:
    day       — sequential day number (1-N)
    users     — cumulative denominator (reach, impressions, etc.)
    events    — cumulative numerator   (purchases, clicks, etc.)
    rate      — events / users
    spenders  — same as events for conversion/ctr metrics

Usage (script):
    python preprocess.py

Usage (import):
    from preprocess import load_and_preprocess
    df_a, df_b, meta = load_and_preprocess(control_path, test_path, numerator, denominator)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Default column choices ─────────────────────────────────────────────────────
# Change these to switch metric:
#   conversion : numerator="# of Purchase",      denominator="Reach"
#   ctr        : numerator="# of Website Clicks", denominator="# of Impressions"
#   add-to-cart: numerator="# of Add to Cart",    denominator="Reach"

DEFAULT_NUMERATOR   = "# of Purchase"
DEFAULT_DENOMINATOR = "Reach"


def load_raw(path: str) -> pd.DataFrame:
    """Load a semicolon-delimited campaign CSV and parse dates."""
    df = pd.read_csv(path, sep=";")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def report_missing(df: pd.DataFrame, label: str) -> None:
    """Print a summary of any missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print(f"  {label}: no missing values")
    else:
        print(f"  {label}: missing values found —")
        for col, count in missing.items():
            rows = df.index[df[col].isnull()].tolist()
            dates = df.loc[rows, "Date"].dt.strftime("%d %b").tolist()
            print(f"    {col}: {count} row(s) on {dates}")


def fix_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linearly interpolate missing numeric values.
    For a campaign metric (e.g. reach), interpolation between the surrounding
    days is more honest than forward-fill or dropping the row.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    return df


def build_engine_df(df: pd.DataFrame, numerator: str, denominator: str) -> pd.DataFrame:
    """
    Reshape a daily campaign DataFrame into the cumulative format that
    run_bayesian_analysis expects.
    """
    daily_events = df[numerator].values
    daily_users  = df[denominator].values

    cum_events = np.cumsum(daily_events)
    cum_users  = np.cumsum(daily_users)
    rate       = cum_events / np.maximum(cum_users, 1)

    return pd.DataFrame({
        "day":      np.arange(1, len(df) + 1),
        "users":    cum_users.astype(int),
        "events":   cum_events.astype(int),
        "rate":     rate,
        "spenders": cum_events.astype(int),   # = events for conversion / ctr
    })


def load_and_preprocess(
    control_path: str,
    test_path: str,
    numerator: str   = DEFAULT_NUMERATOR,
    denominator: str = DEFAULT_DENOMINATOR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full pipeline: load → validate → fix → reshape.

    Returns
    -------
    df_a   : control group in engine format
    df_b   : test group in engine format
    meta   : dict with metric info and raw DataFrames for inspection
    """
    raw_a = load_raw(control_path)
    raw_b = load_raw(test_path)

    print("\n── Missing value report ──────────────────────────────")
    report_missing(raw_a, "Control")
    report_missing(raw_b, "Test")

    raw_a = fix_missing(raw_a)
    raw_b = fix_missing(raw_b)

    # Validate both have the same number of days
    if len(raw_a) != len(raw_b):
        raise ValueError(
            f"Row count mismatch: control has {len(raw_a)} days, "
            f"test has {len(raw_b)} days."
        )

    # Validate chosen columns exist
    for col in (numerator, denominator):
        for label, df in [("Control", raw_a), ("Test", raw_b)]:
            if col not in df.columns:
                available = [c for c in df.columns if c not in ("Campaign Name", "Date")]
                raise ValueError(
                    f"Column '{col}' not found in {label}.\n"
                    f"Available columns: {available}"
                )

    df_a = build_engine_df(raw_a, numerator, denominator)
    df_b = build_engine_df(raw_b, numerator, denominator)

    baseline_rate  = df_a["rate"].iloc[-1]
    treatment_rate = df_b["rate"].iloc[-1]

    meta = {
        "numerator":      numerator,
        "denominator":    denominator,
        "n_days":         len(df_a),
        "baseline_rate":  baseline_rate,
        "treatment_rate": treatment_rate,
        "lift":           (treatment_rate - baseline_rate) / baseline_rate if baseline_rate > 0 else 0,
        "raw_control":    raw_a,
        "raw_test":       raw_b,
    }

    return df_a, df_b, meta


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    HERE = Path(__file__).parent

    CONTROL_PATH = Path.home() / "Downloads/archive/control_group.csv"
    TEST_PATH    = Path.home() / "Downloads/archive/test_group.csv"

    print("── Preprocessing A/B campaign data ──────────────────")
    print(f"  Control : {CONTROL_PATH}")
    print(f"  Test    : {TEST_PATH}")
    print(f"  Metric  : {DEFAULT_NUMERATOR} / {DEFAULT_DENOMINATOR}")

    df_a, df_b, meta = load_and_preprocess(
        str(CONTROL_PATH),
        str(TEST_PATH),
        numerator=DEFAULT_NUMERATOR,
        denominator=DEFAULT_DENOMINATOR,
    )

    out_a = HERE / "processed_control.csv"
    out_b = HERE / "processed_test.csv"
    df_a.to_csv(out_a, index=False)
    df_b.to_csv(out_b, index=False)

    print("\n── Summary ───────────────────────────────────────────")
    print(f"  Days            : {meta['n_days']}")
    print(f"  Control rate    : {meta['baseline_rate']:.3%}")
    print(f"  Test rate       : {meta['treatment_rate']:.3%}")
    print(f"  Observed lift   : {meta['lift']:+.1%}")
    print(f"\n  Saved → {out_a.name}")
    print(f"  Saved → {out_b.name}")

    print("\n── Control (first 5 rows) ────────────────────────────")
    print(df_a.head().to_string(index=False))
    print("\n── Test (first 5 rows) ───────────────────────────────")
    print(df_b.head().to_string(index=False))
