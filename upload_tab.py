"""
upload_tab.py — Real Data Upload & Analysis tab
Accepts two CSVs (control + test), maps columns, runs the full Bayesian engine.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import build_engine_df, fix_missing
from bayesian_model import (
    run_bayesian_analysis, compute_traditional_result,
    get_early_stop_day, compute_time_saved,
    BayesianState, N_SAMPLES,
)
from visualizations import plot_traffic_allocation
from result_widgets import (
    render_result_banner, render_statistical_kpis,
    render_analysis_charts, render_ts_kpis, render_export,
)

FONT_M = "JetBrains Mono, monospace"


# ── True Thompson Sampling replay ──────────────────────────────────────────────

def replay_thompson_sampling(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Thompson Sampling counterfactual driven entirely by real data.

    Each day:
      1. Compute allocation from posteriors built on data seen so far.
      2. Scale the real observed daily events by that allocation —
         if the bandit sent 70% to B, B gets 70% of that day's total users
         at the same rate we actually observed.
      3. Update posteriors with what the bandit would have seen.

    The 50/50 baseline uses the actual cumulative events from the real data
    (which genuinely were split ~50/50).
    """
    state_a = BayesianState()
    state_b = BayesianState()

    rows = []
    cum_users_a = cum_users_b = 0
    cum_events_a = cum_events_b = 0
    prev_users_a = prev_users_b = 0
    prev_events_a = prev_events_b = 0
    prev_spenders_a = prev_spenders_b = 0

    for i, (row_a, row_b) in enumerate(zip(df_a.itertuples(), df_b.itertuples())):
        # ── Real daily increments ──
        inc_users_a    = row_a.users    - prev_users_a
        inc_users_b    = row_b.users    - prev_users_b
        inc_events_a   = row_a.events   - prev_events_a
        inc_events_b   = row_b.events   - prev_events_b
        inc_spenders_a = row_a.spenders - prev_spenders_a
        inc_spenders_b = row_b.spenders - prev_spenders_b
        total_users    = inc_users_a + inc_users_b

        # ── Allocation from posteriors ──
        if i == 0:
            alloc_b  = 0.5
            prob_b   = 0.5
        else:
            sa = state_a.sample_posterior(N_SAMPLES, metric)
            sb = state_b.sample_posterior(N_SAMPLES, metric)
            prob_b  = float(np.mean(sb > sa))
            alloc_b = max(0.10, min(0.90, prob_b))
        alloc_a = 1.0 - alloc_b

        # ── Scale real events by allocation ──
        # Use observed daily rate as the best estimate of the true rate.
        rate_a = inc_events_a / max(inc_users_a, 1)
        rate_b = inc_events_b / max(inc_users_b, 1)

        bandit_n_b = int(total_users * alloc_b)
        bandit_n_a = total_users - bandit_n_b
        bandit_events_a = bandit_n_a * rate_a
        bandit_events_b = bandit_n_b * rate_b

        # Scale spenders proportionally for revenue posterior
        spender_rate_a = inc_spenders_a / max(inc_users_a, 1)
        spender_rate_b = inc_spenders_b / max(inc_users_b, 1)
        bandit_spenders_a = int(bandit_n_a * spender_rate_a)
        bandit_spenders_b = int(bandit_n_b * spender_rate_b)

        # ── Update posteriors with what bandit would have seen ──
        if metric in ("conversion", "ctr"):
            state_a.update_binary(int(bandit_events_a), bandit_n_a)
            state_b.update_binary(int(bandit_events_b), bandit_n_b)
        else:
            state_a.update_revenue(bandit_events_a, bandit_spenders_a)
            state_b.update_revenue(bandit_events_b, bandit_spenders_b)

        cum_users_a  += bandit_n_a;      cum_users_b  += bandit_n_b
        cum_events_a += bandit_events_a; cum_events_b += bandit_events_b

        rows.append({
            "day":                  row_a.day,
            "alloc_a":              alloc_a,
            "alloc_b":              alloc_b,
            "users_a":              bandit_n_a,
            "users_b":              bandit_n_b,
            "cum_users_a":          cum_users_a,
            "cum_users_b":          cum_users_b,
            "events_a":             bandit_events_a,
            "events_b":             bandit_events_b,
            "cum_events_a":         cum_events_a,
            "cum_events_b":         cum_events_b,
            "prob_b_beats_a":       prob_b,
            # Baseline = actual observed cumulative events (they were 50/50)
            "cum_baseline_events":  row_a.events + row_b.events,
        })

        prev_users_a, prev_events_a, prev_spenders_a = row_a.users, row_a.events, row_a.spenders
        prev_users_b, prev_events_b, prev_spenders_b = row_b.users, row_b.events, row_b.spenders

    return pd.DataFrame(rows)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_uploaded(file) -> pd.DataFrame:
    """Read an uploaded file object into a DataFrame, auto-detecting separator."""
    raw = file.read().decode("utf-8")
    # Auto-detect separator
    first_line = raw.split("\n")[0]
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    df = pd.read_csv(io.StringIO(raw), sep=sep)
    # Parse date column if present
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            df = df.sort_values(col).reset_index(drop=True)
            break
    return df


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns suitable for use as numerator/denominator."""
    skip = {"spend", "date", "name", "campaign"}
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not any(s in c.lower() for s in skip)
    ]


# ── Main render ────────────────────────────────────────────────────────────────

def render_upload_tab():

    st.markdown(
        "<h2 style='margin-bottom:4px'>Analyse Real Data</h2>"
        "<p style='color:#7d8590;font-size:13px;margin-top:0;margin-bottom:24px'>"
        "Upload your own A/B test CSVs — one row per day, one file per variant. "
        "The Bayesian engine runs on your data instead of a simulation."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── File upload ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Your data</div>", unsafe_allow_html=True)

    up1, up2 = st.columns(2)
    with up1:
        control_file = st.file_uploader("Control group CSV", type=["csv", "txt"],
                                        help="One row per day for the control variant.")
    with up2:
        test_file = st.file_uploader("Test group CSV", type=["csv", "txt"],
                                     help="One row per day for the treatment variant.")

    if not control_file or not test_file:
        st.markdown(
            "<div class='callout callout-info'>"
            "<b>Expected format</b> — one row per day, with columns for users/reach "
            "(denominator) and events/conversions (numerator). Comma or semicolon "
            "separated. A Date column is optional but will be used for sorting if present."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Parse files ────────────────────────────────────────────────────────────
    try:
        raw_a = _read_uploaded(control_file)
        raw_b = _read_uploaded(test_file)
    except Exception as e:
        st.error(f"Could not read files: {e}")
        return

    if len(raw_a) != len(raw_b):
        st.markdown(
            f"<div class='callout callout-warn'>"
            f"Control has {len(raw_a)} rows, test has {len(raw_b)}. "
            f"Both files must have the same number of days."
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Column mapping ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Column mapping</div>", unsafe_allow_html=True)

    cols = _numeric_cols(raw_a)
    if len(cols) < 2:
        st.error("Need at least two numeric columns (one for users, one for events).")
        return

    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        numerator = st.selectbox(
            "Events column (what you're counting)",
            cols,
            index=cols.index("# of Purchase") if "# of Purchase" in cols else 0,
            help="Conversions, clicks, purchases, etc.",
        )
    with cm2:
        denominator = st.selectbox(
            "Users column (who was exposed)",
            cols,
            index=cols.index("Reach") if "Reach" in cols else min(1, len(cols) - 1),
            help="Reach, impressions, sessions, etc.",
        )
    with cm3:
        metric = st.selectbox(
            "Metric type",
            ["conversion", "ctr", "revenue"],
            format_func=lambda x: {"conversion": "Conversion rate",
                                    "ctr": "Click-through rate",
                                    "revenue": "Revenue"}[x],
            help="Tells the Bayesian model which likelihood to use.",
        )

    if numerator == denominator:
        st.markdown(
            "<div class='callout callout-warn'>Events and users columns must be different.</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Settings ───────────────────────────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        exp_name = st.text_input("Experiment name", value="Real Data Analysis")
    with s2:
        confidence_thresh = st.slider("Confidence threshold", 0.80, 0.99, 0.95, 0.01,
                                      format="%.2f")
    with s3:
        cost_per_day = st.number_input("Cost per day ($)", min_value=0,
                                       max_value=1_000_000, value=5_000, step=500)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    analyze = st.button("Analyze", type="primary")
    if not analyze:
        return

    # ── Preprocess ─────────────────────────────────────────────────────────────
    missing_lines = []

    import io as _io
    import sys

    # Capture report_missing output to show in the UI
    for label, df in [("Control", raw_a), ("Test", raw_b)]:
        missing = df[_numeric_cols(df)].isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing_lines.append(f"<b>{label}</b>: interpolated {missing.sum()} missing value(s)")

    raw_a = fix_missing(raw_a)
    raw_b = fix_missing(raw_b)

    if missing_lines:
        st.markdown(
            "<div class='callout callout-warn'>"
            + " · ".join(missing_lines) +
            " — gaps filled using linear interpolation between adjacent days."
            "</div>",
            unsafe_allow_html=True,
        )

    try:
        df_a = build_engine_df(raw_a, numerator, denominator)
        df_b = build_engine_df(raw_b, numerator, denominator)
    except Exception as e:
        st.error(f"Column mapping failed: {e}")
        return

    n_days = len(df_a)

    # ── Bayesian analysis ──────────────────────────────────────────────────────
    with st.spinner("Running Bayesian inference on real data..."):
        day_results, state_a, state_b = run_bayesian_analysis(df_a, df_b, metric, confidence_thresh)
        freq_result = compute_traditional_result(df_a, df_b, metric, state_a, state_b)
        savings     = compute_time_saved(day_results, total_days=n_days,
                                         cost_per_day=float(cost_per_day))
        stop_day    = get_early_stop_day(day_results)

    final_prob = day_results[-1].prob_b_beats_a
    final_lift = day_results[-1].expected_lift

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Result</div>", unsafe_allow_html=True)

    render_result_banner(
        exp_name, metric, stop_day, final_prob, final_lift,
        confidence_thresh, n_days=n_days, savings=savings,
        label_a="Control", label_b="Test",
    )
    render_statistical_kpis(
        stop_day, final_prob, final_lift, savings, n_days=n_days,
        freq_result=freq_result, cost_per_day=float(cost_per_day),
        label_a="Control", label_b="Test",
    )
    render_analysis_charts(
        day_results, state_a, state_b, df_a, df_b,
        metric, exp_name, confidence_thresh, stop_day,
    )

    # ── Raw data ───────────────────────────────────────────────────────────────
    with st.expander("View preprocessed data"):
        merged = df_a.copy().rename(columns={"rate": "rate_control", "events": "events_control",
                                              "users": "users_control", "spenders": "spenders_control"})
        merged[["users_test", "events_test", "rate_test"]] = df_b[["users", "events", "rate"]].values
        merged["P(Test>Control)"] = [f"{r.prob_b_beats_a:.1%}" for r in day_results]
        merged["Lift"]            = [f"{r.expected_lift:+.1%}" for r in day_results]
        st.dataframe(merged.style.format({"rate_control": ".4%", "rate_test": ".4%"}),
                     use_container_width=True)

    # ── Thompson Sampling counterfactual ───────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Thompson Sampling — Counterfactual</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7d8590;font-size:13px;margin-bottom:16px'>"
        "What would have happened if traffic had been dynamically routed instead of split 50/50? "
        "Each day's allocation is decided by posteriors built from your real data. "
        "Events are scaled by that allocation using the real observed daily rates."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Replaying Thompson Sampling on real data..."):
        ts_df = replay_thompson_sampling(df_a, df_b, metric)

    render_ts_kpis(ts_df, label_b="Test")

    st.plotly_chart(plot_traffic_allocation(ts_df, exp_name), use_container_width=True)

    render_export(
        exp_name, metric, df_a, df_b, day_results,
        state_a, state_b, freq_result, file_prefix="ab_real",
    )
