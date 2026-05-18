"""
app.py — Streamlit Dashboard: A/B Test Outcome Predictor
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

from simulate import PRESET_SCENARIOS, simulate_test, TestConfig
from bayesian_model import (
    run_bayesian_analysis, compute_traditional_result,
    get_early_stop_day, compute_time_saved
)
from visualizations import plot_time_savings_summary, plot_traffic_allocation
from bandit import run_thompson_sampling
from export_report import build_portfolio_csv
from experiment_runner import render_runner_tab
from upload_tab import render_upload_tab
from result_widgets import (
    render_result_banner, render_statistical_kpis,
    render_analysis_charts, render_export,
)

st.set_page_config(
    page_title="A/B Predictor",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0d1117 !important;
    color: #e6edf3;
}

[data-testid="stHeader"] { display: none; }
.block-container { padding: 2rem 2rem 4rem !important; max-width: 1280px !important; }

/* ── Typography ── */
h1 { font-size: 20px !important; font-weight: 700 !important; color: #e6edf3 !important; letter-spacing: -0.02em; margin-bottom: 0 !important; }
h2 { font-size: 15px !important; font-weight: 600 !important; color: #e6edf3 !important; }
h3, h4 { font-size: 11px !important; font-weight: 600 !important; color: #484f58 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; margin-bottom: 14px !important; }

/* ── Metric Cards ── */
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 20px 14px;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover { border-color: #30363d; }
div[data-testid="metric-container"] label {
    font-size: 11px !important; font-weight: 600 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; color: #8b949e !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 24px !important; font-weight: 600 !important;
    color: #e6edf3 !important; letter-spacing: -0.02em !important; padding-top: 4px;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 11px !important; margin-top: 2px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #21262d !important;
    gap: 0 !important; padding: 0 !important; margin-bottom: 28px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #7d8590 !important; font-size: 13px !important; font-weight: 500 !important;
    padding: 10px 18px !important; border-radius: 0 !important; border: none !important;
    border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
    background: transparent !important; transition: color 0.15s !important;
}
.stTabs [aria-selected="true"] { color: #e6edf3 !important; border-bottom-color: #3fb950 !important; }

/* ── Buttons ── */
.stButton > button {
    background: #238636 !important; color: #fff !important;
    border: 1px solid rgba(240,246,252,0.1) !important; border-radius: 6px !important;
    font-size: 13px !important; font-weight: 500 !important; padding: 8px 16px !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #2ea043 !important; }
.stDownloadButton > button {
    background: #21262d !important; color: #e6edf3 !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
    font-size: 13px !important; font-weight: 500 !important;
    width: 100% !important; transition: border-color 0.15s, background 0.15s !important;
}
.stDownloadButton > button:hover { background: #30363d !important; border-color: #8b949e !important; }

/* ── Result Banner ── */
.result-banner { border-radius: 8px; padding: 18px 22px; margin-bottom: 24px; }
.result-banner.b-wins { background: rgba(63,185,80,0.07); border: 1px solid rgba(63,185,80,0.25); }
.result-banner.a-wins { background: rgba(88,166,255,0.07); border: 1px solid rgba(88,166,255,0.25); }
.result-banner.null-result { background: #161b22; border: 1px solid #21262d; }
.result-winner { font-size: 17px; font-weight: 700; color: #e6edf3; letter-spacing: -0.02em; margin-bottom: 4px; }
.result-detail { font-size: 12px; color: #7d8590; line-height: 1.6; }

/* ── Badges ── */
.badge { display: inline-flex; align-items: center; padding: 2px 10px; border-radius: 100px; font-size: 10px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; vertical-align: middle; }
.badge-b { background: rgba(63,185,80,0.12); color: #3fb950; border: 1px solid rgba(63,185,80,0.25); }
.badge-a { background: rgba(88,166,255,0.12); color: #58a6ff; border: 1px solid rgba(88,166,255,0.25); }
.badge-null { background: rgba(125,133,144,0.08); color: #7d8590; border: 1px solid #30363d; }
.badge-metric { background: rgba(210,153,34,0.10); color: #d29922; border: 1px solid rgba(210,153,34,0.25); font-size: 10px; }

/* ── Callouts ── */
.callout { border-radius: 6px; padding: 12px 16px; margin: 16px 0; font-size: 13px; line-height: 1.6; }
.callout b { color: #8b949e; }
.callout-info  { background: rgba(31,111,235,0.07); border-left: 3px solid #1f6feb; color: #7d8590; }
.callout-success { background: rgba(63,185,80,0.07); border-left: 3px solid #3fb950; color: #7d8590; }
.callout-warn  { background: rgba(210,153,34,0.07); border-left: 3px solid #d29922; color: #7d8590; }

/* ── Section labels ── */
.section-label {
    font-size: 11px; font-weight: 600; color: #484f58; text-transform: uppercase;
    letter-spacing: 0.09em; padding-bottom: 10px; border-bottom: 1px solid #21262d;
    margin-bottom: 16px; margin-top: 8px;
}
.sb-label {
    font-size: 10px; font-weight: 600; color: #484f58; text-transform: uppercase;
    letter-spacing: 0.1em; margin: 16px 0 8px; padding-top: 16px;
    border-top: 1px solid #21262d; display: block;
}

/* ── Portfolio stat blocks ── */
.stat-block { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 20px 16px; text-align: center; }
.stat-num { font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; color: #e6edf3; letter-spacing: -0.03em; line-height: 1; margin-bottom: 6px; }
.stat-label { font-size: 10px; font-weight: 600; color: #484f58; text-transform: uppercase; letter-spacing: 0.09em; }


/* ── Expander ── */
[data-testid="stExpander"] { border: 1px solid #21262d !important; border-radius: 6px !important; background: #161b22 !important; }
.streamlit-expanderHeader { font-size: 12px !important; color: #7d8590 !important; font-weight: 500 !important; }
.streamlit-expanderHeader:hover { color: #8b949e !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ── Divider ── */
hr { border-color: #21262d !important; margin: 24px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("<h1>A/B Test Outcome Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#7d8590;font-size:13px;margin-top:4px;margin-bottom:28px'>"
    "Bayesian sequential testing · early stopping · dynamic traffic routing"
    "</p>",
    unsafe_allow_html=True
)

tab_preset, tab_custom, tab_upload = st.tabs(["Preset Experiments", "Run Custom Experiment", "Upload Real Data"])

with tab_preset:

    # ── Inline controls (replaces sidebar) ────────────────────────────────────
    scenario_names = [s.name for s in PRESET_SCENARIOS]

    ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])
    with ctrl1:
        selected_name = st.selectbox("Select experiment", scenario_names, index=0)
    with ctrl2:
        confidence_thresh = st.slider("Confidence threshold", 0.80, 0.99, 0.95, 0.01, format="%.2f",
                                      help="Declare a winner when P(B > A) reaches this level")
    with ctrl3:
        cost_per_day = st.number_input("Cost per day ($)", min_value=0, max_value=1_000_000,
                                       value=5_000, step=500,
                                       help="Estimated daily cost of running the experiment (engineering + infra)")

    selected_cfg = next(s for s in PRESET_SCENARIOS if s.name == selected_name)

    with st.expander("Custom rate override"):
        ov1, ov2, ov3, ov4 = st.columns([1, 1, 1, 1])
        with ov1:
            use_custom = st.checkbox("Enable", value=False)
        with ov2:
            custom_baseline  = st.slider("Control rate (A)",   0.01, 0.40, float(selected_cfg.baseline_rate),  0.005, format="%.3f", disabled=not use_custom)
        with ov3:
            custom_treatment = st.slider("Treatment rate (B)", 0.01, 0.40, float(selected_cfg.treatment_rate), 0.005, format="%.3f", disabled=not use_custom)
        with ov4:
            custom_n         = st.slider("Users per variant",  1000, 10000, selected_cfg.n_users, 500,          disabled=not use_custom)

    with st.expander("How it works"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                "<p style='font-size:12px;color:#7d8590;line-height:1.8;margin:0'>"
                "<span style='color:#8b949e;font-weight:600'>Bayesian Sequential Testing</span><br>"
                "Beliefs update daily as data arrives. No penalty for peeking — stop as soon as you're confident enough, rather than waiting a fixed number of days."
                "</p>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                "<p style='font-size:12px;color:#7d8590;line-height:1.8;margin:0'>"
                "<span style='color:#8b949e;font-weight:600'>Beta-Binomial Model</span><br>"
                "Each variant's conversion rate is modelled as a Beta distribution. After each day, the posterior updates analytically — no simulation required for the inference step."
                "</p>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                "<p style='font-size:12px;color:#7d8590;line-height:1.8;margin:0'>"
                "<span style='color:#8b949e;font-weight:600'>Thompson Sampling</span><br>"
                "Instead of a fixed 50/50 split, traffic shifts toward the leading variant each day. More users see the better experience while the test is still running."
                "</p>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Results", "Dynamic Routing", "Portfolio"])

# ══════════════════════════════════════════════════════════════════════════════
# PRESET → Results
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    cfg = TestConfig(
        name=f"Custom: {selected_name}" if use_custom else selected_name,
        metric=selected_cfg.metric,
        baseline_rate=custom_baseline   if use_custom else selected_cfg.baseline_rate,
        treatment_rate=custom_treatment if use_custom else selected_cfg.treatment_rate,
        n_users=custom_n                if use_custom else selected_cfg.n_users,
        seed=selected_cfg.seed,
    )

    with st.spinner("Running Bayesian inference..."):
        df_a, df_b                    = simulate_test(cfg)
        day_results, state_a, state_b = run_bayesian_analysis(df_a, df_b, cfg.metric, confidence_thresh)
        freq_result                   = compute_traditional_result(df_a, df_b, cfg.metric, state_a, state_b)
        savings                       = compute_time_saved(day_results, cost_per_day=cost_per_day)
        stop_day                      = get_early_stop_day(day_results)

    final_prob = day_results[-1].prob_b_beats_a
    final_lift = day_results[-1].expected_lift

    render_result_banner(
        cfg.name, cfg.metric, stop_day, final_prob, final_lift,
        confidence_thresh, n_days=30, savings=savings,
        label_a="Control A", label_b="Treatment B",
    )
    render_statistical_kpis(
        stop_day, final_prob, final_lift, savings, n_days=30,
        freq_result=freq_result, cost_per_day=cost_per_day,
    )
    render_analysis_charts(
        day_results, state_a, state_b, df_a, df_b,
        cfg.metric, cfg.name, confidence_thresh, stop_day,
    )

    # ── Raw data ──
    with st.expander("View raw data"):
        merged = df_a.copy().rename(columns={"rate": "rate_A", "events": "events_A", "users": "users_A"})
        merged[["users_B", "events_B", "rate_B"]] = df_b[["users", "events", "rate"]].values
        merged["P(B>A)"]          = [f"{r.prob_b_beats_a:.1%}" for r in day_results]
        merged["Expected Lift"]   = [f"{r.expected_lift:+.1%}"  for r in day_results]
        merged["Winner Declared"] = [r.declared_winner or "—"   for r in day_results]
        st.dataframe(
            merged.style.format({"rate_A": ".3%", "rate_B": ".3%"}),
            use_container_width=True
        )

    render_export(
        cfg.name, cfg.metric, df_a, df_b, day_results,
        state_a, state_b, freq_result, file_prefix="ab_test",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Dynamic Routing (Thompson Sampling)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# PRESET → Dynamic Routing
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<h2 style='margin-bottom:4px'>Thompson Sampling</h2>"
        "<p style='color:#7d8590;font-size:13px;margin-top:0;margin-bottom:24px'>"
        "Dynamically routing traffic to the winning variant while the test runs — minimising regret."
        "</p>",
        unsafe_allow_html=True
    )

    with st.spinner("Running Thompson Sampling simulation..."):
        ts_df = run_thompson_sampling(cfg)

    final_day = ts_df.iloc[-1]

    st.markdown("<div class='section-label'>Simulation Results</div>", unsafe_allow_html=True)
    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        st.metric("Final Traffic to B", f"{final_day.alloc_b * 100:.0f}%", "dynamic allocation")
    with kc2:
        if cfg.metric in ("conversion", "ctr"):
            total_events = int(final_day.cum_events_a + final_day.cum_events_b)
            st.metric("Total Events (Bandit)", f"{total_events:,}", "conversions / clicks")
        else:
            total_rev = final_day.cum_events_a + final_day.cum_events_b
            st.metric("Total Revenue (Bandit)", f"${total_rev:,.0f}", "across 30 days")
    with kc3:
        lift_events = (final_day.cum_events_a + final_day.cum_events_b) - final_day.cum_baseline_events
        if cfg.metric in ("conversion", "ctr"):
            delta = f"+{int(lift_events):,}" if lift_events > 0 else f"{int(lift_events):,}"
            st.metric("Gain vs 50/50 Split", delta, "extra conversions")
        else:
            delta = f"+${lift_events:,.0f}" if lift_events > 0 else f"${lift_events:,.0f}"
            st.metric("Gain vs 50/50 Split", delta, "extra revenue")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.plotly_chart(plot_traffic_allocation(ts_df, cfg.name), use_container_width=True)

    st.markdown(
        "<div class='callout callout-info'>"
        "<b>How it works</b> — Instead of a fixed 50/50 split, Thompson Sampling samples from each "
        "variant's Bayesian posterior each day and routes traffic proportionally. More traffic flows "
        "to the likely winner, reducing lost conversions during the experiment. Traffic is bounded "
        "between 10% and 90% to preserve statistical learning."
        "</div>",
        unsafe_allow_html=True
    )

    with st.expander("View raw allocation data"):
        st.dataframe(
            ts_df[["day","users_a","users_b","events_a","events_b","alloc_a","alloc_b","prob_b_beats_a"]]
            .style.format({"alloc_a": ".1%", "alloc_b": ".1%", "prob_b_beats_a": ".1%"}),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Portfolio Summary
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# PRESET → Portfolio
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "<h2 style='margin-bottom:4px'>Experiment Portfolio</h2>"
        "<p style='color:#7d8590;font-size:13px;margin-top:0;margin-bottom:24px'>"
        "Bayesian early stopping applied across all 12 simulated experiments."
        "</p>",
        unsafe_allow_html=True
    )

    with st.spinner("Simulating all 12 experiments..."):
        from simulate import simulate_all_scenarios
        from scipy.stats import false_discovery_control
        all_scenarios  = simulate_all_scenarios()
        portfolio_rows = []
        savings_data   = []
        raw_pvalues    = []

        for cfg_s in PRESET_SCENARIOS:
            s          = all_scenarios[cfg_s.name]
            dfa, dfb   = s["control"], s["treatment"]
            dr, sa, sb = run_bayesian_analysis(dfa, dfb, cfg_s.metric)
            fr         = compute_traditional_result(dfa, dfb, cfg_s.metric)
            sv         = compute_time_saved(dr)
            stop       = get_early_stop_day(dr)
            fp         = dr[-1].prob_b_beats_a
            lft        = dr[-1].expected_lift
            winner     = "B" if fp >= 0.95 else ("A" if fp <= 0.05 else "—")
            raw_pvalues.append(fr["p_value"])

            portfolio_rows.append({
                "Experiment":    cfg_s.name,
                "Metric":        cfg_s.metric.upper(),
                "Bayesian Stop": f"Day {stop}" if stop else "No stop",
                "Days Saved":    sv["days_saved"],
                "Time Saved":    f"{sv['pct_saved']:.0f}%",
                "Cost Saved":    f"${sv['cost_saved_usd']:,.0f}",
                "P(B>A)":        f"{fp:.1%}",
                "Lift":          f"{lft:+.1%}",
                "Winner":        winner,
            })
            savings_data.append({
                "name":       cfg_s.name[:22],
                "days_saved": sv["days_saved"],
                "winner":     winner,
            })

        # Benjamini-Hochberg correction on the 12 frequentist p-values
        bh_pvalues = false_discovery_control(raw_pvalues, method="bh")
        for row, p_adj in zip(portfolio_rows, bh_pvalues):
            row["p-adj (BH)"] = f"{p_adj:.3f}"

    avg_days   = np.mean([r["days_saved"] for r in savings_data])
    total_cost = sum(int(r["Cost Saved"].replace("$", "").replace(",", "")) for r in portfolio_rows)
    b_wins     = sum(1 for r in portfolio_rows if r["Winner"] == "B")
    pct_avg    = avg_days / 30 * 100

    # ── Portfolio stat blocks ──
    st.markdown("<div class='section-label'>Portfolio Summary</div>", unsafe_allow_html=True)
    ps1, ps2, ps3, ps4 = st.columns(4)
    with ps1:
        st.markdown(
            f"<div class='stat-block'>"
            f"  <div class='stat-num'>{avg_days:.0f}d</div>"
            f"  <div class='stat-label'>Avg days saved</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with ps2:
        st.markdown(
            f"<div class='stat-block'>"
            f"  <div class='stat-num'>{pct_avg:.0f}%</div>"
            f"  <div class='stat-label'>Avg time reduction</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with ps3:
        st.markdown(
            f"<div class='stat-block'>"
            f"  <div class='stat-num'>${total_cost // 1000}K</div>"
            f"  <div class='stat-label'>Total cost saved</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with ps4:
        st.markdown(
            f"<div class='stat-block'>"
            f"  <div class='stat-num'>{b_wins}/12</div>"
            f"  <div class='stat-label'>Won by treatment B</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.plotly_chart(plot_time_savings_summary(savings_data), use_container_width=True)

    st.markdown("<div class='section-label'>All Experiments</div>", unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(portfolio_rows),
        use_container_width=True,
        column_config={
            "Days Saved": st.column_config.ProgressColumn("Days Saved", min_value=0, max_value=30),
        }
    )
    st.markdown(
        "<div class='callout callout-info'>"
        "<b>Multiple testing note</b> — "
        "<b>P(B&gt;A)</b> is a per-experiment Bayesian posterior probability; it does not inflate across simultaneous tests and requires no correction. "
        "<b>p-adj (BH)</b> applies Benjamini-Hochberg FDR control to the frequentist p-values across all 12 experiments."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='callout callout-success'>"
        f"<b>Portfolio result</b> — Across 12 experiments, Bayesian early stopping saved an average of "
        f"<b>{avg_days:.1f} days</b> per test ({pct_avg:.0f}% faster). "
        f"Total estimated savings: <b>${total_cost:,.0f}</b> at $5,000/day."
        f"</div>",
        unsafe_allow_html=True
    )

    # ── Portfolio export ──
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Export</div>", unsafe_allow_html=True)
    pe1, _, _ = st.columns(3)
    with pe1:
        st.download_button(
            "⬇  Download Portfolio CSV",
            data=build_portfolio_csv(portfolio_rows),
            file_name=f"ab_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"All {len(portfolio_rows)} experiments — winners, lift, days saved, cost saved, p-values.")


# ══════════════════════════════════════════════════════════════════════════════
# Run Custom Experiment
# ══════════════════════════════════════════════════════════════════════════════
with tab_custom:
    render_runner_tab()

# ══════════════════════════════════════════════════════════════════════════════
# Upload Real Data
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    render_upload_tab()
