"""
experiment_runner.py — Design & Run (merged planning + live simulation)

Single flow:
  1. Fill in 4 simple inputs
  2. See a plain-English preview of what to expect
  3. Click Launch — watch the experiment run day by day
  4. See the result
"""

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from simulate import TestConfig, simulate_test
from bayesian_model import (
    run_bayesian_analysis, compute_traditional_result,
    compute_time_saved, get_early_stop_day,
)
from bandit import run_thompson_sampling
from visualizations import plot_traffic_allocation
from result_widgets import render_ts_kpis

# ── Design tokens ──────────────────────────────────────────────────────────────
C_A      = "#58a6ff"
C_B      = "#3fb950"
C_STOP   = "#d29922"
C_DANGER = "#f85149"
C_BG     = "rgba(0,0,0,0)"
C_GRID   = "rgba(33,38,45,0.8)"
C_TEXT   = "#e6edf3"
C_MUTED  = "#7d8590"
FONT     = "Inter, -apple-system, sans-serif"
FONT_M   = "JetBrains Mono, monospace"


# ── Live chart ─────────────────────────────────────────────────────────────────

def _live_chart(days, probs, confidence_thresh, stop_day, exp_name, n_days=30):
    fig = go.Figure()

    fig.add_hrect(y0=confidence_thresh, y1=1.0, fillcolor=C_B,      opacity=0.06, line_width=0)
    fig.add_hrect(y0=0.0, y1=1 - confidence_thresh, fillcolor=C_DANGER, opacity=0.06, line_width=0)

    fig.add_hline(y=confidence_thresh, line_dash="dot", line_color=C_B, line_width=1.5,
                  annotation_text=f"  {confidence_thresh:.0%} — B wins",
                  annotation_position="top right",
                  annotation_font=dict(color=C_B, size=10))
    fig.add_hline(y=1 - confidence_thresh, line_dash="dot", line_color=C_DANGER, line_width=1.5,
                  annotation_text=f"  {1 - confidence_thresh:.0%} — A wins",
                  annotation_position="bottom right",
                  annotation_font=dict(color=C_DANGER, size=10))
    fig.add_hline(y=0.5, line_dash="dot", line_color=C_GRID, line_width=1)

    fig.add_trace(go.Scatter(
        x=days, y=probs, mode="lines",
        line=dict(color=C_B, width=2.5),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.07)",
        name="P(B > A)",
        hovertemplate="Day %{x}<br>P(B > A) = %{y:.1%}<extra></extra>",
    ))

    # Moving dot while running
    if not stop_day:
        fig.add_trace(go.Scatter(
            x=[days[-1]], y=[probs[-1]], mode="markers",
            marker=dict(color=C_B, size=10, symbol="circle",
                        line=dict(color="#0d1117", width=2)),
            showlegend=False,
        ))

    # Winner marker
    if stop_day and days[-1] >= stop_day:
        idx = days.index(stop_day)
        fig.add_vline(x=stop_day, line_dash="dash", line_color=C_STOP, line_width=1.5)
        fig.add_trace(go.Scatter(
            x=[stop_day], y=[probs[idx]],
            mode="markers+text",
            marker=dict(color=C_STOP, size=12, symbol="diamond",
                        line=dict(color="#0d1117", width=2)),
            text=[f"  Winner · Day {stop_day}"],
            textposition="middle right",
            textfont=dict(color=C_STOP, size=10, family=FONT),
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        font=dict(family=FONT, color=C_TEXT, size=12),
        margin=dict(l=48, r=24, t=52, b=44),
        title=dict(
            text=f"<b>{exp_name}</b>  ·  Day {days[-1]} / {n_days}",
            x=0, xanchor="left", font=dict(size=13, color=C_TEXT),
        ),
        xaxis=dict(range=[1, n_days], title="Day", gridcolor=C_GRID,
                   zerolinecolor=C_GRID, linecolor=C_GRID,
                   tickfont=dict(size=11, color=C_MUTED),
                   title_font=dict(size=11, color=C_MUTED)),
        yaxis=dict(range=[0, 1], tickformat=".0%", title="Confidence B beats A",
                   gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor="rgba(0,0,0,0)",
                   tickfont=dict(size=11, color=C_MUTED),
                   title_font=dict(size=11, color=C_MUTED)),
        showlegend=False,
        height=340,
    )
    return fig


# ── Main render ────────────────────────────────────────────────────────────────

def render_runner_tab():

    st.markdown(
        "<h2 style='margin-bottom:4px'>Run an Experiment</h2>"
        "<p style='color:#7d8590;font-size:13px;margin-top:0;margin-bottom:16px'>"
        "Describe your test, then watch it run day by day."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='callout callout-info'>"
        "<b>How this works</b> — You provide your current conversion rate and the improvement you're hoping for. "
        "The engine simulates the experiment day by day using a Bayesian model, updating its confidence each day as data arrives. "
        "It declares a winner as soon as confidence crosses your threshold — rather than waiting a fixed number of days. "
        "Traffic is also dynamically routed toward the winning variant using Thompson Sampling."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Inputs ─────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Your experiment</div>", unsafe_allow_html=True)

    col_name, col_metric = st.columns([2, 1])
    with col_name:
        exp_name = st.text_input(
            "What are you testing?",
            placeholder="e.g. New checkout button colour",
        )
    with col_metric:
        metric = st.selectbox(
            "What are you measuring?",
            ["conversion", "ctr", "revenue"],
            format_func=lambda x: {
                "conversion": "Conversions",
                "ctr":        "Clicks",
                "revenue":    "Revenue",
            }[x],
        )

    col_a, col_b, col_traffic, col_days, col_conf = st.columns(5)
    with col_a:
        baseline_pct = st.number_input(
            "Your current rate (%)",
            min_value=0.1, max_value=80.0, value=5.0, step=0.1, format="%.1f",
            help="What percentage of users currently convert / click.",
        )
    with col_b:
        mde_pp = st.number_input(
            "Improvement you're looking for (pp)",
            min_value=0.1, max_value=20.0, value=1.0, step=0.1, format="%.1f",
            help="How much better does B need to be for it to matter? E.g. +1pp means 5% → 6%.",
        )
    with col_traffic:
        daily_visitors = st.number_input(
            "Users per variant",
            min_value=50, max_value=500_000, value=5000, step=50,
            help="Total number of users assigned to each variant over the experiment.",
        )
    with col_days:
        n_days = st.number_input(
            "Max duration (days)",
            min_value=7, max_value=90, value=30, step=1,
            help="Stop the experiment after this many days if no winner is found.",
        )
    with col_conf:
        confidence_thresh = st.slider(
            "Confidence threshold",
            min_value=0.80, max_value=0.99, value=0.95, step=0.01, format="%.2f",
            help="Declare a winner when P(B > A) reaches this level.",
        )

    cost_per_day = st.number_input(
        "Estimated daily cost of running experiment ($)",
        min_value=0, max_value=1_000_000, value=5_000, step=500,
        help="Engineering + infra cost per day. Used to calculate total savings.",
    )

    # Revenue-specific AOV inputs — only shown when metric is revenue
    revenue_mean = 25.0
    treatment_revenue_mean = None
    revenue_std = 10.0
    if metric == "revenue":
        st.markdown(
            "<div class='callout callout-info'>"
            "<b>Revenue settings</b> — The conversion rate inputs above control <i>who</i> spends. "
            "Use these to model a difference in <i>how much</i> each spender pays (AOV uplift)."
            "</div>",
            unsafe_allow_html=True,
        )
        rv1, rv2, rv3 = st.columns(3)
        with rv1:
            revenue_mean = st.number_input(
                "Control AOV ($)", min_value=1.0, max_value=10_000.0,
                value=25.0, step=1.0,
                help="Average order value for the control variant.",
            )
        with rv2:
            treatment_revenue_mean = st.number_input(
                "Treatment AOV ($)", min_value=1.0, max_value=10_000.0,
                value=25.0, step=1.0,
                help="Average order value for the treatment. Set higher than control to model AOV uplift.",
            )
        with rv3:
            revenue_std = st.number_input(
                "AOV std dev ($)", min_value=1.0, max_value=10_000.0,
                value=10.0, step=1.0,
                help="Spread of spend per spender (same for both variants).",
            )
        if treatment_revenue_mean == revenue_mean:
            treatment_revenue_mean = None   # no AOV difference — keep model simple

    baseline_rate  = baseline_pct / 100
    mde_absolute   = mde_pp / 100
    treatment_rate = min(baseline_rate + mde_absolute, 0.9999)

    # Guards
    if not exp_name.strip():
        st.markdown(
            "<div class='callout callout-info'>Give your experiment a name to continue.</div>",
            unsafe_allow_html=True,
        )
        return

    if baseline_rate + mde_absolute > 0.9999:
        st.markdown(
            "<div class='callout callout-warn'>That improvement would push the rate above 100%. Try a smaller number.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    launch = st.button("Launch Experiment", type="primary")

    if not launch:
        return

    # ── Animation ──────────────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Running</div>", unsafe_allow_html=True)

    cfg = TestConfig(
        name=exp_name.strip(),
        metric=metric,
        baseline_rate=baseline_rate,
        treatment_rate=treatment_rate,
        n_users=daily_visitors,
        revenue_mean=revenue_mean,
        revenue_std=revenue_std,
        treatment_revenue_mean=treatment_revenue_mean,
        n_days=int(n_days),
        seed=None,
    )
    df_a, df_b = simulate_test(cfg)
    all_results, state_a, state_b = run_bayesian_analysis(df_a, df_b, metric, confidence_thresh)
    stop_day = get_early_stop_day(all_results)
    max_day  = stop_day if stop_day else int(n_days)

    chart_slot  = st.empty()
    status_slot = st.empty()
    days_seen, probs_seen = [], []

    for result in all_results[:max_day]:
        days_seen.append(result.day)
        probs_seen.append(result.prob_b_beats_a)
        is_final = result.day == max_day

        fig = _live_chart(
            days_seen, probs_seen, confidence_thresh,
            stop_day if is_final else None,
            exp_name.strip(),
            n_days=int(n_days),
        )
        chart_slot.plotly_chart(fig, use_container_width=True)

        if is_final and stop_day:
            winner = "Treatment B" if result.prob_b_beats_a >= confidence_thresh else "Control A"
            status_slot.markdown(
                f"<p style='font-size:12px;color:#7d8590;margin:4px 0'>"
                f"Day <span style='color:#e6edf3;font-family:{FONT_M}'>{result.day}</span>"
                f" · Confidence = <span style='color:#e6edf3;font-family:{FONT_M}'>{result.prob_b_beats_a:.1%}</span>"
                f" · <span style='color:#3fb950;font-weight:600'>Winner: {winner}</span>"
                f"</p>",
                unsafe_allow_html=True,
            )
            time.sleep(0.8)
        else:
            status_slot.markdown(
                f"<p style='font-size:12px;color:#7d8590;margin:4px 0'>"
                f"Day <span style='color:#e6edf3;font-family:{FONT_M}'>{result.day}</span>"
                f" · Confidence = <span style='color:#e6edf3;font-family:{FONT_M}'>{result.prob_b_beats_a:.1%}</span>"
                f" · monitoring…"
                f"</p>",
                unsafe_allow_html=True,
            )
            time.sleep(0.3)

    # ── Result ─────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Result</div>", unsafe_allow_html=True)

    final      = all_results[max_day - 1]
    savings    = compute_time_saved(all_results, total_days=int(n_days), cost_per_day=float(cost_per_day))
    final_prob = final.prob_b_beats_a
    final_lift = final.expected_lift

    if stop_day and final_prob >= confidence_thresh:
        banner_cls, winner_label = "b-wins", "Treatment B wins"
        detail = (
            f"We're <b>{final_prob:.0%} confident</b> that B outperforms A. "
            f"Winner declared on <b>day {stop_day}</b> — "
            f"<b>{savings['days_saved']} days</b> faster than waiting the full 30 days. "
            f"Expected lift: <b>{final_lift:+.1%}</b>."
        )
    elif stop_day:
        banner_cls, winner_label = "a-wins", "Control A wins"
        detail = (
            f"Treatment B underperformed. We're <b>{1 - final_prob:.0%} confident</b> "
            f"that A is better. Declared on day {stop_day}. Stick with the control."
        )
    else:
        banner_cls, winner_label = "null-result", f"No clear winner after {int(n_days)} days"
        detail = (
            f"Neither variant reached <b>{confidence_thresh:.0%} confidence</b>. "
            f"Final confidence: <b>{final_prob:.1%}</b>. "
            f"The effect may be too small to detect at this traffic level — "
            f"try increasing users per variant, extending the duration, or looking for a larger improvement."
        )

    st.markdown(
        f"<div class='result-banner {banner_cls}'>"
        f"  <div class='result-winner'>{winner_label}</div>"
        f"  <div class='result-detail'>{detail}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Concluded on", f"Day {stop_day}" if stop_day else "Day 30")
    with r2:
        st.metric("Confidence", f"{final_prob:.1%}", "B beats A")
    with r3:
        st.metric("Lift", f"{final_lift:+.1%}", "expected improvement")
    with r4:
        st.metric("Days saved", f"{savings['days_saved']}d",
                  f"~${savings['cost_saved_usd']:,.0f} saved")

    # ── Thompson Sampling ──────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Dynamic Traffic Routing</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7d8590;font-size:13px;margin-bottom:16px'>"
        "Instead of a fixed 50/50 split, Thompson Sampling shifts traffic toward the winning "
        "variant each day — reducing lost conversions while the test runs."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Running Thompson Sampling..."):
        ts_df = run_thompson_sampling(cfg)

    render_ts_kpis(ts_df, label_b="Treatment B")

    st.plotly_chart(plot_traffic_allocation(ts_df, cfg.name), use_container_width=True)
