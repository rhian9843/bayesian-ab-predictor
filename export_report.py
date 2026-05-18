"""
export_report.py — Report generation for A/B Test Predictor
Produces two downloadable outputs:
  1. CSV  — full day-by-day data with Bayesian metrics
  2. TXT  — executive summary (copy-paste ready for stakeholders)
"""

import io
import csv
import pandas as pd
from datetime import datetime
from typing import List

from bayesian_model import DayResult, BayesianState, compute_time_saved, get_early_stop_day


# ─────────────────────────────────────────────────────────────────────────────
# 1. CSV Export — full experiment data
# ─────────────────────────────────────────────────────────────────────────────
def build_csv_report(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    day_results: List[DayResult],
    test_name: str,
    metric: str,
) -> bytes:
    """
    Returns a UTF-8 encoded CSV as bytes, ready for st.download_button.
    Columns: day, users_A, rate_A, users_B, rate_B,
             P(B>A), expected_lift, winner_declared, early_stop_triggered
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # ── Header block ──
    writer.writerow(["# Bayesian A/B Test Report"])
    writer.writerow(["# Experiment", test_name])
    writer.writerow(["# Metric", metric.upper()])
    writer.writerow(["# Generated", datetime.now().strftime("%Y-%m-%d %H:%M")])
    writer.writerow([])

    # ── Column headers ──
    writer.writerow([
        "Day",
        "Users_A (Control)", "Rate_A",
        "Users_B (Treatment)", "Rate_B",
        "P(B > A)", "Expected Lift %",
        "95% CI Lower (B)", "95% CI Upper (B)",
        "Winner Declared", "Early Stop Triggered",
    ])

    # ── Data rows ──
    for row_a, row_b, dr in zip(df_a.itertuples(), df_b.itertuples(), day_results):
        rate_fmt = f"{row_a.rate:.4%}" if metric != "revenue" else f"${row_a.rate:.2f}"
        rate_b_fmt = f"{row_b.rate:.4%}" if metric != "revenue" else f"${row_b.rate:.2f}"
        writer.writerow([
            dr.day,
            int(row_a.users), rate_fmt,
            int(row_b.users), rate_b_fmt,
            f"{dr.prob_b_beats_a:.1%}",
            f"{dr.expected_lift:+.2%}",
            f"{dr.credible_interval_b[0]:.4f}",
            f"{dr.credible_interval_b[1]:.4f}",
            dr.declared_winner or "—",
            "YES" if dr.early_stop else "no",
        ])

    return output.getvalue().encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Executive Summary TXT — stakeholder-ready
# ─────────────────────────────────────────────────────────────────────────────
def build_text_summary(
    test_name: str,
    metric: str,
    day_results: List[DayResult],
    state_a: BayesianState,
    state_b: BayesianState,
    freq_result: dict,
) -> bytes:
    """
    Returns a formatted plain-text executive summary as bytes.
    Suitable for pasting into Slack, email, or Confluence.
    """
    savings   = compute_time_saved(day_results)
    stop_day  = get_early_stop_day(day_results)
    final     = day_results[-1]
    winner    = "Treatment B" if final.prob_b_beats_a >= 0.95 else (
                "Control A"  if final.prob_b_beats_a <= 0.05 else "No clear winner")
    lift_str  = f"{final.expected_lift:+.1%}"
    rate_a    = state_a.posterior_mean
    rate_b    = state_b.posterior_mean
    rate_fmt  = ".2%" if metric != "revenue" else ".2f"
    prefix    = "" if metric != "revenue" else "$"

    now = datetime.now().strftime("%B %d, %Y at %H:%M")

    lines = [
        "=" * 62,
        "  BAYESIAN A/B TEST — EXECUTIVE SUMMARY",
        "=" * 62,
        f"  Experiment : {test_name}",
        f"  Metric     : {metric.upper()}",
        f"  Generated  : {now}",
        "=" * 62,
        "",
        "── DECISION ─────────────────────────────────────────────",
        f"  🏆  Winner          : {winner}",
        f"  📊  P(B > A)        : {final.prob_b_beats_a:.1%}",
        f"  📈  Observed Lift   : {lift_str}",
        f"  🛑  Early Stop Day  : Day {stop_day}" if stop_day else "  ⏳  No early stop triggered",
        "",
        "── BAYESIAN RESULTS ─────────────────────────────────────",
        f"  Posterior mean A   : {prefix}{rate_a:{rate_fmt}}",
        f"  Posterior mean B   : {prefix}{rate_b:{rate_fmt}}",
        f"  95% Credible Int B : [{prefix}{final.credible_interval_b[0]:.4f}, {prefix}{final.credible_interval_b[1]:.4f}]",
        f"  Confidence level   : {max(final.prob_b_beats_a, 1 - final.prob_b_beats_a):.1%}",
        "",
        "── TRADITIONAL (FREQUENTIST) COMPARISON ─────────────────",
        f"  p-value            : {freq_result['p_value']:.4f}",
        f"  Significant?       : {'Yes (p < 0.05)' if freq_result['significant'] else 'No (p ≥ 0.05)'}",
        f"  Traditional days   : {savings['traditional_days']} days (full horizon)",
        "",
        "── TIME & COST SAVINGS ──────────────────────────────────",
        f"  Bayesian stop day  : Day {savings['bayesian_stop_day']}",
        f"  Days saved         : {savings['days_saved']} days",
        f"  Time reduction     : {savings['pct_saved']:.0f}% faster",
        f"  Est. cost saved    : ${savings['cost_saved_usd']:,.0f}  (@$5,000/day)",
        "",
        "── RECOMMENDATION ───────────────────────────────────────",
    ]

    # Contextual recommendation
    if final.prob_b_beats_a >= 0.95:
        lines += [
            f"  Ship Treatment B. We have {final.prob_b_beats_a:.0%} posterior confidence",
            f"  that B outperforms A on {metric.upper()} with a {lift_str} lift.",
            f"  Recommend full rollout on Day {stop_day or 30}.",
        ]
    elif final.prob_b_beats_a <= 0.05:
        lines += [
            f"  Retain Control A. Treatment B underperforms with only",
            f"  {final.prob_b_beats_a:.0%} probability of beating A.",
            "  Recommend discarding treatment variant.",
        ]
    else:
        lines += [
            "  Inconclusive. Neither variant reached the 95% confidence",
            "  threshold. Consider extending the experiment, increasing",
            "  sample size, or revisiting the hypothesis.",
        ]

    lines += [
        "",
        "=" * 62,
        "  Generated by: Bayesian A/B Test Outcome Predictor",
        "  Method: Beta-Binomial conjugate priors + Monte Carlo",
        "=" * 62,
    ]

    return "\n".join(lines).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Portfolio CSV — all 12 experiments summary
# ─────────────────────────────────────────────────────────────────────────────
def build_portfolio_csv(portfolio_rows: list) -> bytes:
    """Export the full portfolio summary table as CSV."""
    output = io.StringIO()
    df = pd.DataFrame(portfolio_rows)
    df.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")