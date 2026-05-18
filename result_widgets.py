"""
result_widgets.py — Shared Streamlit render functions used across tabs.

Extracted from app.py (preset Results tab) and upload_tab.py to eliminate
copy-paste duplication in: result banner, KPI rows, analysis charts,
Thompson Sampling KPIs, and export section.
"""

import streamlit as st
from datetime import datetime

from visualizations import (
    plot_prob_over_time,
    plot_posterior_distributions,
    plot_running_rates,
)
from export_report import build_csv_report, build_text_summary


def render_result_banner(
    exp_name: str,
    metric: str,
    stop_day,
    final_prob: float,
    final_lift: float,
    confidence_thresh: float,
    n_days: int,
    savings: dict,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Badge header row + result banner HTML block."""
    if stop_day and final_prob >= confidence_thresh:
        banner_cls   = "b-wins"
        winner_label = f"{label_b} wins"
        badge_cls    = "badge-b"
        banner_detail = (
            f"Declared on <b>day {stop_day}</b> — "
            f"<b>{savings['days_saved']} days</b> faster than a fixed {n_days}-day horizon. "
            f"P({label_b}&gt;{label_a}) = <b>{final_prob:.1%}</b> · Lift <b>{final_lift:+.1%}</b>"
        )
    elif stop_day and final_prob <= (1 - confidence_thresh):
        banner_cls   = "a-wins"
        winner_label = f"{label_a} wins"
        badge_cls    = "badge-a"
        banner_detail = (
            f"Declared on <b>day {stop_day}</b>. "
            f"{label_b} underperforms — P({label_b}&gt;{label_a}) = <b>{final_prob:.1%}</b>. "
            f"Recommend retaining the control."
        )
    else:
        banner_cls   = "null-result"
        winner_label = "No clear winner"
        badge_cls    = "badge-null"
        banner_detail = (
            f"Posterior never exceeded {confidence_thresh:.0%} confidence after {n_days} days. "
            f"Final P({label_b}&gt;{label_a}) = <b>{final_prob:.1%}</b>."
        )

    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:16px'>"
        f"  <span style='font-size:15px;font-weight:600;color:#e6edf3'>{exp_name}</span>"
        f"  <span class='badge {badge_cls}'>{winner_label}</span>"
        f"  <span class='badge badge-metric'>{metric.upper()}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='result-banner {banner_cls}'>"
        f"  <div class='result-winner'>{winner_label}</div>"
        f"  <div class='result-detail'>{banner_detail}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_statistical_kpis(
    stop_day,
    final_prob: float,
    final_lift: float,
    savings: dict,
    n_days: int,
    freq_result: dict,
    cost_per_day: float = 5_000,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Two rows of 3 metric cards: Statistical Results then Business Impact."""
    st.markdown("<div class='section-label'>Statistical Results</div>", unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(
            "Decision",
            f"Day {stop_day}" if stop_day else "No stop",
            f"−{savings['days_saved']}d vs {n_days}-day horizon" if stop_day
            else f"Ran full {n_days} days",
        )
    with k2:
        st.metric(
            f"Confidence  P({label_b} > {label_a})",
            f"{final_prob:.1%}",
            "posterior probability",
        )
    with k3:
        st.metric("Expected Lift", f"{final_lift:+.1%}", "posterior mean")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Business Impact</div>", unsafe_allow_html=True)
    k4, k5, k6 = st.columns(3)
    with k4:
        st.metric(
            "Time Saved",
            f"{savings['pct_saved']:.0f}% faster",
            f"{savings['days_saved']}d vs fixed horizon",
        )
    with k5:
        cost_label = f"@${cost_per_day:,.0f}/day"
        st.metric("Est. Cost Saved", f"${savings['cost_saved_usd']:,.0f}", cost_label)
    with k6:
        sig_str = "significant" if freq_result["significant"] else "not significant"
        st.metric("p-value (frequentist)", f"{freq_result['p_value']:.3f}", sig_str)


def render_analysis_charts(
    day_results,
    state_a,
    state_b,
    df_a,
    df_b,
    metric: str,
    exp_name: str,
    confidence_thresh: float,
    stop_day,
) -> None:
    """Full-width probability chart + side-by-side posteriors and running rates."""
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Analysis</div>", unsafe_allow_html=True)

    st.plotly_chart(
        plot_prob_over_time(day_results, exp_name, confidence_thresh),
        use_container_width=True,
    )

    col_post, col_run = st.columns(2)
    with col_post:
        st.plotly_chart(
            plot_posterior_distributions(state_a, state_b, metric, exp_name),
            use_container_width=True,
        )
    with col_run:
        st.plotly_chart(
            plot_running_rates(df_a, df_b, metric, exp_name, stop_day),
            use_container_width=True,
        )


def render_ts_kpis(ts_df, label_b: str = "B") -> None:
    """Three KPI cards: final allocation to B, total bandit events, gain vs 50/50."""
    final_ts = ts_df.iloc[-1]
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.metric(
            f"Final traffic to {label_b}",
            f"{final_ts.alloc_b * 100:.0f}%",
            "dynamic allocation",
        )
    with tc2:
        total_events = int(final_ts.cum_events_a + final_ts.cum_events_b)
        st.metric("Total events (bandit)", f"{total_events:,}")
    with tc3:
        gain = (final_ts.cum_events_a + final_ts.cum_events_b) - final_ts.cum_baseline_events
        st.metric(
            "Gain vs 50/50",
            f"+{int(gain):,}" if gain > 0 else f"{int(gain):,}",
            "extra conversions captured",
        )


def render_export(
    exp_name: str,
    metric: str,
    df_a,
    df_b,
    day_results,
    state_a,
    state_b,
    freq_result: dict,
    file_prefix: str = "ab",
) -> None:
    """Download buttons for CSV report and text summary, plus preview expander."""
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Export</div>", unsafe_allow_html=True)

    safe_name = exp_name.replace(" ", "_").replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d")
    txt_bytes = build_text_summary(exp_name, metric, day_results, state_a, state_b, freq_result)

    ex1, ex2, _ = st.columns([1, 1, 1])
    with ex1:
        st.download_button(
            "⬇  Download CSV — Full Data",
            data=build_csv_report(df_a, df_b, day_results, exp_name, metric),
            file_name=f"{file_prefix}_{safe_name}_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Day-by-day Bayesian metrics, posteriors & credible intervals")
    with ex2:
        st.download_button(
            "⬇  Download TXT — Exec Summary",
            data=txt_bytes,
            file_name=f"ab_summary_{safe_name}_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.caption("Stakeholder-ready summary with recommendation")

    with st.expander("Preview executive summary"):
        st.code(txt_bytes.decode("utf-8"), language=None)
