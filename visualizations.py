"""
visualizations.py — Plotly chart builders for the Streamlit dashboard.
Dark professional aesthetic aligned with the GitHub/Linear design system.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Optional
import pandas as pd

from bayesian_model import DayResult, BayesianState

# ── Design tokens ─────────────────────────────────────────────────────────────
C_A       = "#58a6ff"                  # Control A — blue
C_B       = "#3fb950"                  # Treatment B — green
C_STOP    = "#d29922"                  # Early stop marker — amber
C_DANGER  = "#f85149"                  # A wins / danger — red
C_BG      = "rgba(0,0,0,0)"           # Transparent (Streamlit handles bg)
C_GRID    = "rgba(33,38,45,0.8)"      # Subtle gridlines
C_TEXT    = "#e6edf3"                  # Primary text
C_MUTED   = "#7d8590"                  # Secondary text
FONT      = "Inter, -apple-system, sans-serif"
FONT_MONO = "JetBrains Mono, monospace"

_BASE = dict(
    paper_bgcolor=C_BG,
    plot_bgcolor=C_BG,
    font=dict(family=FONT, color=C_TEXT, size=12),
    margin=dict(l=48, r=24, t=52, b=44),
    xaxis=dict(
        gridcolor=C_GRID, zerolinecolor=C_GRID,
        linecolor=C_GRID, tickfont=dict(size=11, color=C_MUTED),
        title_font=dict(size=11, color=C_MUTED),
    ),
    yaxis=dict(
        gridcolor=C_GRID, zerolinecolor=C_GRID,
        linecolor="rgba(0,0,0,0)", tickfont=dict(size=11, color=C_MUTED),
        title_font=dict(size=11, color=C_MUTED),
    ),
    legend=dict(
        bgcolor="rgba(22,27,34,0.7)",
        bordercolor="rgba(33,38,45,0.8)",
        borderwidth=1,
        font=dict(size=11, color=C_MUTED),
    ),
)


def _layout(**overrides) -> dict:
    layout = dict(**_BASE)
    layout.update(overrides)
    return layout


def _title(text: str) -> dict:
    return dict(text=text, x=0.0, xanchor="left", font=dict(size=13, color=C_TEXT, weight=600))


# ─────────────────────────────────────────────────────────────────────────────
# 1 — P(B > A) confidence trend
# ─────────────────────────────────────────────────────────────────────────────
def plot_prob_over_time(
    day_results: List[DayResult],
    test_name: str,
    confidence_thresh: float = 0.95,
) -> go.Figure:
    days  = [r.day for r in day_results]
    probs = [r.prob_b_beats_a for r in day_results]
    stop_day = next((r.day for r in day_results if r.early_stop), None)

    fig = go.Figure()

    # Decision zones
    fig.add_hrect(y0=confidence_thresh, y1=1.0, fillcolor=C_B,      opacity=0.06, line_width=0)
    fig.add_hrect(y0=0.0, y1=1 - confidence_thresh, fillcolor=C_DANGER, opacity=0.06, line_width=0)

    # Threshold lines
    fig.add_hline(
        y=confidence_thresh, line_dash="dot", line_color=C_B, line_width=1.5,
        annotation_text=f"  {confidence_thresh:.0%} — B wins",
        annotation_position="top right",
        annotation_font=dict(color=C_B, size=10),
    )
    fig.add_hline(
        y=1 - confidence_thresh, line_dash="dot", line_color=C_DANGER, line_width=1.5,
        annotation_text=f"  {1 - confidence_thresh:.0%} — A wins",
        annotation_position="bottom right",
        annotation_font=dict(color=C_DANGER, size=10),
    )
    fig.add_hline(y=0.50, line_dash="dot", line_color=C_GRID, line_width=1)

    # Probability curve
    fig.add_trace(go.Scatter(
        x=days, y=probs, mode="lines",
        line=dict(color=C_B, width=2.5),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.07)",
        name="P(B > A)",
        hovertemplate="Day %{x}<br>P(B > A) = %{y:.1%}<extra></extra>",
    ))

    # Early stop marker
    if stop_day:
        sp = probs[stop_day - 1]
        fig.add_vline(x=stop_day, line_dash="dash", line_color=C_STOP, line_width=1.5)
        fig.add_trace(go.Scatter(
            x=[stop_day], y=[sp],
            mode="markers+text",
            marker=dict(color=C_STOP, size=10, symbol="diamond"),
            text=[f"  Winner · Day {stop_day}"],
            textposition="middle right",
            textfont=dict(color=C_STOP, size=10, family=FONT),
            name="Early stop",
            hovertemplate=f"Day {stop_day}: Winner declared<extra></extra>",
        ))

    fig.update_layout(**_layout(
        title=_title(f"Confidence over time — {test_name}"),
        yaxis=dict(
            tickformat=".0%", range=[0, 1],
            gridcolor=C_GRID, zerolinecolor=C_GRID,
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(size=11, color=C_MUTED),
            title_font=dict(size=11, color=C_MUTED),
            title="P(B > A)",
        ),
        xaxis_title="Day",
        showlegend=True,
        height=340,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2 — Posterior distributions (final state)
# ─────────────────────────────────────────────────────────────────────────────
def plot_posterior_distributions(
    state_a: BayesianState,
    state_b: BayesianState,
    metric: str,
    test_name: str,
) -> go.Figure:
    from scipy.stats import gaussian_kde

    N = 50_000

    if metric in ("conversion", "ctr"):
        rng       = np.random.default_rng(42)
        samples_a = rng.beta(state_a.alpha, state_a.beta, N)
        samples_b = rng.beta(state_b.alpha, state_b.beta, N)
        xlabel    = "Conversion Rate" if metric == "conversion" else "Click-Through Rate"
    else:
        # Use the state's log-normal sampler — consistent with the inference engine
        samples_a = state_a.sample_posterior(N, metric)
        samples_b = state_b.sample_posterior(N, metric)
        xlabel    = "Revenue per User ($)"

    kde_a = gaussian_kde(samples_a, bw_method=0.1)
    kde_b = gaussian_kde(samples_b, bw_method=0.1)
    lo = min(samples_a.min(), samples_b.min())
    hi = max(samples_a.max(), samples_b.max())
    x  = np.linspace(lo, hi, 500)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=kde_a(x), mode="lines",
        fill="tozeroy", fillcolor="rgba(88,166,255,0.10)",
        line=dict(color=C_A, width=2),
        name=f"Control A — μ={state_a.posterior_mean:.4f}",
        hovertemplate=f"Value=%{{x:.4f}}<extra>Control A</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=kde_b(x), mode="lines",
        fill="tozeroy", fillcolor="rgba(63,185,80,0.10)",
        line=dict(color=C_B, width=2),
        name=f"Treatment B — μ={state_b.posterior_mean:.4f}",
        hovertemplate=f"Value=%{{x:.4f}}<extra>Treatment B</extra>",
    ))

    for state, clr, lbl in [(state_a, C_A, "A"), (state_b, C_B, "B")]:
        fig.add_vline(
            x=state.posterior_mean, line_dash="dash", line_color=clr,
            line_width=1, opacity=0.7,
            annotation_text=f"  μ{lbl}",
            annotation_font=dict(color=clr, size=10),
        )

    fig.update_layout(**_layout(
        title=_title(f"Posterior distributions — {test_name}"),
        xaxis_title=xlabel,
        yaxis_title="Density",
        showlegend=True,
        height=300,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3 — Cumulative rate over time
# ─────────────────────────────────────────────────────────────────────────────
def plot_running_rates(
    df_a, df_b,
    metric: str,
    test_name: str,
    stop_day: Optional[int] = None,
) -> go.Figure:
    ylabel_map = {
        "conversion": "Conversion Rate",
        "ctr":        "Click-Through Rate",
        "revenue":    "Revenue / User ($)",
    }
    ylabel = ylabel_map[metric]
    fmt    = ".2%" if metric != "revenue" else "$.2f"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_a.day, y=df_a.rate, mode="lines",
        line=dict(color=C_A, width=2),
        name="Control A",
        hovertemplate=f"Day %{{x}}<br>{ylabel} = %{{y:{fmt}}}<extra>Control A</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_b.day, y=df_b.rate, mode="lines",
        line=dict(color=C_B, width=2),
        name="Treatment B",
        hovertemplate=f"Day %{{x}}<br>{ylabel} = %{{y:{fmt}}}<extra>Treatment B</extra>",
    ))
    if stop_day:
        fig.add_vline(
            x=stop_day, line_dash="dash", line_color=C_STOP, line_width=1.5,
            annotation_text=f"  Stop · Day {stop_day}",
            annotation_font=dict(color=C_STOP, size=10),
        )

    yaxis_extra = {"tickformat": ".1%"} if metric != "revenue" else {}
    fig.update_layout(**_layout(
        title=_title(f"Running {ylabel.lower()} — {test_name}"),
        xaxis_title="Day",
        yaxis_title=ylabel,
        showlegend=True,
        height=300,
        yaxis=dict(
            gridcolor=C_GRID, zerolinecolor=C_GRID,
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(size=11, color=C_MUTED),
            title_font=dict(size=11, color=C_MUTED),
            **yaxis_extra,
        ),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4 — Portfolio: time saved per experiment
# ─────────────────────────────────────────────────────────────────────────────
def plot_time_savings_summary(savings_data: List[dict]) -> go.Figure:
    names   = [d["name"]       for d in savings_data]
    saved   = [d["days_saved"] for d in savings_data]
    winners = [d["winner"]     for d in savings_data]
    colors  = [C_B if w == "B" else C_A if w == "A" else C_GRID for w in winners]
    avg     = np.mean(saved)

    fig = go.Figure(go.Bar(
        x=names, y=saved,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{s}d" for s in saved],
        textposition="outside",
        textfont=dict(size=10, color=C_MUTED, family=FONT_MONO),
        hovertemplate="%{x}<br>Days saved: %{y}<br>Winner: " +
                      "<br>".join([w for w in winners]) + "<extra></extra>",
    ))
    fig.add_hline(
        y=avg, line_dash="dot", line_color=C_STOP, line_width=1.5,
        annotation_text=f"  Avg {avg:.1f}d saved",
        annotation_font=dict(color=C_STOP, size=10),
    )

    fig.update_layout(**_layout(
        title=_title("Days saved vs fixed 30-day horizon — all 12 experiments"),
        xaxis_title="",
        yaxis_title="Days Saved",
        xaxis=dict(
            tickangle=-32, gridcolor=C_GRID, zerolinecolor=C_GRID,
            linecolor=C_GRID, tickfont=dict(size=10, color=C_MUTED),
        ),
        showlegend=False,
        height=360,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5 — Thompson Sampling: traffic allocation
# ─────────────────────────────────────────────────────────────────────────────
def plot_traffic_allocation(df: pd.DataFrame, test_name: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.day, y=df.alloc_b,
        stackgroup="one", fillcolor="rgba(63,185,80,0.45)",
        line=dict(color=C_B, width=1.5),
        name="Treatment B",
        hovertemplate="Day %{x}<br>B allocation = %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.day, y=df.alloc_a,
        stackgroup="one", fillcolor="rgba(88,166,255,0.35)",
        line=dict(color=C_A, width=1.5),
        name="Control A",
        hovertemplate="Day %{x}<br>A allocation = %{y:.1%}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color=C_GRID, line_width=1,
                  annotation_text="  50/50 baseline",
                  annotation_font=dict(color=C_MUTED, size=10))

    fig.update_layout(**_layout(
        title=_title(f"Dynamic traffic allocation — {test_name}"),
        yaxis=dict(
            tickformat=".0%", range=[0, 1],
            gridcolor=C_GRID, zerolinecolor=C_GRID,
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(size=11, color=C_MUTED),
            title_font=dict(size=11, color=C_MUTED),
            title="Traffic Split",
        ),
        xaxis_title="Day",
        showlegend=True,
        height=340,
    ))
    return fig
