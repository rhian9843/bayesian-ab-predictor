"""
simulate.py — Realistic A/B Test Data Simulation
Generates conversion, CTR, and revenue data for both variants
with configurable effect sizes and noise levels.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TestConfig:
    """Configuration for a single A/B test simulation."""
    name: str
    metric: str                  # 'conversion', 'ctr', 'revenue'
    baseline_rate: float         # Control (A) true rate
    treatment_rate: float        # Treatment (B) true rate
    n_users: int = 5000          # Total users per variant
    revenue_mean: float = 25.0   # Control AOV (mean spend per spender)
    revenue_std: float = 10.0    # AOV std dev (shared by both variants)
    treatment_revenue_mean: Optional[float] = None  # Treatment AOV; None = same as revenue_mean
    n_days: int = 30             # Experiment duration
    seed: Optional[int] = None


# Pre-built test scenarios (12 realistic scenarios for portfolio demo)
PRESET_SCENARIOS = [
    TestConfig("Checkout Button Color",    "conversion", 0.050, 0.065, 5000, seed=1),
    TestConfig("Homepage Hero CTA",        "ctr",        0.041, 0.032, 8000, seed=2),  # A wins
    TestConfig("Pricing Page Layout",      "conversion", 0.028, 0.034, 6000, seed=3),
    TestConfig("Email Subject Line A",     "ctr",        0.210, 0.238, 3000, seed=4),
    TestConfig("Free Trial Offer",         "conversion", 0.040, 0.040, 5000, seed=5),  # null result
    TestConfig("Onboarding Flow V2",       "conversion", 0.178, 0.150, 4000, seed=6),  # A wins
    TestConfig("Product Image Carousel",   "ctr",        0.085, 0.091, 7000, seed=7),
    TestConfig("Upsell Modal Timing",      "revenue",    0.120, 0.160, 4000, seed=8),  # B wins
    TestConfig("Search Result Ranking",    "ctr",        0.138, 0.120, 9000, seed=9),  # A wins
    TestConfig("Cart Abandonment Email",   "conversion", 0.018, 0.024, 6000, seed=10),
    TestConfig("Dashboard Redesign",       "revenue",    0.150, 0.110, 5000, seed=11), # A wins
    TestConfig("Mobile Nav Simplification","ctr",        0.070, 0.073, 8000, seed=12),
]


def simulate_test(config: TestConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate day-by-day cumulative data for one A/B test.
    Returns (control_df, treatment_df) with columns:
        day, users, conversions/clicks/revenue, cumulative_rate
    """
    rng = np.random.default_rng(config.seed)
    daily_users = max(config.n_users // config.n_days, 20)

    # Log-normal parameters: separate for each variant so AOV uplift can be modelled
    _aov_a = config.revenue_mean
    _aov_b = config.treatment_revenue_mean if config.treatment_revenue_mean is not None else config.revenue_mean
    _sigma_log_a = np.sqrt(np.log(1 + (config.revenue_std / _aov_a) ** 2))
    _mu_log_a    = np.log(_aov_a) - _sigma_log_a ** 2 / 2
    _sigma_log_b = np.sqrt(np.log(1 + (config.revenue_std / _aov_b) ** 2))
    _mu_log_b    = np.log(_aov_b) - _sigma_log_b ** 2 / 2

    rows_a, rows_b = [], []
    cum_users_a = cum_events_a = cum_spenders_a = 0
    cum_users_b = cum_events_b = cum_spenders_b = 0

    for day in range(1, config.n_days + 1):
        # Slight daily variation in traffic
        n_a = int(rng.normal(daily_users, daily_users * 0.1))
        n_b = int(rng.normal(daily_users, daily_users * 0.1))
        n_a, n_b = max(n_a, 5), max(n_b, 5)

        if config.metric in ("conversion", "ctr"):
            events_a   = int(rng.binomial(n_a, config.baseline_rate))
            events_b   = int(rng.binomial(n_b, config.treatment_rate))
            spenders_a = events_a   # every conversion is a "spender"
            spenders_b = events_b
        else:
            # Revenue: log-normal spend per spender (always positive, right-skewed).
            # Conversion rate drives who spends; AOV is the same for both variants.
            spenders_a = int(rng.binomial(n_a, config.baseline_rate))
            spenders_b = int(rng.binomial(n_b, config.treatment_rate))
            events_a = float(rng.lognormal(_mu_log_a, _sigma_log_a, spenders_a).sum()) if spenders_a > 0 else 0.0
            events_b = float(rng.lognormal(_mu_log_b, _sigma_log_b, spenders_b).sum()) if spenders_b > 0 else 0.0

        cum_users_a    += n_a;        cum_events_a   += events_a;   cum_spenders_a += spenders_a
        cum_users_b    += n_b;        cum_events_b   += events_b;   cum_spenders_b += spenders_b

        rate_a = cum_events_a / cum_users_a
        rate_b = cum_events_b / cum_users_b

        rows_a.append({"day": day, "users": cum_users_a, "events": cum_events_a, "rate": rate_a, "spenders": cum_spenders_a})
        rows_b.append({"day": day, "users": cum_users_b, "events": cum_events_b, "rate": rate_b, "spenders": cum_spenders_b})

    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def simulate_all_scenarios() -> dict:
    """Run all 12 preset scenarios. Returns dict keyed by scenario name."""
    results = {}
    for cfg in PRESET_SCENARIOS:
        df_a, df_b = simulate_test(cfg)
        results[cfg.name] = {"config": cfg, "control": df_a, "treatment": df_b}
    return results