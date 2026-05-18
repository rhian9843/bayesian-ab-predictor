"""
bayesian_model.py — Bayesian A/B Test Engine
Uses conjugate priors (Beta-Binomial for rates, Normal-Normal for revenue)
for fast, GPU-free inference. Updates beliefs as new data arrives daily.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
from scipy import stats


# ── Constants ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.95   # Declare winner when P(B > A) ≥ 95% or ≤ 5%
# 7-day burn-in: on day 1 a lucky 3-conversion streak can hit 99% confidence
# with fewer than 10 observations. One full week ensures the posterior has
# enough data to be meaningful before we act on it.
MIN_DAYS_BEFORE_STOPPING = 7
N_SAMPLES = 10_000             # Monte Carlo samples for posterior comparison


@dataclass
class BayesianState:
    """
    Tracks the posterior distribution parameters for one variant.
    For Binary metrics  → Beta(alpha, beta) posterior
    For Revenue metrics → Log-Normal: Welford's online algorithm on log scale
    """
    # Beta-Binomial (conversion / CTR)
    alpha: float = 1.0   # prior: uniform Beta(1,1)
    beta: float  = 1.0

    # Log-Normal (revenue) — Welford's on log(daily_revenue_per_user)
    log_n: int     = 0
    log_mean: float = 0.0
    log_M2: float  = 0.0

    def update_binary(self, successes: int, trials: int):
        """Bayesian update for binary outcome: Beta posterior."""
        self.alpha += successes
        self.beta  += (trials - successes)

    def update_revenue(self, daily_revenue: float, n_spenders: int):
        """
        Log-normal Bayesian update using batch Welford's weighted by spender count.
        Each spender is one observation; log(revenue/spender) is the log-scale value.
        This gives ~conversion_rate * daily_users observations per day instead of 1.
        """
        if n_spenders <= 0 or daily_revenue <= 0:
            return
        log_per_spender = np.log(daily_revenue / n_spenders)
        n_new = self.log_n + n_spenders
        delta = log_per_spender - self.log_mean
        self.log_mean += n_spenders * delta / n_new
        delta2 = log_per_spender - self.log_mean
        self.log_M2 += n_spenders * delta * delta2
        self.log_n = n_new

    def sample_posterior(self, n_samples: int = N_SAMPLES,
                         metric: str = "conversion") -> np.ndarray:
        """Draw samples from the posterior distribution."""
        rng = np.random.default_rng()
        if metric in ("conversion", "ctr"):
            return rng.beta(self.alpha, self.beta, n_samples)
        else:
            # Log-normal posterior: sample mu on log scale, exponentiate
            log_var = self.log_M2 / max(self.log_n - 1, 1)
            log_se  = np.sqrt(max(log_var, 1e-6) / max(self.log_n, 1))
            return np.exp(rng.normal(self.log_mean, log_se, n_samples))

    @property
    def posterior_mean(self) -> float:
        if self.log_n > 0:
            log_var = self.log_M2 / max(self.log_n - 1, 1)
            return float(np.exp(self.log_mean + log_var / 2))  # log-normal mean
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_std(self) -> float:
        if self.log_n > 0:
            log_var = self.log_M2 / max(self.log_n - 1, 1)
            log_se  = np.sqrt(max(log_var, 1e-6) / max(self.log_n, 1))
            return float(np.exp(self.log_mean) * log_se)
        a, b = self.alpha, self.beta
        return np.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))


@dataclass
class DayResult:
    """Results snapshot for one day of experiment."""
    day: int
    prob_b_beats_a: float
    prob_a_beats_b: float
    expected_lift: float          # (B_mean - A_mean) / A_mean
    credible_interval_b: Tuple    # 95% CI for B's rate
    early_stop: bool
    declared_winner: Optional[str]  # 'A', 'B', or None


def run_bayesian_analysis(
    df_a: "pd.DataFrame",
    df_b: "pd.DataFrame",
    metric: str = "conversion",
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[List[DayResult], BayesianState, BayesianState]:
    """
    Core engine: process day-by-day data, update posteriors, check early stop.

    Returns:
        day_results: list of DayResult (one per day)
        state_a, state_b: final posterior states
    """
    state_a = BayesianState()
    state_b = BayesianState()
    day_results: List[DayResult] = []

    prev_events_a = prev_users_a = prev_spenders_a = 0
    prev_events_b = prev_users_b = prev_spenders_b = 0
    early_stopped = False
    stop_day = None

    for idx, (row_a, row_b) in enumerate(zip(df_a.itertuples(), df_b.itertuples())):
        day = row_a.day

        # ── Daily increments ──
        inc_users_a    = row_a.users    - prev_users_a
        inc_users_b    = row_b.users    - prev_users_b
        inc_events_a   = row_a.events   - prev_events_a
        inc_events_b   = row_b.events   - prev_events_b
        inc_spenders_a = row_a.spenders - prev_spenders_a
        inc_spenders_b = row_b.spenders - prev_spenders_b

        # ── Update posteriors ──
        if metric in ("conversion", "ctr"):
            state_a.update_binary(int(inc_events_a), int(inc_users_a))
            state_b.update_binary(int(inc_events_b), int(inc_users_b))
        else:
            state_a.update_revenue(inc_events_a, inc_spenders_a)
            state_b.update_revenue(inc_events_b, inc_spenders_b)

        # ── Monte Carlo posterior comparison ──
        samples_a = state_a.sample_posterior(N_SAMPLES, metric)
        samples_b = state_b.sample_posterior(N_SAMPLES, metric)

        prob_b_beats_a = float(np.mean(samples_b > samples_a))
        prob_a_beats_b = 1.0 - prob_b_beats_a

        # Expected lift: (B - A) / A
        mean_a = state_a.posterior_mean
        mean_b = state_b.posterior_mean
        expected_lift = (mean_b - mean_a) / mean_a if mean_a > 0 else 0.0

        # 95% credible interval for B
        ci_b = (float(np.percentile(samples_b, 2.5)),
                float(np.percentile(samples_b, 97.5)))

        # ── Early stop decision ──
        can_stop = (day >= MIN_DAYS_BEFORE_STOPPING) and not early_stopped
        stop_now = can_stop and (
            prob_b_beats_a >= confidence_threshold or
            prob_a_beats_b >= confidence_threshold
        )

        declared_winner = None
        if stop_now and not early_stopped:
            early_stopped = True
            stop_day = day
            declared_winner = "B" if prob_b_beats_a >= confidence_threshold else "A"

        # After stop, keep recording but mark as stopped
        if early_stopped and declared_winner is None:
            declared_winner = "B" if prob_b_beats_a >= confidence_threshold else "A"

        day_results.append(DayResult(
            day=day,
            prob_b_beats_a=prob_b_beats_a,
            prob_a_beats_b=prob_a_beats_b,
            expected_lift=expected_lift,
            credible_interval_b=ci_b,
            early_stop=early_stopped,
            declared_winner=declared_winner if early_stopped else None,
        ))

        prev_users_a, prev_events_a, prev_spenders_a = row_a.users, row_a.events, row_a.spenders
        prev_users_b, prev_events_b, prev_spenders_b = row_b.users, row_b.events, row_b.spenders

    return day_results, state_a, state_b


def compute_traditional_result(
    df_a, df_b, metric: str,
    state_a: "BayesianState | None" = None,
    state_b: "BayesianState | None" = None,
) -> dict:
    """
    Traditional frequentist test (two-proportion z-test or Welch's t-test).
    For revenue, uses the log-scale variance from the Bayesian states so the
    standard error is derived from actual data rather than a hardcoded guess.
    """
    final_a = df_a.iloc[-1]
    final_b = df_b.iloc[-1]

    if metric in ("conversion", "ctr"):
        # Two-proportion z-test
        n_a, x_a = int(final_a.users), int(final_a.events)
        n_b, x_b = int(final_b.users), int(final_b.events)
        p_a = x_a / n_a;  p_b = x_b / n_b
        p_pool = (x_a + x_b) / (n_a + n_b)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        z  = (p_b - p_a) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        significant = p_value < 0.05
        lift = (p_b - p_a) / p_a if p_a > 0 else 0
    else:
        # Welch's t-test on log scale using variance from the Bayesian state
        if state_a is not None and state_b is not None and state_a.log_n > 1 and state_b.log_n > 1:
            log_var_a = state_a.log_M2 / (state_a.log_n - 1)
            log_var_b = state_b.log_M2 / (state_b.log_n - 1)
            se_a_sq   = log_var_a / state_a.log_n
            se_b_sq   = log_var_b / state_b.log_n
            se        = np.sqrt(se_a_sq + se_b_sq)
            t         = (state_b.log_mean - state_a.log_mean) / se if se > 0 else 0
            # Welch-Satterthwaite degrees of freedom
            df_w      = (se_a_sq + se_b_sq) ** 2 / (
                se_a_sq ** 2 / (state_a.log_n - 1) + se_b_sq ** 2 / (state_b.log_n - 1)
            )
            p_value   = 2 * (1 - stats.t.cdf(abs(t), df_w))
            lift      = float(np.exp(state_b.log_mean - state_a.log_mean) - 1)
        else:
            # Fallback when states unavailable (shouldn't happen in normal use)
            mean_a, mean_b = final_a.rate, final_b.rate
            se_a = mean_a * 0.3 / np.sqrt(final_a.users)
            se_b = mean_b * 0.3 / np.sqrt(final_b.users)
            se   = np.sqrt(se_a**2 + se_b**2)
            t    = (mean_b - mean_a) / se if se > 0 else 0
            df_w = min(final_a.users, final_b.users) - 1
            p_value = 2 * (1 - stats.t.cdf(abs(t), df_w))
            lift = (mean_b - mean_a) / mean_a if mean_a > 0 else 0
        significant = p_value < 0.05

    return {
        "significant": significant,
        "p_value": p_value,
        "lift": lift,
        "days_required": 30,   # Full horizon
        "winner": ("B" if lift > 0 else "A") if significant else "No clear winner",
    }


def get_early_stop_day(day_results: List[DayResult]) -> Optional[int]:
    """Return first day early stopping was triggered, or None."""
    for r in day_results:
        if r.early_stop and r.declared_winner is not None:
            return r.day
    return None


def compute_time_saved(
    day_results: List[DayResult],
    total_days: int = 30,
    cost_per_day: float = 5_000,
) -> dict:
    """Calculate time and cost savings vs fixed-horizon testing."""
    stop_day = get_early_stop_day(day_results)
    if stop_day is None:
        stop_day = total_days  # No early stop

    days_saved = total_days - stop_day
    pct_saved  = days_saved / total_days * 100
    cost_saved = days_saved * cost_per_day

    return {
        "bayesian_stop_day": stop_day,
        "traditional_days":  total_days,
        "days_saved":        days_saved,
        "pct_saved":         pct_saved,
        "cost_saved_usd":    cost_saved,
    }