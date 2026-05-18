"""
bandit.py — Multi-Armed Bandit (Thompson Sampling) Simulation
Simulates dynamically routing traffic based on Bayesian posterior probabilities.
"""

import numpy as np
import pandas as pd
from simulate import TestConfig
from bayesian_model import BayesianState, N_SAMPLES

def run_thompson_sampling(config: TestConfig, total_days: int = 30) -> pd.DataFrame:
    """
    Simulates a Multi-Armed Bandit using Thompson Sampling.
    Traffic is dynamically routed based on P(B > A).
    """
    rng          = np.random.default_rng(config.seed)
    baseline_rng = np.random.default_rng(42)   # fixed seed — stable baseline regardless of bandit path

    daily_users = max((config.n_users * 2) // total_days, 40)

    # Separate log-normal parameters per variant so AOV uplift is modelled correctly
    _aov_a = config.revenue_mean
    _aov_b = config.treatment_revenue_mean if config.treatment_revenue_mean is not None else config.revenue_mean
    _sigma_log_a = np.sqrt(np.log(1 + (config.revenue_std / _aov_a) ** 2))
    _mu_log_a    = np.log(_aov_a) - _sigma_log_a ** 2 / 2
    _sigma_log_b = np.sqrt(np.log(1 + (config.revenue_std / _aov_b) ** 2))
    _mu_log_b    = np.log(_aov_b) - _sigma_log_b ** 2 / 2
    
    state_a = BayesianState()
    state_b = BayesianState()
    
    rows = []
    
    cum_users_a = cum_events_a = 0
    cum_users_b = cum_events_b = 0
    
    # Track the standard A/B test (50/50 split) events for comparison
    cum_baseline_events = 0
    
    for day in range(1, total_days + 1):
        # 1. Calculate P(B > A) from current posteriors
        if day == 1:
            prob_b_beats_a = 0.5 # Start 50/50
        else:
            samples_a = state_a.sample_posterior(N_SAMPLES, config.metric)
            samples_b = state_b.sample_posterior(N_SAMPLES, config.metric)
            prob_b_beats_a = float(np.mean(samples_b > samples_a))
            
        # 2. Allocate traffic
        # Bound exploration between 10% and 90% so we never completely stop learning
        alloc_b = max(0.10, min(0.90, prob_b_beats_a))
        alloc_a = 1.0 - alloc_b
        
        # Total users for the day
        daily_n = int(rng.normal(daily_users, daily_users * 0.1))
        daily_n = max(daily_n, 10)
        
        # Route traffic
        n_b = rng.binomial(daily_n, alloc_b)
        n_a = daily_n - n_b
        
        # 3. Simulate conversions
        # Baseline 50/50 uses a fixed RNG so the comparison is stable across renders
        base_n_a = daily_n // 2
        base_n_b = daily_n - base_n_a

        if config.metric in ("conversion", "ctr"):
            events_a = int(rng.binomial(n_a, config.baseline_rate))
            events_b = int(rng.binomial(n_b, config.treatment_rate))
            base_events_a = int(baseline_rng.binomial(base_n_a, config.baseline_rate))
            base_events_b = int(baseline_rng.binomial(base_n_b, config.treatment_rate))
            cum_baseline_events += base_events_a + base_events_b
        else:
            # Revenue: separate log-normal params per variant
            spenders_a = rng.binomial(n_a, config.baseline_rate)
            spenders_b = rng.binomial(n_b, config.treatment_rate)
            events_a = float(rng.lognormal(_mu_log_a, _sigma_log_a, spenders_a).sum()) if spenders_a > 0 else 0.0
            events_b = float(rng.lognormal(_mu_log_b, _sigma_log_b, spenders_b).sum()) if spenders_b > 0 else 0.0

            b_spenders_a = baseline_rng.binomial(base_n_a, config.baseline_rate)
            b_spenders_b = baseline_rng.binomial(base_n_b, config.treatment_rate)
            b_events_a = float(baseline_rng.lognormal(_mu_log_a, _sigma_log_a, b_spenders_a).sum()) if b_spenders_a > 0 else 0.0
            b_events_b = float(baseline_rng.lognormal(_mu_log_b, _sigma_log_b, b_spenders_b).sum()) if b_spenders_b > 0 else 0.0
            cum_baseline_events += b_events_a + b_events_b
            
        # 4. Update posteriors
        if config.metric in ("conversion", "ctr"):
            state_a.update_binary(events_a, n_a)
            state_b.update_binary(events_b, n_b)
        else:
            state_a.update_revenue(float(events_a), int(spenders_a))
            state_b.update_revenue(float(events_b), int(spenders_b))
            
        cum_users_a += n_a
        cum_users_b += n_b
        cum_events_a += events_a
        cum_events_b += events_b
        
        rows.append({
            "day": day,
            "alloc_a": alloc_a,
            "alloc_b": alloc_b,
            "users_a": n_a,
            "users_b": n_b,
            "cum_users_a": cum_users_a,
            "cum_users_b": cum_users_b,
            "events_a": events_a,
            "events_b": events_b,
            "cum_events_a": cum_events_a,
            "cum_events_b": cum_events_b,
            "prob_b_beats_a": prob_b_beats_a,
            "cum_baseline_events": cum_baseline_events
        })
        
    return pd.DataFrame(rows)
