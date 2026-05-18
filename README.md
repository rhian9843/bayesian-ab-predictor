# Bayesian A/B Test Outcome Predictor

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-69%20passed-brightgreen?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

A production-quality Bayesian A/B testing engine with a Streamlit dashboard. Declare experiment winners faster than fixed-horizon testing using sequential inference with early stopping, run a multi-armed bandit for dynamic traffic routing, and analyse your own data by uploading real CSVs.

**Across 12 simulated experiments, Bayesian early stopping saves an average of ~17 days per test (56% faster) compared to a fixed 30-day horizon.**

---

## Features

**Three tabs, one dashboard:**

- **Preset Experiments** — 12 realistic scenarios (conversion, CTR, revenue) with configurable confidence threshold and cost-per-day. Includes a portfolio view across all experiments, a Thompson Sampling dynamic routing simulator, and CSV/text export.
- **Run Custom Experiment** — Live day-by-day animation as the experiment runs. Configurable metric, conversion rate, improvement size, traffic, duration, and confidence threshold. Revenue experiments support separate control and treatment AOV.
- **Upload Real Data** — Upload two CSVs (one per variant), map columns, run the full Bayesian engine on your own data. Includes a true Thompson Sampling replay driven by real observed rates rather than a fixed-rate simulation.

---

## How It Works

### Conversion and CTR — Beta-Binomial model

Each variant's rate is modelled as a Beta distribution. The posterior updates analytically — no MCMC, no simulation required for inference:

```
Prior:      θ ~ Beta(1, 1)                     # uniform, no initial bias
Likelihood: X | θ ~ Binomial(n, θ)
Posterior:  θ | X ~ Beta(1 + successes, 1 + failures)
```

Each day: update both posteriors, draw 10,000 Monte Carlo samples, compute `P(B > A) = mean(samples_B > samples_A)`. Stop when confidence crosses the threshold (default 95%).

### Revenue — Log-Normal model with Welford's algorithm

Revenue per spender follows a log-normal distribution. The model runs Welford's online algorithm on the log scale, weighted by actual spender count per day. This gives hundreds of effective observations per experiment rather than 30 (one per day), making the posterior meaningfully informative within the first week.

```
Each spender is one observation: log(revenue / n_spenders)
Posterior mean: exp(μ_log + σ²_log / 2)   ← correct log-normal mean
```

Revenue lift can be modelled through conversion rate, average order value (AOV), or both independently.

### Early stopping

A 7-day burn-in prevents false positives from early lucky streaks — on day 1, three conversions in a row can reach 99% confidence with fewer than 10 observations. After day 7, a winner is declared as soon as `P(B > A) >= threshold` or `P(A > B) >= threshold`.

### Thompson Sampling

Instead of a fixed 50/50 split, traffic is dynamically routed proportionally to `P(B > A)`, bounded between 10% and 90% to preserve statistical learning. The baseline comparison uses a fixed-seed RNG, decoupled from the bandit, so "gain vs 50/50" is stable across renders.

### Frequentist comparison

A traditional test runs in parallel for comparison:
- **Conversion/CTR**: two-proportion z-test
- **Revenue**: Welch's t-test on the log scale with Welch-Satterthwaite degrees of freedom
- **Portfolio**: Benjamini-Hochberg FDR correction applied across all 12 simultaneous frequentist p-values. The Bayesian `P(B > A)` column needs no correction — it is a per-experiment posterior probability, not a test statistic.

---

## Bayesian vs Frequentist

| | Frequentist | Bayesian (this project) |
|---|---|---|
| Safe to peek early? | No — inflates false positive rate | Yes — designed for sequential updates |
| Output | p < 0.05 (binary) | P(B > A) = 94% (continuous) |
| Pre-committed sample size? | Required | Not required |
| Stakeholder communication | "Statistically significant" | "94% confident B is better" |
| Average time to decision | 30 days (fixed) | ~13 days (adaptive) |

---

## Quick Start

**Option 1 — pip**
```bash
git clone https://github.com/yourusername/bayesian-ab-predictor
cd bayesian-ab-predictor
pip install -r requirements.txt
streamlit run app.py
```

**Option 2 — Docker**
```bash
docker build -t ab-predictor .
docker run -p 8501:8501 ab-predictor
```

Open [http://localhost:8501](http://localhost:8501).

---

## Upload Your Own Data

The upload tab accepts any two CSVs with one row per day — comma or semicolon separated, with or without a date column.

**Expected format:**

| Date | Reach | # of Purchase |
|---|---|---|
| 01.08.2019 | 18204 | 1216 |
| 02.08.2019 | 17634 | 1197 |
| ... | ... | ... |

Column mapping is done in the UI — select which column is users (denominator) and which is events (numerator). The engine handles missing values via linear interpolation.

The project ships with a preprocessing script for the [Kaggle Facebook A/B Test dataset](https://www.kaggle.com/datasets/favourfavour/ab-test-campaign-data):

```bash
python preprocess.py
```

---

## Project Structure

```
.
├── app.py                  # Streamlit dashboard — tabs, controls, portfolio view
├── bayesian_model.py       # Core engine: Beta-Binomial, Log-Normal, early stopping
├── simulate.py             # 12 preset scenarios + TestConfig dataclass
├── bandit.py               # Thompson Sampling simulation
├── experiment_runner.py    # Custom experiment tab with live animation
├── upload_tab.py           # Real data upload, preprocessing, TS replay
├── preprocess.py           # Kaggle dataset preprocessing script
├── visualizations.py       # Plotly dark-theme charts
├── result_widgets.py       # Shared Streamlit render functions across tabs
├── export_report.py        # CSV + executive text summary export
├── tests/
│   ├── test_bayesian_model.py   # 27 tests: Beta-Binomial, Log-Normal, early stop
│   ├── test_simulate.py         # 14 tests: output schema, monotonicity, reproducibility
│   ├── test_bandit.py           # 10 tests: allocation bounds, cumulative columns
│   └── test_preprocess.py       # 18 tests: build_engine_df, fix_missing, load pipeline
├── Dockerfile
└── requirements.txt
```

---

## Preset Experiments

| Experiment | Metric | True Effect | Winner |
|---|---|---|---|
| Checkout Button Color | Conversion | +30% lift | B |
| Homepage Hero CTA | CTR | A outperforms | A |
| Pricing Page Layout | Conversion | +21% lift | B |
| Email Subject Line | CTR | +13% lift | B |
| Free Trial Offer | Conversion | No effect (null) | — |
| Onboarding Flow V2 | Conversion | A outperforms | A |
| Product Image Carousel | CTR | +7% lift | B |
| Upsell Modal Timing | Revenue | +33% spend lift | B |
| Search Result Ranking | CTR | A outperforms | A |
| Cart Abandonment Email | Conversion | +33% lift | B |
| Dashboard Redesign | Revenue | A outperforms | A |
| Mobile Nav Simplification | CTR | +4% lift | B |

Three experiments where A wins and one null result are intentional — they demonstrate the system avoids false positives and correctly detects regressions.

---

## Running the Tests

```bash
pytest tests/ -v
```

69 tests, no external dependencies beyond the requirements file. Tests that require the Kaggle dataset are automatically skipped if the files are not present.

---

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub (must be public)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **Create app**, select the repository, set main file to `app.py`
4. Click **Deploy** — Streamlit reads `requirements.txt` automatically

You get a public URL in ~2 minutes.

---

## Tech Stack

| | Tool | Why |
|---|---|---|
| Statistical inference | NumPy, SciPy | Conjugate priors and Welch's t-test; no MCMC, runs in milliseconds |
| Online statistics | Welford's algorithm | Numerically stable variance on the log scale without storing raw data |
| Visualisation | Plotly | Interactive dark-theme charts with hover, zoom, and export |
| Dashboard | Streamlit | Reactive UI, file upload, download buttons, tab layout |
| Containerisation | Docker | Reproducible single-command deployment |

No GPU. No LLMs. No heavy dependencies. Runs on any laptop.

---

## License

MIT — free to use, fork, and build on.
