# Unscented Kalman Filter for Bitcoin Volatility

A 5-state Unscented Kalman Filter (UKF) that tracks hidden price cycles and volatility in real-time Bitcoin data — and out-forecasts the entire GARCH family on every standard metric. A second section then uses those latent states to construct mean-reversion signals, comparing the UKF against a plain rolling z-score.

---

# Part 1 — Volatility Model

## Why This Matters

Bitcoin's daily volatility swings between roughly 20% annualised (quiet markets) and 150%+ (crisis periods) — ten times more volatile than the S&P 500 on its worst days. A $10,000 BTC position that costs you $200 in daily risk one month can cost you $1,500+ the next. Getting the volatility estimate wrong by even 20% cascades into bad decisions across every downstream task:

- **Risk management** — position sizing, margin calls, stop-loss placement
- **Options pricing** — the implied vol surface is priced off a vol forecast; a lag here misprices every contract
- **Portfolio construction** — vol-adjusted weights only work if the vol estimate tracks reality

The industry standard is GARCH: a formula that sets today's expected volatility as a weighted average of yesterday's forecast and yesterday's squared return. It is fast, robust, and widely understood. It is also backward-looking by construction — it *follows* volatility rather than anticipating it, and consistently lags during regime shifts.

This project asks whether a probabilistic state-space model, fitted properly to BTC data, can do better. The answer is yes: the UKF achieves **0.40 correlation with realised volatility vs 0.25 for the best GARCH variant** — a near-doubling of predictive signal — with a lower absolute forecast error across the board.

---

## What the Model Does (Plain English)

Think of a Kalman filter as a GPS navigator for hidden quantities. A GPS doesn't directly observe where you are — it takes noisy signals from satellites and continuously refines its estimate, balancing what it predicted based on where it thought you were moving against what the new signal says. A Kalman filter does the same thing for quantities we care about but cannot directly see.

Here, the hidden quantities are:

1. **A slow price cycle (~29 days)** — a long-wavelength rhythm in BTC returns, like the broad swell of an ocean.
2. **A fast price cycle (~5 days)** — a short-wavelength rhythm, like the chop on top of the swell.
3. **The current volatility level** — how wildly prices are swinging right now, tracked as a hidden state that evolves day to day.

The model doesn't assume these cycle periods in advance. It finds them from five years of daily BTC data using Maximum Likelihood Estimation (MLE): a statistical procedure that asks "what parameter values make the data most likely?" The result — 28.9 days and 5.1 days — is the data's own answer to the question.

The "Unscented" in UKF refers to how the filter handles the fact that volatility is nonlinear (it is tracked in log-space to stop it going negative). Instead of approximating curves with straight lines — which can accumulate errors — the UKF propagates a carefully chosen set of sample points through the exact nonlinear function, giving a more accurate result with no derivatives required.

---

## Key Results

All metrics are computed out-of-sample on the final 30% of the 5-year dataset (~548 trading days not used in fitting).

### Forecast accuracy vs GARCH benchmarks

| Model | MAE ↓ | QLIKE ↓ | Corr(σ, \|r\|) ↑ |
|-------|--------|---------|-------------------|
| **UKF (this model)** | **0.0132** | **-6.707** | **0.403** |
| EGARCH(1,1) | 0.0161 | -6.564 | 0.251 |
| GJR-GARCH(1,1) | 0.0161 | -6.561 | 0.246 |
| GARCH(1,1) | 0.0160 | -6.551 | 0.210 |
| EWMA (λ=0.94) | 0.0140 | -6.548 | 0.182 |

- **MAE** (Mean Absolute Error): average gap between forecast vol and realised vol. Lower = better.
- **QLIKE**: a loss function that penalises confident but wrong forecasts more heavily. More negative = better.
- **Corr**: rank correlation between the model's daily vol forecast and the absolute return that day. Higher = the model tracks when big days actually happen.

The correlation advantage is the most meaningful number here. Getting the *level* of vol right (MAE) is useful; getting *when* it is high or low right (Corr) is what drives real trading and risk decisions. A 0.40 vs 0.25 gap means the UKF is tracking regime shifts that GARCH's one-step updating consistently misses.

---

## Understanding the Plots

### Plot 1 — Price history, UKF volatility estimate, standardised innovations

![Price and Volatility](plots/vol_comparison.png)

**Top panel:** BTC price (log, right axis) with the UKF's annualised volatility estimate as the green shaded band (left axis). Watch how the band expands sharply around the 2022 crash and the late-2021 ATH, and compresses during the quiet stretches. This is the filter adapting in real-time: it does not wait for volatility to appear in past returns — it infers it from the current hidden state.

**Middle panel:** UKF volatility (blue) vs EWMA benchmark (orange dashes) over the same period. The shaded region marks the out-of-sample window. The UKF tends to move more decisively during regime transitions; EWMA smooths them out.

**Bottom panel:** Standardised innovations — each day's return divided by the model's predicted standard deviation. A well-calibrated model should produce innovations that look like independent draws from N(0,1): scattered randomly inside the ±2σ dotted lines with no obvious clustering. The result is mostly good, with a few outliers during extreme market events (expected for any model on BTC).

---

### Plot 2 — Return confidence bands and OOS forecast vs realised scatter

![Return Fit and OOS Scatter](plots/return_fit.png)

**Left panel:** Actual daily log-returns (orange) vs the model's ±2σ confidence band (teal shaded region). A calibrated model should contain roughly 95% of returns inside this band. Watch how the band breathes with the market — it is wide during volatile periods and narrow during quiet ones. The vertical dashed line is the train/test split.

**Right panel:** Out-of-sample scatter. Each dot is a single trading day. X-axis: what the model forecast for volatility that morning. Y-axis: what actually happened (|return|). The closer the cloud hugs the y = x diagonal, the better the forecast. The UKF (green, ρ = 0.41) sits noticeably closer to the diagonal than EWMA (red, ρ = 0.19), confirming the correlation advantage seen in the table.

---

### Plot 3 — Innovation diagnostics

![Diagnostics](plots/diagnostics.png)

These three panels are technical health checks on the model's residuals.

**Left (QQ Plot):** If the model's errors were perfectly Gaussian, all dots would sit on the red line. They mostly do, with slight lifting at the tails — the kurtosis is 4.7 vs 3.0 for a perfect normal. BTC has genuinely fat-tailed events that no daily model fully captures, but the UKF is in reasonable shape.

**Middle (ACF of innovations):** Bars show the autocorrelation of standardised errors at each time lag. Bars inside the red dashed lines are statistically indistinguishable from zero. Some mild negative autocorrelations at short lags suggest slight over-smoothing (the filter sometimes over-corrects), but there is no strong, persistent pattern — the model is not obviously misspecified.

**Right (ACF of squared innovations):** This tests whether volatility clustering remains in the residuals after the filter has done its job. The lag-1 bar (0.18) exceeds the 95% threshold, indicating some residual ARCH structure. The model captures most of the volatility dynamics but not all — a jump component or heavier-tailed process noise on h_t would likely close this gap.

---

### Plot 4 — GARCH benchmark comparison

![GARCH Benchmark](plots/benchmark.png)

**Top panel:** All five models' out-of-sample conditional volatility estimates plotted together over the last ~18 months of data. The UKF (blue) traces a broadly similar path to the GARCH variants but reacts to regime changes more sharply and with less lag.

**Bottom row:** Bar charts for each metric (MAE, QLIKE, Corr). The UKF bar dominates across all three. The star (✦) marks the best result in each panel. The size of the Corr advantage — nearly double the GARCH(1,1) value — is the headline finding.

---

## Technical Details

### State vector

The filter tracks 5 hidden (latent) states simultaneously at each timestep:

| State | Symbol | Description |
|-------|--------|-------------|
| 1 | `p1` | Slow-cycle position (~30-day damped oscillator) |
| 2 | `v1` | Slow-cycle velocity |
| 3 | `p2` | Fast-cycle position (~5-day damped oscillator) |
| 4 | `v2` | Fast-cycle velocity |
| 5 | `h`  | Log-variance AR(1) — latent stochastic volatility |

### State transition

Each oscillator follows a damped harmonic transition. Let `w = 2π/T` (natural angular frequency in rad/day), `wd = w·sqrt(1−ζ²)` (damped frequency), and `ζ` (damping ratio):

```
[p_t]   =  A(w, ζ) · [p_{t-1}]
[v_t]                 [v_{t-1}]

A(w, ζ) = exp(−ζw) · [ cos(wd)       sin(wd)/wd ]
                       [ −wd·sin(wd)   cos(wd)    ]
```

Log-variance follows a mean-reverting AR(1):

```
h_t = mu_h + phi·(h_{t-1} − mu_h) + w_h,    w_h ~ N(0, sigma_h²)
```

With phi = 0.949, the half-life of a volatility shock is approximately 13 days.

### Dual-observation design

A single return observation leaves the log-variance state unidentifiable — there are infinitely many (cycle, vol) combinations consistent with any one return. The observation vector is augmented with a log-realised-variance proxy using the Harvey-Ruiz-Shephard (1994) log-linearisation:

```
z1  =  p1 + p2 + eps_t,     eps_t ~ N(0, exp(h_t))       [daily log-return]
z2  ≈  h_t − 1.27 + eta_t,  eta_t ~ N(0, π²/2)           [log(r_t²) proxy]
```

The second observation exploits the fact that `log(r_t²) = h_t + log(ε_t²)`, and `log(χ²(1))` has known mean −1.27 and variance π²/2. This gives the filter an independent signal on `h_t` without requiring intraday data, restoring identifiability.

### MLE parameter fitting

Structural parameters `(T_slow, ζ_slow, T_fast, ζ_fast, phi, log_qh)` are estimated by maximising the exact log-likelihood using Harvey's (1989) Prediction Error Decomposition (PED). Each innovation and its predicted covariance are produced directly by the filter, making the likelihood tractable. Optimisation uses L-BFGS-B; standard errors come from the numerical Hessian at the optimum.

**Fitted parameters on 5-year BTC-USD history (1,826 observations):**

| Parameter | MLE estimate | 95% CI |
|-----------|-------------|--------|
| T_slow (days) | 28.9 | [23.8, 34.1] |
| ζ_slow (damping) | 0.950 | — (boundary) |
| T_fast (days) | 5.1 | [3.3, 6.8] |
| ζ_fast (damping) | 0.284 | [0.048, 0.519] |
| phi (vol persistence) | 0.949 | [0.933, 0.964] |
| log q_h | -2.69 | [-3.12, -2.26] |

The 28.9-day and 5.1-day cycle periods are not assumed — they are the data's answer to "what are the dominant periodicities in BTC returns?" extracted from first principles via MLE.

---

## Design Decisions

| Choice | Rationale |
|--------|-----------|
| UKF over EKF | No Jacobian required; better accuracy for the log-variance nonlinearity |
| Log-variance state | Positivity guaranteed without constrained optimisation |
| Dual observation | Resolves rank-deficiency in the observation mapping |
| Damped oscillator | Parsimonious autocorrelation structure; MLE-fitted periods are interpretable |
| PED MLE | Exact likelihood for state-space models; analytical standard errors via Hessian |
| EWMA + GARCH benchmarks | Industry-standard baselines; positions the UKF in a familiar context |

---

## Model Comparison Notes

### Gaussian UKF vs Student-t observation noise (VB filter)

A variational-Bayes extension replacing Gaussian observation noise with Student-t was implemented (`run_student_t_filter` in `btc_modal.py`) and evaluated via grid search over ν ∈ {3, …, 15}.

![Gaussian vs Student-t Comparison](plots/comparison.png)

**What the plot shows:** Top-left compares the two models' out-of-sample volatility estimates. Top-right shows innovation distributions against their theoretical references. Bottom-left is a QQ plot (Gaussian kurtosis 4.68 vs Student-t kurtosis 6.13). Bottom-right is a downside VaR calibration test — bar heights should match the black expected line if the model is well-calibrated; both models over-estimate tail risk, but the Gaussian UKF is closer to the expected level.

**Finding:** The Student-t approach worsened all OOS metrics. Innovation kurtosis *increased* from 4.68 to 6.13 (optimal ν = 15). The reason: BTC fat tails arise from genuine volatility regime shifts, not measurement noise. The VB mechanism treats outliers as noise and downweights them — which prevents the filter from updating the log-variance state during the most informative observations, exactly when you need it most.

The correction belongs in the **process noise** (heavier-tailed transitions on `h_t`) or in an explicit jump component — not the observation model.

See `compare_models.py` for the full comparison.

---

# Part 2 — Mean Reversion Signal Exploration

The latent states produced by the UKF — the log-variance `h_t`, the fast-cycle position `p2_t`, and (in the 6-state extension) the drift `μ_t` — contain information beyond what a rolling volatility window can see. This section tests whether that information translates into a tradeable mean-reversion edge.

## Data

| Source | What | Access |
|--------|------|--------|
| Yahoo Finance | BTC-USD daily close, last 5 years (~1,826 observations) | `yfinance`, no key needed |
| Simulated | Synthetic SV process calibrated to BTC (seed=7) | `data_loader.py`, `USE_REAL_DATA = False` |

**Train/test split:** 70/30. The first 1,278 days are used for UKF warm-up and filter stabilisation; the final 548 days are the strictly held-out OOS test set. No signal parameter is fitted on OOS data. All signals use only information available at the close of day *t*; the resulting position is held over day *t+1*.

---

## The Three Strategies

### Strategy A — Baseline rolling z-score

The simplest possible fade: short when today's return was more than 2 standard deviations above a 20-day rolling mean, long when it was more than 2 below.

```
signal_t = –sign(r_t / σ_roll_t)   if |r_t / σ_roll_t| > 2.0   else 0
```

No UKF states are used. This is the benchmark that the UKF approaches must beat.

---

### Strategy B — 5-state UKF Composite

Uses three latent states from the 5-state UKF. All three conditions must hold simultaneously to generate a signal.

**Condition 1 — Low-vol regime: `h_{t-1} ≤ μ_h`**

The UKF's log-variance state `h_t` tracks the current level of market volatility. `μ_h` is its long-run unconditional mean. When `h_{t-1}` is below that mean the market is in a quiet, ranging regime.

This is the most important filter. In equity markets, mean reversion is often strongest during high-volatility periods (panic selling, overshoots). BTC is different: high-vol in BTC typically means momentum and cascade dynamics — liquidation spirals, FOMO breakouts, forced unwinds. Big moves in a high-vol BTC market tend to *continue*. Mean reversion is most reliable during low-vol, oscillating conditions when there is no sustained trend.

**Condition 2 — UKF z-score: `|r_t / σ_{t-1}| > 1.5`**

Instead of normalising by a rolling standard deviation (which is slow to update at regime transitions), the signal is normalised by the UKF's vol estimate from the previous day. This z-score reacts faster to real changes in the volatility state.

**Condition 3 — Cycle alignment: `sign(p2_{t-1}) == sign(r_t)`**

The 5-day oscillator state `p2_t` tracks where BTC sits in its short-term cycle. When the cycle was already moving in the same direction as today's return — meaning the cycle and the return reinforce each other — the market is near a natural turning point. A fade signal here has the cycle working in its favour.

```
signal_t = –sign(r_t / σ_ukf_t)   if all three conditions hold   else 0
```

Sizing: `TARGET_VOL / σ_ukf_{t-1}`, then scaled ±30% based on how deep into the low-vol regime the market is (larger position when `h_{t-1}` is well below `μ_h`).

---

### Strategy C — 6-state UKF (adds slope/drift filter)

The 6-state model extends the 5-state by adding a time-varying drift state `μ_t` (a random walk) that absorbs any persistent directional trend in returns. The oscillator states `p1, p2` then track only the cyclical residual around that trend, making them cleaner mean-reversion signals.

Strategy C uses the same three conditions as B — but computed from the 6-state filter's states — plus one additional filter:

**Condition 4 — Slope filter: `|μ_{t-1}| < 0.3%/day`**

The drift state `μ_t` captures whether BTC is in a sustained directional trend. When `|μ_t|` is large, fading a big return risks fighting an ongoing trend — the kind of move where mean reversion fails and losses accumulate quickly. Restricting signals to periods when `|μ_t|` is near zero confines trades to genuinely non-trending market conditions.

Note that `μ_t` is distinct from the oscillator velocity states `v1, v2`. The velocities measure cycle-phase speed within each oscillator; `μ_t` measures drift that persists across multiple oscillator cycles.

```
signal_t = –sign(r_t / σ_6ukf_t)   if all four conditions hold   else 0
```

---

## Backtest Setup

| Parameter | Value |
|-----------|-------|
| Universe | BTC-USD, daily close-to-close log-returns |
| OOS window | 548 trading days (final 30% of 5-year dataset) |
| Transaction costs | 10 bp round-trip (conservative estimate for spot BTC) |
| Vol target | 1% daily ≈ 16% annualised |
| Max leverage | 5× |
| Sizing rule | `TARGET_VOL / σ_{t-1}`, with ±30% regime adjustment for B/C |
| Hold period | 1 day |

---

## Results

### Backtest Plot

![Mean Reversion Backtest](plots/mean_reversion_backtest.png)

**Panel 1 — Equity curves.** Cumulative P&L for all three strategies and a vol-scaled buy-and-hold. Strategy B accumulates steadily; Strategy A drifts negative. The dotted line marks the start of the most recent 90-day window.

**Panel 2 — Vol regime and signal entries.** The log-variance deviation `h_{t-1} − μ_h` (orange) divides the OOS period into low-vol ranging zones (blue fill) and high-vol trending periods (red fill). Entry markers show that B and C fire almost exclusively inside the blue zone, confirming the vol regime filter is doing real work.

<<<<<<< HEAD
The UKF's correlation advantage is most striking: **0.40 vs 0.25 (EGARCH)**
: the state-space decomposition tracks vol regime shifts that GARCH one-step
updating consistently lags behind.
=======
**Panel 3 — UKF z-score.** Standardised returns `r_t / σ_{t-1}`. The threshold lines for A (±2.0) and B/C (±1.5) show that B/C enter on smaller overextensions but in better market conditions.
>>>>>>> 430f583 (Add mean reversion backtest: 5-state vs 6-state UKF vs rolling z-score)

**Panel 4 — Cycle and drift states.** Left: the 5-day oscillator `p2` at entry days, showing entries cluster near cycle turning points. Right: the 6-state drift state `μ_t` with Strategy C's slope filter threshold; C only trades when drift is near zero.

<<<<<<< HEAD
**Price history, UKF volatility estimate and standardised innovations**

![Price and Volatility](plots/vol_comparison.png)

**Return confidence bands and OOS forecast vs realised scatter**

![Return Fit and OOS Scatter](plots/return_fit.png)

**Innovation diagnostics - QQ, ACF, ARCH test**

![Diagnostics](plots/diagnostics.png)

**GARCH benchmark comparison**

![GARCH Benchmark](plots/benchmark.png)
=======
**Panel 5 — Rolling Sharpe and conditional decomposition.** Left: 60-day rolling Sharpe for all three strategies. Right: mean next-day fade return for each progressive filter combination, mapped directly to A, B, and C.
>>>>>>> 430f583 (Add mean reversion backtest: 5-state vs 6-state UKF vs rolling z-score)

---

### Performance Table — Full OOS (548 days)

| Metric | BTC Buy-and-Hold | A: Rolling z-score | B: 5-state UKF | C: 6-state UKF |
|--------|:-:|:-:|:-:|:-:|
| Ann. return | −9.1% | −5.3% | **+5.0%** | +2.0% |
| Sharpe | −0.55 | −0.90 | **0.70** | 0.30 |
| Sortino | −0.76 | −0.40 | **0.40** | 0.14 |
| Max drawdown | −32.7% | −12.5% | **−4.2%** | −5.9% |
| Calmar | −0.28 | −0.42 | **1.17** | 0.33 |
| Trade days | 547 | 73 | 57 | 43 |
| Trade freq | 99.8% | 13.3% | 10.4% | 7.8% |
| Dir. win rate | 48.4% | 46.2% | **63.3%** | 60.9% |

<<<<<<< HEAD
A variational-Bayes extension replacing Gaussian observation noise with a
Student-t was implemented (`run_student_t_filter` in `btc_modal.py`) and
evaluated via grid search over nu in {3, ..., 15}.

**Finding:** The VB approach worsened all OOS metrics. Innovation kurtosis
*increased* from 4.68 to 6.13 (optimal nu = 15). The reason: BTC fat tails
arise from genuine volatility regime shifts, not measurement noise. The VB
mechanism treats outliers as noise and downweights them, preventing the filter
from updating the log-variance state during the most informative observations.

The correction belongs in the **process noise** (Student-t transitions on
`h_t`) or in an explicit jump component, not the observation model.
See `compare_models.py` for the full comparison.
=======
*10 bp round-trip costs. 1% daily vol target. Regime-adjusted sizing for B and C.*
>>>>>>> 430f583 (Add mean reversion backtest: 5-state vs 6-state UKF vs rolling z-score)

---

### Performance Table — Most Recent 90 OOS Days

| Metric | A: Rolling z-score | B: 5-state UKF | C: 6-state UKF |
|--------|:-:|:-:|:-:|
| Ann. return | −19.4% | −8.2% | −7.7% |
| Sharpe | −2.17 | −1.82 | −1.71 |
| Max drawdown | −7.5% | −3.0% | −2.8% |
| Trade days | 10 | 5 | 4 |
| Dir. win rate | 33.3% | 33.3% | 50.0% |

The recent window was an unfavourable period for all fade strategies — BTC trended for extended stretches, which is exactly when the vol regime filter says not to trade. B and C fire fewer signals and suffer smaller drawdowns than A as a result.

---

### Conditional Decomposition

This table measures mean next-day fade return (pre-cost, no sizing) for progressively stricter filter combinations. It isolates the marginal contribution of each UKF component and maps each row directly to one of the three strategies.

| Condition | n | Mean ret | Win% | t |
|-----------|:-:|:-:|:-:|:-:|
| ① \|z\|>1.5, no filter | 71 | +0.39% | 54.9% | 1.09 |
| ② + low-vol (h≤μ) | 63 | +0.45% | 55.6% | 1.15 |
| ③ + high-vol (h>μ) | 8 | −0.12% | 50.0% | −0.17 |
| ④ + cycle alignment | 34 | +0.89% | 58.8% | 1.47 |
| **⑤ + low-vol + cycle  [Strategy B]** | **30** | **+1.19%** | **63.3%** | **1.80 ·** |
| ⑥ 6-state base (low-vol + cycle) | 30 | +1.05% | 60.0% | 1.56 · |
| **⑦ + slope filter  [Strategy C]** | **23** | **+0.83%** | **60.9%** | **1.16** |
| ⑧ \|z\|>2.0 only  [Strategy A] | 35 | +0.41% | 51.4% | 0.77 |

*· borderline significant (\|t\| ≥ 1.5). No filter reached \|t\| ≥ 1.96 at this sample size.*

---

## Evaluation and Findings

### Does the 5-state UKF have edge over a plain z-score?

Yes, on this OOS sample. Strategy B (5-state UKF) outperforms Strategy A (rolling z-score) on every metric:

- **Sharpe: +0.70 vs −0.90** — a 1.6-point swing
- **Directional win rate: 63.3% vs 46.2%** — the UKF z-score correctly identifies the next day's direction on 2 in 3 signals vs fewer than half for the rolling approach
- **Max drawdown: −4.2% vs −12.5%** — the vol regime filter keeps the strategy out of the market's worst stretches

The decomposition table shows exactly why. Row ① (z-score alone) has a t-statistic of 1.09 and a win rate of 54.9%. Each UKF component added incrementally:

- Adding the **low-vol regime filter** (row ②) barely moves the signal alone (+0.06%), but it is a prerequisite for the cycle condition to matter.
- Adding **cycle alignment** (row ④) boosts mean return to +0.89% and win rate to 58.8%.
- Combining both **low-vol + cycle** (row ⑤, Strategy B) gives +1.19% mean return, 63.3% win rate, and t=1.80 — the best result in the table. Both signals are unique to the UKF; a rolling z-score cannot produce them.

The critical economic insight is that **vol regime direction is backwards from equities**. Adding the high-vol filter (row ③) produces a *negative* mean return (−0.12%), confirming that BTC's high-vol periods are trending/momentum environments, not mean-reverting ones. The UKF's h_t state correctly identifies which regime the market is in.

---

### Does the 6-state slope filter improve on the 5-state?

Not on this OOS sample. The 6-state base (row ⑥) produces near-identical results to the 5-state (row ⑤): same trade count (n=30), slightly lower mean (+1.05% vs +1.19%), slightly lower t-stat (1.56 vs 1.80). Adding the slope filter (row ⑦, Strategy C) reduces the trade count to 23 and lowers the mean return further to +0.83%, cutting the Sharpe from 0.70 to 0.30.

The slope filter is removing some of the 5-state's best trades — it is over-filtering. This is likely because the low-vol regime condition (condition 1) already selects for non-trending periods. By construction, if `h_{t-1} ≤ μ_h`, the market is already in a quiet state where sustained drift `μ_t` tends to be small anyway. The slope filter then removes a subset of those trades where drift is slightly elevated but the market is still effectively ranging, and those turns out to be genuinely good mean-reversion opportunities.

The conclusion is that the 5-state model's conditions are better calibrated to this signal than the 6-state extension on this dataset. The 6-state model's drift state `μ_t` adds interpretive value (it shows when a trend is in progress) but does not improve the signal in practice.

---

### Statistical power and limitations

None of the strategies reach conventional 5% significance (|t| ≥ 1.96). Strategy B's t=1.80 is borderline. This is not a sign the signal is weak — it is a sample size problem:

- At ~26 signals/year (Strategy B's rate), reaching 100 observations takes ~4 years of live OOS data
- The 5% significance threshold at n=30 requires |t| ≥ 2.05; the signal mean of +1.19% is economically meaningful but the variance is wide

What would strengthen the evidence:

1. **Longer hold period.** A 3–5 day hold captures the full ~5-day fast-cycle arc, reduces the per-trade cost burden by ~3×, and increases statistical power without requiring more calendar time.
2. **MLE-fitted 6-state parameters.** The drift noise `σ_μ = 0.001/day` is hand-set. Fitting it via prediction-error decomposition would let the data choose how reactive the drift state should be, potentially making the slope filter more selective.
3. **Intraday vol data.** The dual-observation design currently uses `log(r²)` as a daily vol proxy. Adding realised variance from 5-minute returns would give the h-state a much cleaner signal, likely improving regime identification at transitions.

---

## Repository Structure

```
├── btc_modal.py                # 5-state and 6-state UKF; Student-t VB filter
├── mle_ped.py                  # MLE via prediction error decomposition
├── benchmark_garch.py          # GARCH(1,1), EGARCH, GJR-GARCH vs UKF
├── compare_models.py           # Gaussian vs Student-t UKF comparison
├── make_plots.py               # Regenerate all plots from live data
├── mean_reversion_backtest.py  # 3-strategy mean reversion backtest (A/B/C)
├── data_loader.py              # yfinance loader (USE_REAL_DATA toggle)
├── ukf_bitcoin_mle.ipynb       # Self-contained notebook
├── plots/                      # Generated figures
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Run the filter and base evaluation:**
```bash
python btc_modal.py
```

**Run MLE parameter fitting** (slower — optimisation + Hessian, ~3–5 minutes):
```bash
python mle_ped.py
```

**Run GARCH benchmark:**
```bash
python benchmark_garch.py
```

**Run mean reversion backtest (A/B/C):**
```bash
python mean_reversion_backtest.py
```

**Regenerate all plots from live data:**
```bash
python make_plots.py
```

**Run the notebook:**
```bash
jupyter notebook ukf_bitcoin_mle.ipynb
```

### Data

<<<<<<< HEAD
`data_loader.py` defaults to `USE_REAL_DATA = True`, downloading the last
5 years of BTC-USD daily closes from Yahoo Finance via `yfinance`. Set
`USE_REAL_DATA = False` to use a synthetic series (calibrated to BTC empirics,
seeded for reproducibility): useful for fast iteration when developing the
model.

---

## Design decisions

| Choice | Rationale |
|--------|-----------|
| UKF over EKF | No Jacobian required; better accuracy for the log-variance nonlinearity |
| Log-variance state | Positivity guaranteed without constrained optimisation |
| Dual observation | Resolves rank-deficiency in the observation mapping |
| Damped oscillator | Parsimonious autocorrelation structure; MLE-fitted periods are interpretable |
| PED MLE | Exact likelihood for state-space models; analytical standard errors via Hessian |
| EWMA + GARCH benchmarks | Industry-standard baselines; positions the UKF in a familiar context |
=======
`data_loader.py` defaults to `USE_REAL_DATA = True`, downloading the last 5 years of BTC-USD daily closes from Yahoo Finance via `yfinance`. Set `USE_REAL_DATA = False` to use a synthetic series (calibrated to BTC empirics, seeded for reproducibility) — useful for fast iteration when developing the model.
>>>>>>> 430f583 (Add mean reversion backtest: 5-state vs 6-state UKF vs rolling z-score)

---

## Requirements

Python 3.9+

Dependencies: `filterpy`, `numpy`, `scipy`, `matplotlib`, `yfinance`, `arch`, `jupyter`

---

## Background

Built as part of a personal research project exploring state-space methods in crypto markets. The identifiability issue and its dual-observation fix are discussed in detail in the notebook. The GARCH benchmark and Student-t comparison were added to give the model honest quantitative context rather than evaluating it in isolation. The mean reversion backtest in Part 2 was a natural extension — the UKF's latent states are potential alpha signals, and this section tests that hypothesis on live OOS data.
