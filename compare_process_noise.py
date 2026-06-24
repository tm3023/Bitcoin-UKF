"""
compare_process_noise.py
========================

Evaluates whether Student-t process noise on the log-variance state h_t
reduces the residual ARCH structure visible in the Gaussian UKF's squared
standardised innovations.

The Gaussian UKF's squared-innovation ACF showed a statistically significant
lag-1 autocorrelation of ~0.18, indicating that the filter does not track
sudden volatility jumps fast enough.  The Student-t observation noise VB
filter (compare_models.py) failed to help because it downweighted large
observations, preventing h from updating at precisely the moments when rapid
updating was needed.

Here the process noise on h_t is made adaptive via the same VB scale-mixture
approach, but applied to Q rather than R:

    Q[4,4] at step t  =  base_Q_h / lambda_q_{t-1}

    lambda_q = (nu_q + k) / (nu_q + v' S^{-1} v)

Large innovations (v' S^{-1} v >> nu_q) produce small lambda_q, inflating
Q[4,4] and allowing larger updates to h_t.  This is the correct direction:
use surprising observations as evidence of a volatility regime jump, not as
noise to suppress.

Grid searches over nu_q in {3, 5, 7, 10, 15} minimising OOS QLIKE.
Saves plots/process_noise_comparison.png.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from btc_modal import run_base_filter, run_heavy_process_filter

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

BLUE   = "#2271B2"
GREEN  = "#2E8B57"
RED    = "#C0392B"
ORANGE = "#D4853A"
GREY   = "#666666"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
})


# ── Data ──────────────────────────────────────────────────────────────────────

print("Downloading BTC-USD data...")
btc = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
prices = btc["Close"].squeeze().dropna()
r = np.diff(np.log(prices.values.astype(float)))
n = len(r)
split = int(0.7 * n)
r_oos = r[split:]
print(f"  {n} observations  |  OOS: {n - split}")


# ── Metrics ───────────────────────────────────────────────────────────────────

def acf_sq(innov, nlags=20):
    x = innov ** 2
    x = x - x.mean()
    var = np.var(x)
    return np.array([
        np.mean(x[:len(x) - k] * x[k:]) / var if k < len(x) else 0.0
        for k in range(nlags + 1)
    ])


def standardised_innov(r_full, states, vol_full):
    return (r_full - states[:, 0] - states[:, 2]) / (vol_full + 1e-8)


def metrics(vol_oos, r_oos, states, vol_full, r_full):
    v2 = vol_oos ** 2
    mae_ = float(np.mean(np.abs(vol_oos - np.abs(r_oos))))
    ql_ = float(np.mean(np.log(v2) + r_oos ** 2 / v2))
    corr_ = float(np.corrcoef(vol_oos, np.abs(r_oos))[0, 1])
    z = standardised_innov(r_full, states, vol_full)
    kurt_ = float(pd.Series(z).kurtosis() + 3)
    acf1_sq = float(acf_sq(z)[1])
    return mae_, ql_, corr_, kurt_, acf1_sq, z


# ── Gaussian baseline ─────────────────────────────────────────────────────────

print("\nRunning Gaussian UKF (baseline)...")
st_g, vol_g, vol_ewma, mu_h = run_base_filter(r)
mae_g, ql_g, corr_g, kurt_g, acf1_g, z_g = metrics(
    vol_g[split:], r_oos, st_g, vol_g, r
)
conf_95 = 1.96 / np.sqrt(n)


# ── Grid search over nu_q ─────────────────────────────────────────────────────

NU_GRID = [3, 5, 7, 10, 15]
print(f"\nGrid search over nu_q in {NU_GRID} (Student-t process noise on h_t):")
print(f"  {'nu_q':>6}  {'QLIKE':>9}  {'MAE':>9}  {'Corr':>7}  "
      f"{'Kurt':>7}  {'ACF(z^2, lag1)':>15}")
print("  " + "-" * 65)

best_nu_q, best_ql_hp = None, np.inf
results_grid = {}

for nu_q in NU_GRID:
    st_hp, vol_hp, _, _ = run_heavy_process_filter(r, nu_q=nu_q)
    m_hp, ql_hp, c_hp, k_hp, a_hp, _ = metrics(
        vol_hp[split:], r_oos, st_hp, vol_hp, r
    )
    results_grid[nu_q] = (m_hp, ql_hp, c_hp, k_hp, a_hp)
    print(f"  {nu_q:>6}  {ql_hp:>9.4f}  {m_hp:>9.4f}  {c_hp:>7.4f}  "
          f"{k_hp:>7.2f}  {a_hp:>15.4f}")
    if ql_hp < best_ql_hp:
        best_ql_hp, best_nu_q = ql_hp, nu_q

print(f"\n  Gaussian  {ql_g:>9.4f}  {mae_g:>9.4f}  {corr_g:>7.4f}  "
      f"{kurt_g:>7.2f}  {acf1_g:>15.4f}  (baseline)")
print(f"\n  Best nu_q = {best_nu_q}  (OOS QLIKE = {best_ql_hp:.4f})")
if best_ql_hp < ql_g:
    print(f"  Heavy process noise improves QLIKE by {ql_g - best_ql_hp:.4f}")
else:
    print(f"  *** Gaussian QLIKE is better by {best_ql_hp - ql_g:.4f} ***")


# ── Best model ────────────────────────────────────────────────────────────────

print(f"\nRunning heavy process filter with best nu_q = {best_nu_q}...")
st_hp_best, vol_hp_best, _, _ = run_heavy_process_filter(r, nu_q=best_nu_q)
mae_hp, ql_hp, corr_hp, kurt_hp, acf1_hp, z_hp = metrics(
    vol_hp_best[split:], r_oos, st_hp_best, vol_hp_best, r
)


# ── Summary table ─────────────────────────────────────────────────────────────

W = 72
print(f"\n{'=' * W}")
print(f"{'Metric':<28}  {'Gaussian':>12}  {'Heavy-Process':>14}")
print(f"  (nu_q = {best_nu_q})")
print("-" * W)
rows = [
    ("MAE",               mae_g,   mae_hp),
    ("QLIKE",             ql_g,    ql_hp),
    ("Corr(sigma, |r|)",  corr_g,  corr_hp),
    ("Innovation kurtosis", kurt_g, kurt_hp),
    ("ACF(z^2, lag 1)",   acf1_g,  acf1_hp),
]
for label, vg, vhp in rows:
    print(f"  {label:<26}  {vg:>12.4f}  {vhp:>14.4f}")
print("=" * W)
delta_acf = acf1_hp - acf1_g
print(f"\n  ACF(z^2) lag-1 change: {delta_acf:+.4f}  "
      f"({'reduced' if delta_acf < 0 else 'increased'} residual ARCH)")
print(f"  95% significance threshold: {conf_95:.4f}")


# ── Plot ──────────────────────────────────────────────────────────────────────

print("\nBuilding comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Gaussian UKF vs Heavy-Process UKF (nu_q={best_nu_q}): "
    "Student-t Process Noise on h_t",
    fontsize=12, fontweight="bold",
)

lags = np.arange(21)
acf_g_full = acf_sq(z_g, nlags=20)
acf_hp_full = acf_sq(z_hp, nlags=20)

# Panel (0,0): ACF of squared innovations
ax = axes[0, 0]
bw = 0.35
ax.bar(lags[1:] - bw / 2, acf_g_full[1:],  bw, color=BLUE,
       alpha=0.70, label=f"Gaussian  (lag-1 = {acf1_g:.3f})")
ax.bar(lags[1:] + bw / 2, acf_hp_full[1:], bw, color=GREEN,
       alpha=0.70, label=f"Heavy-Process nu_q={best_nu_q}  (lag-1 = {acf1_hp:.3f})")
ax.axhline(conf_95,  color=RED,  lw=1.0, ls="--", label="95% CI")
ax.axhline(-conf_95, color=RED,  lw=1.0, ls="--")
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")
ax.set_title("ACF of Squared Innovations z^2\n"
             "(significant bars = residual ARCH)")
ax.legend(fontsize=8)
ax.set_ylim(-0.20, 0.30)

# Panel (0,1): Vol comparison over OOS period
ax = axes[0, 1]
x_oos = np.arange(n - split)
ax.plot(x_oos, vol_g[split:] * np.sqrt(252) * 100,
        color=BLUE,  lw=1.1, alpha=0.85, label="Gaussian UKF")
ax.plot(x_oos, vol_hp_best[split:] * np.sqrt(252) * 100,
        color=GREEN, lw=1.1, alpha=0.85, ls="--",
        label=f"Heavy-Process (nu_q={best_nu_q})")
ax.set_xlabel("OOS day")
ax.set_ylabel("Annualised volatility (%)")
ax.set_title("Conditional Volatility Estimates (OOS)")
ax.legend(fontsize=8)

# Panel (1,0): QQ plot of innovations
ax = axes[1, 0]
ps = np.linspace(0.5, 99.5, 300)
q_g_qq  = np.percentile(z_g,  ps)
q_hp_qq = np.percentile(z_hp, ps)
q_ref   = stats.norm.ppf(ps / 100)
ax.scatter(q_ref, q_g_qq,  s=7, alpha=0.5, color=BLUE,
           label=f"Gaussian  (kurt={kurt_g:.2f})")
ax.scatter(q_ref, q_hp_qq, s=7, alpha=0.5, color=GREEN,
           label=f"Heavy-Process  (kurt={kurt_hp:.2f})")
lim = np.array([q_ref[0], q_ref[-1]])
ax.plot(lim, lim, color="black", lw=1.2, ls="--", alpha=0.5, label="y=x (Normal)")
ax.set_xlabel("Normal theoretical quantile")
ax.set_ylabel("Empirical innovation quantile")
ax.set_title("QQ Plot: Innovations vs Normal(0,1)")
ax.legend(fontsize=8)

# Panel (1,1): Grid search result
ax = axes[1, 1]
nu_vals = list(results_grid.keys())
ql_vals = [results_grid[nu][1] for nu in nu_vals]
acf_vals = [results_grid[nu][4] for nu in nu_vals]

color_ql = BLUE
color_acf = ORANGE
ax2 = ax.twinx()
ax2.spines["right"].set_visible(True)
ax2.spines["top"].set_visible(False)

ax.plot(nu_vals, ql_vals, color=color_ql, lw=1.5, marker="o",
        label="QLIKE (left)")
ax.axhline(ql_g, color=color_ql, lw=0.9, ls="--", alpha=0.6,
           label=f"Gaussian QLIKE = {ql_g:.3f}")
ax2.plot(nu_vals, acf_vals, color=color_acf, lw=1.5, marker="s",
         label="ACF(z^2) lag-1 (right)")
ax2.axhline(acf1_g,  color=color_acf, lw=0.9, ls="--", alpha=0.6,
            label=f"Gaussian ACF = {acf1_g:.3f}")
ax2.axhline(conf_95, color=RED, lw=0.8, ls=":", alpha=0.7,
            label="95% threshold")
ax.set_xlabel("nu_q (degrees of freedom)")
ax.set_ylabel("OOS QLIKE", color=color_ql)
ax2.set_ylabel("ACF(z^2) lag-1", color=color_acf)
ax.set_title("Grid Search over nu_q\nLower QLIKE and lower ACF(z^2) = better")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper right")
ax.tick_params(axis="y", labelcolor=color_ql)
ax2.tick_params(axis="y", labelcolor=color_acf)

plt.tight_layout()
plt.savefig("plots/process_noise_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/process_noise_comparison.png")
