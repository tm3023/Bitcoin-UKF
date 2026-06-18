"""
mean_reversion_backtest.py
==========================

Three mean-reversion strategies evaluated out-of-sample on BTC-USD data,
comparing a plain rolling z-score against the 5-state and 6-state UKF.

DATA
----
BTC-USD daily close log-returns, last 5 years via yfinance.
Split: 70% in-sample (UKF warm-up + signal calibration), 30% OOS.
Recent: last 90 OOS days reported separately.

STRATEGIES
----------
A  Baseline rolling z-score
     Signal: –sign(r_t / σ_roll_t),  |z| > 2.0  (20-day window)
     Sizing: TARGET_VOL / σ_roll_t  (no regime adjustment)

B  5-state UKF Composite
     Conditions (all required):
       1. Low-vol regime  : h_{t-1} ≤ μ_h  — in BTC, low vol = ranging
                            market; high vol = trend/momentum. Mean reversion
                            is only reliable in quiet, oscillating markets.
       2. UKF z-score     : |r_t / σ_{t-1}| > 1.5  (using UKF vol)
       3. Cycle alignment : sign(p2_{t-1}) == sign(r_t)  — the fast 5-day
                            oscillator is already pointing in the same
                            direction as today's move, placing us near a
                            natural cycle turning point.
     Sizing: vol-regime-adjusted (±30% around base; clipped [0.5×, 2.0×])

C  6-state UKF Composite
     All conditions from B, plus:
       4. Slope filter    : |μ_{t-1}| < 0.3%/day  — the 6-state model's
                            drift state μ_t is near zero, confirming no
                            sustained directional trend is in place.
     Uses the 6-state filter's h and p2 states throughout.
     Sizing: same vol-regime-adjusted rule as B.

ASSUMPTIONS
-----------
· Daily close-to-close log-returns; no intraday data.
· 10 bp round-trip cost charged on every position change.
· Target 1% daily vol (≈16% annualised) for position sizing.
· Position adjusted ±30% based on how far h deviates below μ_h.
· signal[t] uses only info at close of day t; position held over t+1.
"""

import os
import sys
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from btc_modal import run_base_filter, run_6state_filter
from data_loader import load_returns_with_dates

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────────
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

COLOURS = {
    "B&H":       "#888888",
    "Strategy A": "#C0392B",
    "Strategy B": "#2271B2",
    "Strategy C": "#2E8B57",
}

# ── Parameters ─────────────────────────────────────────────────────────────────
COSTS_BPS    = 10      # round-trip transaction costs (basis points)
Z_THRESH_A   = 2.0     # Strategy A: rolling-vol z-score entry threshold
Z_THRESH_BC  = 1.5     # Strategy B/C: UKF z-score entry threshold
ROLL_WIN     = 20      # rolling vol window (trading days)
TARGET_VOL   = 0.01    # daily vol target (1% ≈ 16% annualised)
MAX_SIZE     = 5.0     # max leverage cap
REGIME_SCALE = 0.30    # regime sizing sensitivity (per σ of h deviation)
SLOPE_THRESH = 0.003   # Strategy C: |μ_{t-1}| must be < 0.3%/day
RECENT_DAYS  = 90      # "recent window" reported separately within OOS


# ── Utilities ──────────────────────────────────────────────────────────────────

def rolling_std(r, window):
    n   = len(r)
    out = np.zeros(n)
    for t in range(n):
        s      = max(0, t - window + 1)
        out[t] = r[s:t + 1].std() if t > 0 else abs(r[0])
    return np.maximum(out, 1e-8)


def _lag(arr, fill):
    out    = np.empty_like(arr)
    out[0] = fill
    out[1:] = arr[:-1]
    return out


# ── Signal construction ────────────────────────────────────────────────────────

def compute_signals(r, states5, vol5, mu_h5, states6, vol6):
    """
    Build all three signal arrays.  signal[t] → position on day t+1.
    All conditions use only information available at the close of day t.
    """
    # Lagged states (info at close of day t-1, used to generate signal at t)
    h_lag5   = _lag(states5[:, 4], mu_h5)      # 5-state log-variance
    p2_lag5  = _lag(states5[:, 2], 0.0)        # 5-state fast-cycle position
    vol_lag5 = _lag(vol5,          vol5[0])    # 5-state conditional vol
    h_lag6   = _lag(states6[:, 4], mu_h5)      # 6-state log-variance
    p2_lag6  = _lag(states6[:, 2], 0.0)        # 6-state fast-cycle position
    vol_lag6 = _lag(vol6,          vol6[0])    # 6-state conditional vol
    mu_lag6  = _lag(states6[:, 5], 0.0)        # 6-state drift state

    vol_roll = rolling_std(r, ROLL_WIN)

    z_roll  = r / vol_roll                             # rolling z-score (A)
    z_ukf5  = r / np.maximum(vol_lag5, 1e-8)          # UKF z-score (B)
    z_ukf6  = r / np.maximum(vol_lag6, 1e-8)          # UKF z-score (C)

    # Vol-regime scale factor: larger position when deeper into low-vol
    sigma_h      = states5[:, 4].std()
    h_dev_norm5  = (mu_h5 - h_lag5) / max(sigma_h, 1e-6)   # + = low-vol
    regime_scale = np.clip(1.0 + REGIME_SCALE * h_dev_norm5, 0.5, 2.0)

    # ── Strategy A ────────────────────────────────────────────────────────────
    signal_A = np.where(
        np.abs(z_roll) > Z_THRESH_A, -np.sign(z_roll), 0.0
    )

    # ── Strategy B: 5-state UKF ───────────────────────────────────────────────
    cond_low_vol5 = h_lag5 <= mu_h5
    cond_z5       = np.abs(z_ukf5) > Z_THRESH_BC
    cond_cycle5   = (np.sign(p2_lag5) == np.sign(r)) & (p2_lag5 != 0.0)
    signal_B      = np.where(
        cond_low_vol5 & cond_z5 & cond_cycle5, -np.sign(z_ukf5), 0.0
    )

    # ── Strategy C: 6-state UKF (adds slope/drift filter) ────────────────────
    cond_low_vol6 = h_lag6 <= mu_h5
    cond_z6       = np.abs(z_ukf6) > Z_THRESH_BC
    cond_cycle6   = (np.sign(p2_lag6) == np.sign(r)) & (p2_lag6 != 0.0)
    cond_slope    = np.abs(mu_lag6) < SLOPE_THRESH
    signal_C      = np.where(
        cond_low_vol6 & cond_z6 & cond_cycle6 & cond_slope,
        -np.sign(z_ukf6), 0.0
    )

    return (
        signal_A, signal_B, signal_C,
        z_roll, z_ukf5, z_ukf6,
        vol_roll, vol_lag5, vol5,
        h_lag5, h_lag6, p2_lag5, p2_lag6, mu_lag6,
        regime_scale, mu_h5,
    )


# ── Backtest engine ────────────────────────────────────────────────────────────

def backtest_signal(r, signal, vol_for_sizing,
                    regime_scale=None, apply_regime=False):
    """
    Simulate a 1-day fade strategy.
    signal[t] → position at close of t → earns r[t+1].
    """
    n     = len(r)
    costs = COSTS_BPS / 10_000.0

    position  = np.zeros(n)
    net_pnl   = np.zeros(n)
    gross_pnl = np.zeros(n)

    for t in range(1, n):
        raw  = TARGET_VOL / max(vol_for_sizing[t - 1], 1e-8)
        if apply_regime and regime_scale is not None:
            raw *= regime_scale[t - 1]
        size = np.clip(raw, 0.0, MAX_SIZE)
        pos  = signal[t - 1] * size

        position[t]  = pos
        gross_pnl[t] = pos * r[t]
        net_pnl[t]   = gross_pnl[t] - abs(pos - position[t - 1]) * costs

    return net_pnl, gross_pnl, position


# ── Performance statistics ─────────────────────────────────────────────────────

def performance_stats(pnl, ann=252):
    pnl    = np.asarray(pnl)
    mu     = pnl.mean()
    sig    = pnl.std()
    sharpe = mu / sig * np.sqrt(ann) if sig > 0 else 0.0
    neg    = pnl[pnl < 0]
    sortino = mu / neg.std() * np.sqrt(ann) if len(neg) > 1 else 0.0
    equity  = np.cumsum(pnl)
    peak    = np.maximum.accumulate(equity)
    max_dd  = (equity - peak).min()
    calmar  = mu * ann / abs(max_dd) if max_dd < 0 else float("nan")
    n_tr    = int((pnl != 0).sum())
    return {
        "Ann. return":   mu * ann,
        "Sharpe":        sharpe,
        "Sortino":       sortino,
        "Max drawdown":  max_dd,
        "Calmar":        calmar,
        "Trade days":    n_tr,
        "Trade freq":    n_tr / len(pnl),
    }, np.cumsum(pnl)


def dir_win_rate(sig, r_ref):
    """Directional win rate: fraction of signal days with correct next-day direction."""
    idx = np.where(sig[:-1] != 0)[0]
    if len(idx) == 0:
        return float("nan"), 0
    return float((sig[idx] * r_ref[idx + 1] > 0).mean()), len(idx)


# ── Conditional decomposition ──────────────────────────────────────────────────

def conditional_decomposition(r_oos, z5_oos, z6_oos,
                               h_lag5_oos, h_lag6_oos,
                               p2_lag5_oos, p2_lag6_oos,
                               mu_lag6_oos, mu_h):
    """
    Measure mean next-day fade return for progressive filter combinations.
    Maps directly to the three strategies and isolates each component's
    marginal contribution.
    """
    z5  = z5_oos[:-1];      h5   = h_lag5_oos[:-1]
    z6  = z6_oos[:-1];      h6   = h_lag6_oos[:-1]
    p25 = p2_lag5_oos[:-1]; p26  = p2_lag6_oos[:-1]
    mu  = mu_lag6_oos[:-1]; rt   = r_oos[:-1]
    outcome = r_oos[1:]

    # 5-state conditions
    c_z5     = np.abs(z5) > Z_THRESH_BC
    c_low5   = h5 <= mu_h
    c_high5  = h5 > mu_h
    c_cycle5 = (np.sign(p25) == np.sign(rt)) & (p25 != 0.0)
    fade5    = -np.sign(z5) * outcome

    # 6-state conditions
    c_z6     = np.abs(z6) > Z_THRESH_BC
    c_low6   = h6 <= mu_h
    c_cycle6 = (np.sign(p26) == np.sign(rt)) & (p26 != 0.0)
    c_slope  = np.abs(mu) < SLOPE_THRESH
    fade6    = -np.sign(z6) * outcome

    combos = [
        ("① |z|>1.5, no filter  (5-state)",               c_z5,                                  fade5),
        ("② + low-vol  [h≤μ]",                             c_z5 & c_low5,                         fade5),
        ("③ + high-vol  [h>μ]",                            c_z5 & c_high5,                        fade5),
        ("④ + cycle alignment",                             c_z5 & c_cycle5,                       fade5),
        ("⑤ + low-vol + cycle  [Strategy B, 5-state]",     c_z5 & c_low5 & c_cycle5,              fade5),
        ("⑥ + low-vol + cycle  [6-state]",                 c_z6 & c_low6 & c_cycle6,              fade6),
        ("⑦ + low-vol + cycle + slope  [Strategy C]",      c_z6 & c_low6 & c_cycle6 & c_slope,    fade6),
        ("⑧ |z|>2.0 only  [Strategy A threshold]",         np.abs(z5) > Z_THRESH_A,               fade5),
    ]

    results = {}
    for label, mask, fade in combos:
        n_s  = int(mask.sum())
        sl   = fade[mask] if n_s > 0 else np.array([])
        t_s  = (sl.mean() / (sl.std() / np.sqrt(n_s))
                if n_s > 1 and sl.std() > 0 else 0.0)
        results[label] = {
            "n":        n_s,
            "mean_ret": float(sl.mean()) if n_s > 0 else 0.0,
            "win_rate": float((sl > 0).mean()) if n_s > 0 else 0.5,
            "t_stat":   float(t_s),
        }
    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def make_plots(r_oos, pnl_A, pnl_B, pnl_C, pnl_bnh,
               eq_A, eq_B, eq_C, eq_bnh,
               sA_oos, sB_oos, sC_oos,
               z_ukf5_oos, h_lag5_oos, h_lag6_oos,
               p2_5_oos, p2_6_oos, mu_lag6_oos,
               mu_h, cond, recent_days, n_oos):

    os.makedirs("plots", exist_ok=True)
    n   = len(r_oos)
    x   = np.arange(n)
    cut = n - recent_days

    fig = plt.figure(figsize=(15, 22))
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.50, wspace=0.32,
                            height_ratios=[1.4, 0.9, 0.75, 0.9, 0.9])
    fig.suptitle(
        "Mean Reversion Backtest — Rolling z-score (A) vs 5-State UKF (B) vs 6-State UKF (C)\n"
        f"OOS: last 30% of 5-year BTC-USD  |  {COSTS_BPS} bp round-trip  |  "
        f"{TARGET_VOL*100:.0f}% daily vol target",
        fontsize=11, fontweight="bold", y=1.001,
    )

    # ── 1. Equity curves (full width) ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, eq_bnh * 100, color=COLOURS["B&H"],        lw=1.0, alpha=0.5,
             label="BTC Buy-and-Hold (vol-scaled)")
    ax1.plot(x, eq_A   * 100, color=COLOURS["Strategy A"], lw=1.3,
             label=f"A: Rolling z-score  (|z|>{Z_THRESH_A})")
    ax1.plot(x, eq_B   * 100, color=COLOURS["Strategy B"], lw=1.8,
             label=f"B: 5-state UKF  (low-vol + |z|>{Z_THRESH_BC} + cycle)")
    ax1.plot(x, eq_C   * 100, color=COLOURS["Strategy C"], lw=1.8, ls="--",
             label=f"C: 6-state UKF  (B + slope filter |μ|<{SLOPE_THRESH*100:.1f}%/d)")
    ax1.axhline(0, color="black", lw=0.7)
    ax1.axvline(cut, color="grey", lw=1.0, ls=":", alpha=0.7)
    ax1.text(cut + 2, ax1.get_ylim()[0] * 0.75,
             f"← recent {recent_days}d →", fontsize=7, color="grey")
    ax1.set_ylabel("Cumulative P&L  (% of unit position)")
    ax1.set_title("Cumulative P&L — Full OOS Period")
    ax1.legend(fontsize=7.5, loc="upper left")

    # ── 2. Vol regime + entry markers (full width) ────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    h_dev = h_lag5_oos - mu_h
    ax2.fill_between(x, h_dev, 0, where=(h_lag5_oos <= mu_h),
                     color="lightblue", alpha=0.35, label="Low-vol (fade zone)")
    ax2.fill_between(x, h_dev, 0, where=(h_lag5_oos > mu_h),
                     color="salmon",    alpha=0.25, label="High-vol (no trade)")
    ax2.plot(x, h_dev, color="darkorange", lw=0.8, alpha=0.85,
             label="h_{t-1} − μ_h  (5-state)")
    ax2.axhline(0, color="black", lw=0.8)
    ylo = h_dev.min() - 0.10
    yhi = h_dev.max() + 0.10
    for sig, col, mk, lab in [
        (sB_oos, COLOURS["Strategy B"], "^", "B"),
        (sC_oos, COLOURS["Strategy C"], "D", "C"),
    ]:
        lx = np.where(sig > 0)[0]
        sx = np.where(sig < 0)[0]
        ax2.scatter(lx, np.full(len(lx), ylo), marker=mk, s=22,
                    color=col, zorder=3, alpha=0.8, label=f"{lab} Long")
        ax2.scatter(sx, np.full(len(sx), yhi), marker=mk, s=22,
                    color=col, zorder=3, alpha=0.8, label=f"{lab} Short")
    ax2.set_ylabel("Log-var deviation")
    ax2.set_title("Volatility Regime and Signal Entries (▲=B, ◆=C)")
    ax2.legend(fontsize=7, ncol=4, loc="upper right")

    # ── 3. UKF z-score (full width) ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(x, z_ukf5_oos, color=COLOURS["Strategy B"], lw=0.65, alpha=0.85,
             label="UKF z-score  (5-state, r_t / σ_{t-1})")
    ax3.axhline(0, color="black", lw=0.8)
    for thr, col, lab in [
        (Z_THRESH_BC, COLOURS["Strategy B"], f"±{Z_THRESH_BC}  B/C entry"),
        (Z_THRESH_A,  COLOURS["Strategy A"], f"±{Z_THRESH_A}   A entry"),
    ]:
        ax3.axhline(+thr, color=col, lw=0.9, ls="--", alpha=0.7, label=lab)
        ax3.axhline(-thr, color=col, lw=0.9, ls="--", alpha=0.7)
    ax3.set_ylabel("σ units")
    ax3.set_title("UKF Standardised Returns  (z-score)")
    ax3.legend(fontsize=7, ncol=3)

    # ── 4L. Fast cycle p2 (5-state) ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(x, p2_5_oos * 100, color=COLOURS["Strategy B"], lw=0.8, alpha=0.85,
             label="p2  (5-state fast cycle)")
    ax4.fill_between(x, p2_5_oos * 100, 0, alpha=0.08, color=COLOURS["Strategy B"])
    ax4.axhline(0, color="black", lw=0.8)
    lx5 = np.where(sB_oos > 0)[0]; sx5 = np.where(sB_oos < 0)[0]
    ax4.scatter(lx5, p2_5_oos[lx5] * 100, marker="^", s=20,
                color=COLOURS["Strategy B"], zorder=3, alpha=0.7, label="B Long")
    ax4.scatter(sx5, p2_5_oos[sx5] * 100, marker="v", s=20,
                color=COLOURS["Strategy B"], zorder=3, alpha=0.7, label="B Short")
    ax4.set_ylabel("Cycle position (%)")
    ax4.set_title("5-State Fast Cycle p2 with B Entries")
    ax4.legend(fontsize=7)

    # ── 4R. 6-state drift state μ_t ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.plot(x, mu_lag6_oos * 100, color=COLOURS["Strategy C"], lw=0.85,
             label="μ_{t-1}  (6-state drift, %/day)")
    ax5.axhline(0, color="black", lw=0.8)
    ax5.axhline(+SLOPE_THRESH * 100, color="red", lw=0.9, ls="--", alpha=0.65,
                label=f"±{SLOPE_THRESH*100:.1f}%  slope filter")
    ax5.axhline(-SLOPE_THRESH * 100, color="red", lw=0.9, ls="--", alpha=0.65)
    lx6 = np.where(sC_oos > 0)[0]; sx6 = np.where(sC_oos < 0)[0]
    ax5.scatter(lx6, mu_lag6_oos[lx6] * 100, marker="D", s=20,
                color=COLOURS["Strategy C"], zorder=3, alpha=0.7, label="C Long")
    ax5.scatter(sx6, mu_lag6_oos[sx6] * 100, marker="D", s=20,
                color=COLOURS["Strategy C"], zorder=3, alpha=0.7, label="C Short")
    ax5.set_ylabel("Drift (%/day)")
    ax5.set_title("6-State Drift μ_t (Slope / Trend State) with C Entries")
    ax5.legend(fontsize=7)

    # ── 5L. Rolling 60-day Sharpe ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[4, 0])
    win = 60

    def roll_sharpe(pnl, w):
        out = np.full(len(pnl), np.nan)
        for t in range(w, len(pnl)):
            sl = pnl[t - w:t]
            sd = sl.std()
            out[t] = sl.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
        return out

    ax6.plot(x, roll_sharpe(pnl_A, win), color=COLOURS["Strategy A"], lw=1.0,
             alpha=0.8, label="A  (rolling z)")
    ax6.plot(x, roll_sharpe(pnl_B, win), color=COLOURS["Strategy B"], lw=1.3,
             label="B  (5-state UKF)")
    ax6.plot(x, roll_sharpe(pnl_C, win), color=COLOURS["Strategy C"], lw=1.3,
             ls="--", label="C  (6-state UKF)")
    ax6.axhline(0,   color="black", lw=0.8)
    ax6.axhline(0.5, color="green", lw=0.7, ls=":", alpha=0.6, label="Sharpe=0.5")
    ax6.axvline(cut, color="grey",  lw=1.0, ls=":", alpha=0.7)
    ax6.set_ylabel("Rolling Sharpe (ann.)")
    ax6.set_title(f"Rolling {win}-Day Sharpe")
    ax6.legend(fontsize=7)
    ax6.set_ylim(-5, 6)

    # ── 5R. Conditional decomposition bar chart ────────────────────────────
    ax7 = fig.add_subplot(gs[4, 1])
    keys     = list(cond.keys())
    mean_r   = [cond[k]["mean_ret"] * 100 for k in keys]
    t_vals   = [cond[k]["t_stat"]         for k in keys]
    ns       = [cond[k]["n"]               for k in keys]
    labels_s = ["①\nraw z5", "②\n+lv", "③\n+hv", "④\n+cyc",
                 "⑤ B\n5st", "⑥\n6st\nbase", "⑦ C\n+slp", "⑧ A\nz>2"]
    bx       = np.arange(len(keys))
    bcols    = ["#888888", COLOURS["Strategy B"], "#C0392B", "#5D8AA8",
                 COLOURS["Strategy B"], COLOURS["Strategy C"],
                 COLOURS["Strategy C"], "#555555"]

    bars = ax7.bar(bx, mean_r, color=bcols, alpha=0.70, width=0.65)
    for b, t_s in zip(bars, t_vals):
        b.set_edgecolor("black" if abs(t_s) >= 1.5 else "none")
        b.set_linewidth(2.0 if abs(t_s) >= 1.96 else 1.0)
        b.set_alpha(0.80 if abs(t_s) >= 1.5 else 0.35)
    ax7.axhline(0, color="black", lw=0.8)
    ax7.set_xticks(bx)
    ax7.set_xticklabels(labels_s, fontsize=6.5)
    ax7.set_ylabel("Mean next-day fade ret (%)")
    ax7.set_title("Conditional Decomposition\n(solid border=|t|≥1.5, thicker=≥1.96)")
    for i, (n_s, t_s) in enumerate(zip(ns, t_vals)):
        yp = max(mean_r[i], 0) + 0.02
        ax7.text(i, yp, f"n={n_s}\nt={t_s:.1f}",
                 ha="center", va="bottom", fontsize=5.5)

    plt.savefig("plots/mean_reversion_backtest.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved  plots/mean_reversion_backtest.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def fmt(k, v):
    if isinstance(v, float) and v != v:
        return "n/a"
    m = {
        "Ann. return":   lambda x: f"{x*100:+.1f}%",
        "Sharpe":        lambda x: f"{x:.3f}",
        "Sortino":       lambda x: f"{x:.3f}",
        "Max drawdown":  lambda x: f"{x*100:.1f}%",
        "Calmar":        lambda x: f"{x:.2f}",
        "Trade days":    lambda x: f"{int(x)}",
        "Trade freq":    lambda x: f"{x:.1%}",
    }
    return m[k](v) if k in m else str(v)


def main():
    W = 76
    SEP = "=" * W

    print(SEP)
    print("Loading data …")
    r, _, dates = load_returns_with_dates()
    n     = len(r)
    split = int(0.7 * n)
    n_oos = n - split
    print(f"  Total: {n} days  |  Train: {split}  |  OOS: {n_oos}")

    print("\nRunning 5-state UKF …")
    states5, vol5, _, mu_h5 = run_base_filter(r)

    print("Running 6-state UKF (adds drift / slope state μ_t) …")
    states6, vol6, _ = run_6state_filter(r)

    print("\nBuilding signals …")
    (signal_A, signal_B, signal_C,
     z_roll, z_ukf5, z_ukf6,
     vol_roll, vol_lag5, vol5_curr,
     h_lag5, h_lag6, p2_lag5, p2_lag6, mu_lag6,
     regime_scale, mu_h) = compute_signals(
        r, states5, vol5, mu_h5, states6, vol6
    )

    # ── OOS slices ────────────────────────────────────────────────────────
    def oos(arr):
        return arr[split:]

    r_oos       = oos(r)
    sA          = oos(signal_A)
    sB          = oos(signal_B)
    sC          = oos(signal_C)
    vR          = oos(vol_roll)
    vU5         = oos(vol5_curr)
    vU6         = oos(vol6)
    rs          = oos(regime_scale)
    z_ukf5_oos  = oos(z_ukf5)
    z_ukf6_oos  = oos(z_ukf6)
    h_lag5_oos  = oos(h_lag5)
    h_lag6_oos  = oos(h_lag6)
    p2_5_oos    = oos(states5[:, 2])
    p2_6_oos    = oos(states6[:, 2])
    p2_lag5_oos = oos(p2_lag5)
    p2_lag6_oos = oos(p2_lag6)
    mu_lag6_oos = oos(mu_lag6)

    # ── Backtest ──────────────────────────────────────────────────────────
    print("Running backtests …")
    pnl_A, _, _ = backtest_signal(r_oos, sA, vR,
                                  apply_regime=False)
    pnl_B, _, _ = backtest_signal(r_oos, sB, vU5, rs,
                                  apply_regime=True)
    pnl_C, _, _ = backtest_signal(r_oos, sC, vU6, rs,
                                  apply_regime=True)

    bnh_sz  = np.clip(TARGET_VOL / np.maximum(vU5, 1e-8), 0, MAX_SIZE)
    pnl_bnh = np.concatenate([[0.0], bnh_sz[:-1] * r_oos[1:]])

    stats_A,   eq_A   = performance_stats(pnl_A)
    stats_B,   eq_B   = performance_stats(pnl_B)
    stats_C,   eq_C   = performance_stats(pnl_C)
    stats_bnh, eq_bnh = performance_stats(pnl_bnh)

    dw_A, ns_A = dir_win_rate(sA, r_oos)
    dw_B, ns_B = dir_win_rate(sB, r_oos)
    dw_C, ns_C = dir_win_rate(sC, r_oos)

    # ── Full OOS results ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"FULL OOS ({n_oos} days) | Costs: {COSTS_BPS} bp | "
          f"Vol target: {TARGET_VOL*100:.0f}%/day")
    print(SEP)
    hdr = (f"{'Metric':<20}  {'BTC B&H':>9}  {'A: z-score':>11}  "
           f"{'B: 5-state':>11}  {'C: 6-state':>11}")
    print(hdr)
    print("-" * W)
    for k in stats_A:
        print(f"{k:<20}  {fmt(k,stats_bnh[k]):>9}  "
              f"{fmt(k,stats_A[k]):>11}  "
              f"{fmt(k,stats_B[k]):>11}  "
              f"{fmt(k,stats_C[k]):>11}")
    print(f"\n  Directional win rate (signal direction vs next-day return):")
    print(f"    BTC  long-only: {(r_oos>0).mean():.1%}")
    print(f"    A  ({ns_A:>3} signals): {dw_A:.1%}")
    print(f"    B  ({ns_B:>3} signals): {dw_B:.1%}")
    print(f"    C  ({ns_C:>3} signals): {dw_C:.1%}")

    # ── Recent window ──────────────────────────────────────────────────────
    cut = n_oos - RECENT_DAYS
    if cut > 0:
        r_r = r_oos[cut:]

        def run_recent(sig_full, vsize, apply_reg=False):
            pnl_r, _, _ = backtest_signal(
                r_r, sig_full[cut:], vsize[cut:],
                regime_scale=rs[cut:], apply_regime=apply_reg,
            )
            s, _ = performance_stats(pnl_r)
            dw, ns = dir_win_rate(sig_full[cut:], r_r)
            return s, dw, ns

        sA_r, dwA_r, nsA_r = run_recent(sA, vR)
        sB_r, dwB_r, nsB_r = run_recent(sB, vU5, apply_reg=True)
        sC_r, dwC_r, nsC_r = run_recent(sC, vU6, apply_reg=True)

        print(f"\n{SEP}")
        print(f"RECENT WINDOW (last {RECENT_DAYS} OOS days = most recent data)")
        print(SEP)
        hdr2 = (f"{'Metric':<20}  {'A: z-score':>11}  "
                f"{'B: 5-state':>11}  {'C: 6-state':>11}")
        print(hdr2)
        print("-" * W)
        for k in sA_r:
            print(f"{k:<20}  {fmt(k,sA_r[k]):>11}  "
                  f"{fmt(k,sB_r[k]):>11}  {fmt(k,sC_r[k]):>11}")
        print(f"\n  Directional win rate (recent {RECENT_DAYS} days):")
        print(f"    A ({nsA_r:>3} signals): {dwA_r:.1%}")
        print(f"    B ({nsB_r:>3} signals): {dwB_r:.1%}")
        print(f"    C ({nsC_r:>3} signals): {dwC_r:.1%}")

    # ── Conditional decomposition ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("CONDITIONAL DECOMPOSITION  (next-day fade return, OOS, pre-cost)")
    print(SEP)
    cond = conditional_decomposition(
        r_oos, z_ukf5_oos, z_ukf6_oos,
        h_lag5_oos, h_lag6_oos,
        p2_lag5_oos, p2_lag6_oos,
        mu_lag6_oos, mu_h,
    )
    print(f"{'Condition':<55}  {'n':>4}  {'Mean':>8}  {'Win%':>6}  {'t':>6}")
    print("-" * W)
    for label, res in cond.items():
        sig = "  ★" if abs(res["t_stat"]) >= 1.96 else (
              "  ·" if abs(res["t_stat"]) >= 1.5  else "")
        print(f"{label:<55}  {res['n']:>4d}  "
              f"{res['mean_ret']*100:>+7.3f}%  "
              f"{res['win_rate']:>5.1%}  "
              f"{res['t_stat']:>6.2f}{sig}")
    print("  ★ |t|≥1.96 (5%)  · |t|≥1.5 (borderline)")

    # ── Summary ───────────────────────────────────────────────────────────
    sh_B = stats_B["Sharpe"]; sh_C = stats_C["Sharpe"]
    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    print(f"  5-state UKF (B) vs rolling z-score (A): "
          f"Sharpe {sh_B:+.3f} vs {stats_A['Sharpe']:+.3f}")
    print(f"  6-state UKF (C) vs 5-state UKF (B):     "
          f"Sharpe {sh_C:+.3f} vs {sh_B:+.3f}")
    print(f"  Directional win rate — A: {dw_A:.1%}  B: {dw_B:.1%}  C: {dw_C:.1%}")
    if sh_C > sh_B:
        print(f"  Slope filter adds +{sh_C-sh_B:.3f} Sharpe  "
              f"(reduced trades: {ns_B}→{ns_C}, higher signal quality).")
    else:
        print(f"  Slope filter: {sh_C-sh_B:+.3f} Sharpe vs B  "
              f"(reduced trades: {ns_B}→{ns_C}).")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    make_plots(
        r_oos, pnl_A, pnl_B, pnl_C, pnl_bnh,
        eq_A, eq_B, eq_C, eq_bnh,
        sA, sB, sC,
        z_ukf5_oos, h_lag5_oos, h_lag6_oos,
        p2_5_oos, p2_6_oos, mu_lag6_oos,
        mu_h, cond, RECENT_DAYS, n_oos,
    )


if __name__ == "__main__":
    main()
