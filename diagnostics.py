"""
E0: innovation diagnostics for the 5-state UKF.

Re-runs the base filter from btc_modal.run_base_filter step by step and
records the one-step-ahead innovation v_t = z_t - E[z_t | z_{1:t-1}] and its
predicted covariance S_t (filterpy's ukf.y, ukf.S after each update). Under a
correctly specified Gaussian model:

  - each channel's standardised innovation v_t[i] / sqrt(S_t[i,i]) ~ N(0, 1)
  - the normalised innovation squared NIS_t = v_t' S_t^{-1} v_t ~ chi^2(2)

Channel 1 is the return (Gaussian noise assumed). Channel 2 is log(r^2), whose
true noise is log chi^2(1), approximated as Gaussian (QML). Even a perfectly
specified SV model gives channel 2 skewness -1.535 and excess kurtosis 4.0
(the standardised cumulants of log chi^2(1)), so channel 1 is the direct test
of the Gaussian return assumption.

The data window is fixed (START..END) rather than a rolling "5y" so that the
results are reproducible; prices are cached in data/ (gitignored).

Outputs: results/e0/  (summary.json, summary.md, qq.png, nis_timeline.png,
environment.txt)

Usage:
    python diagnostics.py
"""

import json
import os
import platform
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from scipy import stats

from btc_modal import DT, damped_osc_matrix, run_base_filter

warnings.filterwarnings("ignore")

START = "2021-10-02"
END = "2026-10-02"          # inclusive; yfinance's end is exclusive
BURN_IN = 50                # same burn-in as the MLE likelihood in mle_ped.py
VOL_WINDOW = 30             # days, for the realised-vol regime shading
OUT = os.path.join("results", "e0")
CACHE = os.path.join("data", f"btc_usd_close_{START}_{END}.csv")

# Well-documented BTC stress episodes inside the window (for plot annotation only)
EVENTS = {
    "2022-05-09": "Terra/LUNA",
    "2022-06-13": "Celsius/3AC",
    "2022-11-08": "FTX",
    "2023-03-10": "SVB/USDC",
    "2024-08-05": "Yen carry unwind",
}


def load_prices():
    if os.path.exists(CACHE):
        s = pd.read_csv(CACHE, index_col=0, parse_dates=True).iloc[:, 0]
        return s
    import yfinance as yf
    end_excl = (pd.Timestamp(END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    btc = yf.download("BTC-USD", start=START, end=end_excl, interval="1d",
                      progress=False, auto_adjust=False)
    s = btc["Close"].squeeze().dropna().astype(float)
    os.makedirs("data", exist_ok=True)
    s.to_frame("close").to_csv(CACHE)
    return s


def run_filter_with_innovations(r):
    """Exact replica of run_base_filter that also records (y_t, S_t)."""
    n = len(r)
    mu_h = 2 * np.log(r.std())
    phi = 0.95
    A1 = damped_osc_matrix(2 * np.pi / 30.0, 0.15)
    A2 = damped_osc_matrix(2 * np.pi / 5.0, 0.35)

    def fx(state, dt):
        p1, v1, p2, v2, h = state
        p1n, v1n = A1 @ np.array([p1, v1])
        p2n, v2n = A2 @ np.array([p2, v2])
        return np.array([p1n, v1n, p2n, v2n, mu_h + phi * (h - mu_h)])

    def hx(state):
        p1, _, p2, _, h = state
        return np.array([p1 + p2, h - 1.27])

    sp = MerweScaledSigmaPoints(n=5, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=DT, fx=fx, hx=hx, points=sp)
    ukf.x = np.array([0.0, 0.0, 0.0, 0.0, mu_h])
    ukf.P = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 0.1])
    ukf.Q = np.diag([1e-6, 1e-5, 1e-5, 1e-4, 0.05])

    states = np.zeros((n, 5))
    ys = np.zeros((n, 2))
    Ss = np.zeros((n, 2, 2))
    for t in range(n):
        h_est = ukf.x[4]
        ukf.R = np.array([[np.exp(h_est), 0.0], [0.0, np.pi**2 / 2.0]])
        ukf.predict()
        ukf.update(np.array([r[t], np.log(r[t]**2 + 1e-8)]))
        states[t] = ukf.x
        ys[t] = ukf.y
        Ss[t] = ukf.S
    return states, ys, Ss


def moments(z):
    return {
        "n": int(len(z)),
        "mean": float(np.mean(z)),
        "std": float(np.std(z, ddof=1)),
        "skewness": float(stats.skew(z)),
        "excess_kurtosis": float(stats.kurtosis(z, fisher=True)),
        "jarque_bera_stat": float(stats.jarque_bera(z).statistic),
        "jarque_bera_p": float(stats.jarque_bera(z).pvalue),
    }


def nis_stats(nis, dof):
    lo, hi = stats.chi2.ppf([0.025, 0.975], dof)
    q95 = stats.chi2.ppf(0.95, dof)
    return {
        "dof": dof,
        "mean_nis": float(np.mean(nis)),
        "expected_mean": float(dof),
        "band_95_two_sided": [float(lo), float(hi)],
        "frac_outside_two_sided": float(np.mean((nis < lo) | (nis > hi))),
        "frac_below_lower": float(np.mean(nis < lo)),
        "frac_above_upper": float(np.mean(nis > hi)),
        "frac_above_one_sided_95": float(np.mean(nis > q95)),
        "expected_frac": 0.05,
        # each tail of the two-sided band should hold 2.5% of steps
        "binom_p_below": float(stats.binomtest(int(np.sum(nis < lo)), len(nis), 0.025).pvalue),
        "binom_p_above": float(stats.binomtest(int(np.sum(nis > hi)), len(nis), 0.025).pvalue),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    prices = load_prices()
    dates = prices.index[1:]
    r = np.diff(np.log(prices.values))
    n = len(r)
    split = int(0.7 * n)

    states, ys, Ss = run_filter_with_innovations(r)
    ref_states = run_base_filter(r)[0]
    assert np.allclose(states, ref_states), "replica diverged from run_base_filter"

    # Per-channel standardised innovations and NIS (both from the prior S_t)
    z1 = ys[:, 0] / np.sqrt(Ss[:, 0, 0])
    z2 = ys[:, 1] / np.sqrt(Ss[:, 1, 1])
    nis = np.einsum("ti,tij,tj->t", ys, np.linalg.inv(Ss), ys)
    nis1, nis2 = z1**2, z2**2

    # The README's "standardised innovation": post-update residual / filtered vol
    vol = np.exp(states[:, 4] / 2)
    z_readme = (r - (states[:, 0] + states[:, 2])) / (vol + 1e-8)

    keep = np.arange(n) >= BURN_IN
    oos = np.arange(n) >= split
    summary = {
        "window": {"start": str(dates[0].date()), "end": str(dates[-1].date()),
                   "n_returns": n, "burn_in": BURN_IN,
                   "oos_start": str(dates[split].date())},
        "replica_matches_run_base_filter": True,
        "reference_logchi2_1": {"skewness": -1.535, "excess_kurtosis": 4.0},
        "channel1_return": {"all": moments(z1[keep]), "oos": moments(z1[oos])},
        "channel2_logr2": {"all": moments(z2[keep]), "oos": moments(z2[oos])},
        "readme_style_residual": {"all": moments(z_readme[keep])},
        "nis_joint": {"all": nis_stats(nis[keep], 2), "oos": nis_stats(nis[oos], 2)},
        "nis_channel1": {"all": nis_stats(nis1[keep], 1), "oos": nis_stats(nis1[oos], 1)},
        "nis_channel2": {"all": nis_stats(nis2[keep], 1), "oos": nis_stats(nis2[oos], 1)},
    }

    # Realised-vol regime: top decile of 30-day rolling std of returns
    rv = pd.Series(r, index=dates).rolling(VOL_WINDOW).std()
    hi_vol = rv >= rv.quantile(0.9)
    summary["high_vol_regime"] = {
        "definition": f"{VOL_WINDOW}-day rolling std in top decile",
        "frac_outside_joint_band_in_high_vol": float(
            np.mean(((nis < stats.chi2.ppf(0.025, 2)) | (nis > stats.chi2.ppf(0.975, 2)))[keep & hi_vol.values])),
        "frac_outside_joint_band_elsewhere": float(
            np.mean(((nis < stats.chi2.ppf(0.025, 2)) | (nis > stats.chi2.ppf(0.975, 2)))[keep & ~hi_vol.values])),
    }

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── QQ plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, z, title in [
        (axes[0], z1[keep], "Channel 1: return innovation"),
        (axes[1], z2[keep], "Channel 2: log(r²) innovation"),
        (axes[2], z_readme[keep], "README-style residual (post-update)"),
    ]:
        (osm, osr), (slope, icpt, _) = stats.probplot(z, dist="norm")
        ax.scatter(osm, osr, s=5, alpha=0.5, color="#2271B2")
        lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        ax.plot(lim, lim, color="black", lw=1, ls="--", label="N(0,1)")
        m = moments(z)
        ax.set_title(f"{title}\nexcess kurt {m['excess_kurtosis']:.2f}, skew {m['skewness']:.2f}, "
                     f"JB p={m['jarque_bera_p']:.1e}", fontsize=9)
        ax.set_xlabel("Theoretical N(0,1) quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "qq.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── NIS timeline ──────────────────────────────────────────────────────
    lo2, hi2 = stats.chi2.ppf([0.025, 0.975], 2)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(dates[keep], nis[keep], s=4, color="#2271B2", alpha=0.6, label="NIS (joint, χ²₂)")
    ax.axhline(lo2, color="firebrick", lw=1, ls="--", label="95% χ²₂ band")
    ax.axhline(hi2, color="firebrick", lw=1, ls="--")
    ax.set_yscale("log")
    in_hv = False
    for d, flag in hi_vol.items():
        if flag and not in_hv:
            start, in_hv = d, True
        elif not flag and in_hv:
            ax.axvspan(start, d, color="orange", alpha=0.15, lw=0)
            in_hv = False
    if in_hv:
        ax.axvspan(start, dates[-1], color="orange", alpha=0.15, lw=0)
    for d, name in EVENTS.items():
        ax.axvline(pd.Timestamp(d), color="grey", lw=0.8, ls=":")
        ax.text(pd.Timestamp(d), ax.get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=7)
    ax.axvline(dates[split], color="black", lw=1, label="OOS start (70%)")
    ax.plot([], [], color="orange", alpha=0.3, lw=8, label="top-decile 30d realised vol")
    ax.set_title("Normalised innovation squared, base 5-state UKF, BTC-USD "
                 f"{dates[0].date()} – {dates[-1].date()}")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "nis_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Markdown summary ─────────────────────────────────────────────────
    c1, c2, cr = summary["channel1_return"]["all"], summary["channel2_logr2"]["all"], summary["readme_style_residual"]["all"]
    nj, n1, n2 = summary["nis_joint"]["all"], summary["nis_channel1"]["all"], summary["nis_channel2"]["all"]
    hv = summary["high_vol_regime"]
    lines = [
        f"# E0: UKF innovation diagnostics ({summary['window']['start']} – {summary['window']['end']}, "
        f"n={n}, burn-in {BURN_IN})",
        "",
        "| Series | skew | excess kurt | JB stat | JB p |",
        "|---|---|---|---|---|",
        f"| Channel 1 (return) | {c1['skewness']:.2f} | {c1['excess_kurtosis']:.2f} | {c1['jarque_bera_stat']:.0f} | {c1['jarque_bera_p']:.1e} |",
        f"| Channel 2 (log r²) | {c2['skewness']:.2f} | {c2['excess_kurtosis']:.2f} | {c2['jarque_bera_stat']:.0f} | {c2['jarque_bera_p']:.1e} |",
        "| *log χ²(1) reference (correct model, ch. 2)* | *-1.54* | *4.00* | | |",
        f"| README-style residual | {cr['skewness']:.2f} | {cr['excess_kurtosis']:.2f} | {cr['jarque_bera_stat']:.0f} | {cr['jarque_bera_p']:.1e} |",
        "",
        "| NIS | mean (expected) | outside 95% two-sided band (expected 5%) | below / above (2.5% each) | binomial p below / above |",
        "|---|---|---|---|---|",
        *[f"| {lab} | {d['mean_nis']:.2f} ({d['dof']}) | {d['frac_outside_two_sided']:.1%} | "
          f"{d['frac_below_lower']:.1%} / {d['frac_above_upper']:.1%} | {d['binom_p_below']:.1e} / {d['binom_p_above']:.1e} |"
          for lab, d in [("Joint χ²₂", nj), ("Channel 1 χ²₁", n1), ("Channel 2 χ²₁", n2)]],
        "",
        f"Joint NIS outside band: {hv['frac_outside_joint_band_in_high_vol']:.1%} in top-decile realised-vol days "
        f"vs {hv['frac_outside_joint_band_elsewhere']:.1%} elsewhere.",
    ]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    import filterpy, matplotlib, scipy, yfinance
    with open(os.path.join(OUT, "environment.txt"), "w") as f:
        f.write(f"python {platform.python_version()}\nnumpy {np.__version__}\npandas {pd.__version__}\n"
                f"scipy {scipy.__version__}\nfilterpy {filterpy.__version__}\n"
                f"matplotlib {matplotlib.__version__}\nyfinance {yfinance.__version__}\n"
                f"data cache {CACHE}\n")

    print("\n".join(lines))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
