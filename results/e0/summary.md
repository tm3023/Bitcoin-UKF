# E0: UKF innovation diagnostics (2021-10-03 – 2026-10-02, n=1826, burn-in 50)

| Series | skew | excess kurt | JB stat | JB p |
|---|---|---|---|---|
| Channel 1 (return) | 0.17 | 2.22 | 375 | 3.7e-82 |
| Channel 2 (log r²) | -0.82 | 0.75 | 239 | 1.3e-52 |
| *log χ²(1) reference (correct model, ch. 2)* | *-1.54* | *4.00* | | |
| README-style residual | 0.18 | 1.85 | 263 | 8.9e-58 |

| NIS | mean (expected) | outside 95% two-sided band (expected 5%) | below / above (2.5% each) | binomial p below / above |
|---|---|---|---|---|
| Joint χ²₂ | 2.23 (2) | 8.8% | 2.8% / 6.0% | 4.0e-01 / 1.6e-15 |
| Channel 1 χ²₁ | 1.03 (1) | 7.3% | 3.3% / 4.0% | 4.8e-02 / 1.8e-04 |
| Channel 2 χ²₁ | 1.20 (1) | 7.3% | 2.4% / 4.9% | 8.2e-01 / 7.8e-09 |

Joint NIS outside band: 10.6% in top-decile realised-vol days vs 8.6% elsewhere.
