"""
Data loader for BTC daily log-returns.

Set USE_REAL_DATA = True to download live BTC-USD prices via yfinance.
Set USE_REAL_DATA = False to use a synthetic series calibrated to BTC empirics.

Both paths return (r, h_true) where h_true is the latent log-variance array
(only available for simulated data; None for real data).
"""

import numpy as np

USE_REAL_DATA = True  # Toggle here to switch between simulated and live data


def simulate_btc_returns(n=1100):
    """
    Stochastic volatility model calibrated to BTC-USD daily data.
    Two engineered high-vol regimes (days 300-380, 700-760) test regime tracking.
    """
    np.random.seed(7)
    mu_h  = 2 * np.log(0.035)
    phi   = 0.97
    sig_h = 0.18

    h = np.zeros(n)
    h[0] = mu_h
    for t in range(1, n):
        h[t] = mu_h + phi * (h[t - 1] - mu_h) + sig_h * np.random.randn()

    regime = np.zeros(n)
    regime[300:380] = 1.5
    regime[700:760] = 1.2
    h += regime

    eps   = np.random.randn(n)
    jumps = (np.random.rand(n) < 0.02) * np.random.randn(n) * 0.06
    r = np.exp(h / 2) * eps + jumps
    r += 0.0008
    return r, h


def load_returns(use_real=USE_REAL_DATA, n=1100):
    """
    Load BTC daily log-returns.

    Parameters
    ----------
    use_real : bool
        True  → download BTC-USD from Yahoo Finance via yfinance (last 5 years)
        False → generate synthetic series (reproducible, seed=7)
    n : int
        Number of observations for simulated data (ignored for real data).

    Returns
    -------
    r : np.ndarray  — daily log-returns
    h_true : np.ndarray or None  — true log-variance (None for real data)
    """
    if use_real:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required for real data: pip install yfinance")

        print("Downloading BTC-USD daily data from Yahoo Finance...")
        btc = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
        prices = btc["Close"].squeeze().dropna().values.astype(float)
        r = np.diff(np.log(prices))
        print(f"Downloaded {len(r)} daily log-returns (BTC-USD, 5-year history)")
        return r, None

    return simulate_btc_returns(n)


def load_returns_with_dates(use_real=USE_REAL_DATA, n=1100):
    """
    Like load_returns(), but also returns the calendar date for each return.

    Returns
    -------
    r      : np.ndarray            — daily log-returns
    h_true : np.ndarray or None    — true log-variance (None for real data)
    dates  : pandas DatetimeIndex or None — date of each return; None for sim
    """
    if use_real:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for real data: pip install yfinance"
            )
        print("Downloading BTC-USD daily data from Yahoo Finance...")
        btc    = yf.download("BTC-USD", period="5y", interval="1d",
                             progress=False)
        prices = btc["Close"].squeeze().dropna()
        dates  = prices.index[1:]                      # date of each return
        r      = np.diff(np.log(prices.values.astype(float)))
        print(f"Downloaded {len(r)} daily log-returns "
              f"({dates[0].date()} → {dates[-1].date()})")
        return r, None, dates

    r, h_true = simulate_btc_returns(n)
    return r, h_true, None


def load_funding_rates(dates, symbol="BTCUSDT"):
    """
    Download BTC perpetual funding rates from Binance FAPI and aggregate
    to a daily average (funding is published every 8 h at 00:00 / 08:00 /
    16:00 UTC, giving three observations per calendar day).

    Uses the same dates as the BTC return series for alignment.  Missing
    dates are forward-filled with the most recent known rate; days before
    Binance perpetuals existed are zero-filled.

    Parameters
    ----------
    dates  : pandas DatetimeIndex   — dates from load_returns_with_dates()
    symbol : str                    — Binance perp symbol (default BTCUSDT)

    Returns
    -------
    np.ndarray of shape (len(dates),)
        Daily average funding rate in decimal (e.g. 0.0001 = 0.01 %).
        Returns all zeros and prints a warning if the API is unreachable.

    Data source
    -----------
    Binance FAPI public endpoint (no API key required):
        GET https://fapi.binance.com/fapi/v1/fundingRate
    Covers BTCUSDT perpetual from 2019-09-10 onwards.
    """
    try:
        import requests
        import pandas as pd
        from datetime import datetime, timezone
    except ImportError:
        print("Warning: 'requests' not installed — funding filter disabled. "
              "pip install requests")
        return np.zeros(len(dates))

    print(f"Downloading {symbol} funding rates from Binance FAPI …")

    start_date = dates[0].date()
    end_date   = dates[-1].date()
    start_ms   = int(datetime(start_date.year, start_date.month,
                              start_date.day,
                              tzinfo=timezone.utc).timestamp() * 1000)
    end_ms     = int(datetime(end_date.year, end_date.month,
                              end_date.day, 23, 59,
                              tzinfo=timezone.utc).timestamp() * 1000)

    url     = "https://fapi.binance.com/fapi/v1/fundingRate"
    records = []
    cur     = start_ms
    try:
        while cur <= end_ms:
            resp = requests.get(
                url,
                params={"symbol": symbol, "startTime": cur, "limit": 1000},
                timeout=15,
            )
            data = resp.json()
            if not isinstance(data, list) or not data:
                break
            records.extend(data)
            cur = int(data[-1]["fundingTime"]) + 1
            if len(data) < 1000:
                break
    except Exception as exc:
        print(f"Warning: Binance API error ({exc}) — funding filter disabled.")
        return np.zeros(len(dates))

    if not records:
        print("Warning: no funding data returned — funding filter disabled.")
        return np.zeros(len(dates))

    df = pd.DataFrame(records)
    df["date"] = (pd.to_datetime(df["fundingTime"].astype(int),
                                 unit="ms", utc=True)
                  .dt.date)
    df["fundingRate"] = df["fundingRate"].astype(float)
    daily_fr = df.groupby("date")["fundingRate"].mean()

    print(f"  {len(records)} funding records → {len(daily_fr)} daily averages "
          f"({daily_fr.index[0]} → {daily_fr.index[-1]})")

    # Align to return dates; forward-fill gaps; zero-fill the distant past
    result     = np.zeros(len(dates))
    last_known = 0.0
    for i, d in enumerate([dt.date() for dt in dates]):
        if d in daily_fr.index:
            last_known = float(daily_fr[d])
        result[i] = last_known

    return result
