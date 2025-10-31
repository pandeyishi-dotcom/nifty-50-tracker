"""
nifty50_tracker.py

Requirements:
    pip install yfinance pandas numpy requests

What it does:
    - Attempts to fetch current NIFTY 50 constituents from NSE (fallback to embedded list).
    - Downloads price & fundamental data via yfinance.
    - Computes weights, volatility, efficiency (mean return / std), beta vs NIFTY index, correlation.
    - Estimates price impact when NIFTY moves by x points (x in [100,200,250,300,500]).
    - Flags each stock as gainer/loser under each scenario.
    - Aggregates sector-wise holdings.
    - Exports results to CSV.

Usage:
    python nifty50_tracker.py
"""

import time
import math
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ---------------------------
# User parameters (changeable)
# ---------------------------
HIST_DAYS = 365  # history window for returns & beta
SLIPPAGE_PCT = 0.001  # assumed slippage (0.001 = 0.1%)
LEVERAGE = 1.0  # user leverage factor, e.g., 2.0 for 2x
INDEX_TICKER = "^NSEI"  # Yahoo ticker for NIFTY 50 index
POINT_MOVES = [100, 200, 250, 300, 500]  # index point shocks to test
OUTPUT_CSV = "nifty50_analysis.csv"
# ---------------------------


def fetch_nifty50_constituents_from_nse():
    """
    Tries to fetch NIFTY 50 constituents from NSE India API.
    NSE blocks some scrapers; this is a best-effort attempt.
    Returns list of symbol strings (without .NS suffix).
    """
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; tracker/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    s = requests.Session()
    try:
        # First get landing page to obtain cookies
        _ = s.get("https://www.nseindia.com", headers=headers, timeout=10)
        r = s.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        symbols = [row["symbol"] for row in data["data"]]
        return symbols
    except Exception as e:
        print(f"[warning] NSE fetch failed: {e}. Falling back to built-in list.")
        return None


def get_fallback_nifty50_list():
    """
    Reasonably complete fallback list of NIFTY50 tickers (as of recent years).
    The tickers below are Yahoo-style symbols WITHOUT the .NS suffix; we'll append .NS
    This list may need manual maintenance - it's a fallback only.
    """
    tickers = [
        "ADANIENT","AMBUJACEM","APOLLOHOSP","ASIANPAINT","AXISBANK",
        "BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BHARTIARTL","BPCL",
        "BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY",
        "EICHERMOT","GRASIM","HCLTECH","HDFC","HDFCBANK",
        "HDFCLIFE","HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK",
        "INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK",
        "LT","LICHSGFIN","M&M","MARUTI","NESTLEIND",
        "NTPC","ONGC","POWERGRID","RELIANCE","SBILIFE",
        "SBIN","SHREECEM","SUNPHARMA","TATAMOTORS","TATASTEEL",
        "TCS","TECHM","TITAN","ULTRACEMCO","WIPRO"
    ]
    return tickers


def make_yahoo_symbols(tickers):
    # append .NS
    return [t + ".NS" for t in tickers]


def fetch_yahoo_data(tickers_ns, hist_days=HIST_DAYS):
    """
    Batch download of daily price history and per-ticker info via yfinance.
    Returns:
        - price_df: DataFrame of adjusted close prices (columns = tickers)
        - info_map: dict symbol->info (from yfinance.Ticker.info)
        - last_price: dict symbol->last_price
        - last_volume: dict symbol->last_volume
    """
    end = datetime.today()
    start = end - timedelta(days=hist_days + 30)
    print(f"[info] Downloading historical prices from {start.date()} to {end.date()}")
    # yfinance allows a list; it returns multiindex columns when group_by='ticker' omitted
    price_data = yf.download(tickers_ns, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, threads=True, group_by='column', auto_adjust=False)
    # price_data may have single-level columns for single ticker; normalize
    if isinstance(price_data.columns, pd.MultiIndex):
        # Extract Adj Close per ticker
        adj_close = {}
        volumes = {}
        for ticker in tickers_ns:
            try:
                adj_close[ticker] = price_data[ticker]['Adj Close']
                volumes[ticker] = price_data[ticker]['Volume']
            except Exception:
                # some tickers might be missing
                print(f"[warn] missing data for {ticker}")
        price_df = pd.DataFrame(adj_close)
        volume_df = pd.DataFrame(volumes)
    else:
        # Single ticker case
        price_df = pd.DataFrame({tickers_ns[0]: price_data['Adj Close']})
        volume_df = pd.DataFrame({tickers_ns[0]: price_data['Volume']})

    # Info and last values per ticker
    info_map = {}
    last_price = {}
    last_volume = {}
    for t in tickers_ns:
        tk = yf.Ticker(t)
        try:
            info = tk.info
        except Exception:
            info = {}
        info_map[t] = info
        # last available price from price_df
        try:
            last_price[t] = float(price_df[t].dropna().iloc[-1])
        except Exception:
            last_price[t] = np.nan
        try:
            last_volume[t] = int(volume_df[t].dropna().iloc[-1])
        except Exception:
            last_volume[t] = np.nan

    return price_df, info_map, last_price, last_volume


def compute_metrics(price_df, info_map, last_price_map, last_volume_map, index_series):
    """
    Compute returns, vol, mean return, efficiency, beta, correlation, marketcap, weight, EPS, sector.
    Returns results DataFrame indexed by ticker (with .NS) and sector aggregation.
    """
    # daily returns
    returns = price_df.pct_change().dropna(how='all')

    index_ret = index_series.pct_change().dropna()
    index_last = index_series.dropna().iloc[-1]

    rows = []
    tickers = list(price_df.columns)
    for t in tickers:
        series = price_df[t].dropna()
        if len(series) < 10:
            print(f"[warn] insufficient price history for {t}")
            mean_ret = np.nan
            vol = np.nan
        else:
            rets = series.pct_change().dropna()
            mean_ret = rets.mean() * 252  # annualized mean return approx
            vol = rets.std() * math.sqrt(252)  # annualized volatility

        # Efficiency: mean_return / vol (Sharpe-like without rf)
        efficiency = mean_ret / vol if (not np.isnan(mean_ret) and not np.isnan(vol) and vol != 0) else np.nan

        # Covariance and beta with index (use overlapping index)
        try:
            combined = pd.concat([returns[t], index_ret], axis=1).dropna()
            cov = combined.cov().iloc[0,1]
            var_index = combined[1].var()
            beta = cov / var_index if var_index != 0 else np.nan
            corr = combined.corr().iloc[0,1]
        except Exception:
            beta = np.nan
            corr = np.nan

        info = info_map.get(t, {})
        marketcap = info.get("marketCap", np.nan)
        eps = info.get("trailingEps", info.get("epsTrailingTwelveMonths", np.nan))
        sector = info.get("sector", info.get("industry", np.nan))  # fallback to industry if sector missing
        name = info.get("shortName", info.get("longName", t))

        price = last_price_map.get(t, np.nan)
        volume = last_volume_map.get(t, np.nan)

        rows.append({
            "symbol": t,
            "name": name,
            "price": price,
            "marketCap": marketcap,
            "volume": volume,
            "eps": eps,
            "sector": sector,
            "mean_return_annual": mean_ret,
            "volatility_annual": vol,
            "efficiency": efficiency,
            "beta": beta,
            "correlation": corr
        })

    df = pd.DataFrame(rows).set_index("symbol")
    # compute weights by marketCap (drop nans)
    total_marketcap = df['marketCap'].dropna().sum()
    df['weight'] = df['marketCap'] / total_marketcap
    df['%capital'] = df['weight'] * 100
    # risk classification (simple buckets by volatility)
    df['risk_label'] = pd.qcut(df['volatility_annual'].rank(method='first'), q=4, labels=['Very Low','Low','Moderate','High'])
    # last updated time
    df['last_updated'] = datetime.now().isoformat(timespec='seconds')
    # add index last level
    df.attrs['index_last'] = index_last
    return df


def compute_impacts(df, index_last, point_moves=POINT_MOVES, slippage_pct=SLIPPAGE_PCT, leverage=LEVERAGE):
    """
    For each point move in point_moves, compute expected % change and price change using beta.
    Decide gainer/loser by positive/negative price_change.
    Add columns for each move like: impact_100_pts_price_change, impact_100_pts_new_price, impact_100_pts_gainer
    """
    for pts in point_moves:
        idx_pct = pts / index_last  # approximate index % move
        col_price_change = f"impact_{pts}_price_change"
        col_new_price = f"impact_{pts}_new_price"
        col_pct = f"impact_{pts}_pct_change"
        col_gainer = f"impact_{pts}_gainer"

        price_changes = []
        new_prices = []
        pct_changes = []
        gainers = []
        for sym, row in df.iterrows():
            beta = row['beta']
            price = row['price']
            if np.isnan(beta) or np.isnan(price):
                price_change = np.nan
                new_price = np.nan
                pct_change = np.nan
                gainer = np.nan
            else:
                expected_pct = beta * idx_pct
                price_change = expected_pct * price
                new_price = price + price_change
                pct_change = expected_pct * 100
                gainer = "Gainer" if price_change > 0 else ("Loser" if price_change < 0 else "Neutral")

            price_changes.append(price_change)
            new_prices.append(new_price)
            pct_changes.append(pct_change)
            gainers.append(gainer)

        df[col_price_change] = price_changes
        df[col_new_price] = new_prices
        df[col_pct] = pct_changes
        df[col_gainer] = gainers

    # slippage cost estimate per trade (assuming slippage_pct fraction of price)
    df['slippage_cost_per_share'] = df['price'] * slippage_pct
    # leverage exposure multiplier
    df['leverage'] = leverage
    # implied exposure per 1 unit capital assuming you allocate based on weight:
    # if you have $1 capital, the position in each symbol = weight * leverage
    df['exposure_fraction_per_1_unit_capital'] = df['weight'] * leverage

    return df


def sector_aggregation(df):
    """
    Aggregates weights and counts by sector.
    """
    sector_df = df.groupby('sector').agg({
        'weight': 'sum',
        'marketCap': 'sum',
        'symbol': lambda x: ','.join(x.index.tolist())  # not perfect but informative
    }).rename(columns={'weight': 'sector_weight', 'marketCap': 'sector_marketCap'})
    # convert weight to %
    sector_df['sector_weight_%'] = sector_df['sector_weight'] * 100
    sector_df = sector_df.sort_values('sector_weight_%', ascending=False)
    return sector_df


def rank_and_sort(df):
    """
    Add rank columns: by efficiency, by marketCap, by weight.
    """
    df['rank_efficiency'] = df['efficiency'].rank(ascending=False, method='min')
    df['rank_marketcap'] = df['marketCap'].rank(ascending=False, method='min')
    df['rank_weight'] = df['weight'].rank(ascending=False, method='min')
    # Efficiency percentile
    df['eff_percentile'] = df['efficiency'].rank(pct=True)
    return df.sort_values('rank_efficiency')


def main():
    # 1) Get constituents
    symbols = fetch_nifty50_constituents_from_nse()
    if symbols is None:
        base = get_fallback_nifty50_list()
        print(f"[info] using fallback list of {len(base)} tickers")
        symbols = base
    # make yahoo symbols with .NS
    yahoo_symbols = make_yahoo_symbols(symbols)

    # 2) Download data for tickers and index
    price_df, info_map, last_price_map, last_volume_map = fetch_yahoo_data(yahoo_symbols, hist_days=HIST_DAYS)
    # fetch index series (close prices)
    idx_hist = yf.download(INDEX_TICKER, period=f"{HIST_DAYS+30}d", interval="1d", progress=False, auto_adjust=False)
    index_series = idx_hist['Adj Close']
    if index_series.dropna().empty:
        raise RuntimeError("Failed to download index historical series. Check network or ticker.")

    # 3) Compute metrics
    df = compute_metrics(price_df, info_map, last_price_map, last_volume_map, index_series)

    # 4) Impact calculations
    df = compute_impacts(df, index_last=index_series.dropna().iloc[-1], point_moves=POINT_MOVES, slippage_pct=SLIPPAGE_PCT, leverage=LEVERAGE)

    # 5) Sector aggregation
    sector_df = sector_aggregation(df)

    # 6) Ranking and sorting
    df = rank_and_sort(df)

    # 7) Output
    print("\n=== Top 10 by Efficiency (return/volatility) ===")
    display_cols = ['name', 'price', 'marketCap', '%capital', 'volume', 'eps', 'sector', 'mean_return_annual', 'volatility_annual', 'efficiency', 'beta', 'correlation']
    print(df[display_cols].head(10))

    print("\n=== Sector-wise weight (%) ===")
    print(sector_df[['sector_weight_%','sector_marketCap']].head(20))

    # Save to CSV
    df_to_save = df.copy()
    # Flatten columns to safe names if needed
    df_to_save.to_csv(OUTPUT_CSV)
    print(f"\n[saved] Detailed results exported to {OUTPUT_CSV}")

    # Also print a small summary table for each shock scenario
    for pts in POINT_MOVES:
        col_price_change = f"impact_{pts}_price_change"
        col_new_price = f"impact_{pts}_new_price"
        col_gainer = f"impact_{pts}_gainer"
        summary = df[[col_price_change, col_new_price, col_gainer, 'price', 'weight', '%capital']].copy()
        # Top 5 potential absolute winners & losers by absolute price change
        summary['abs_change'] = summary[col_price_change].abs()
        top5 = summary.sort_values('abs_change', ascending=False).head(5)
        print(f"\n--- Top 5 absolute movers if NIFTY moves {pts} points ---")
        print(top5[[col_price_change, col_new_price, 'price', 'abs_change']])

    # Return objects in case this module is imported
    return df, sector_df


if __name__ == "__main__":
    df_out, sector_out = main()
