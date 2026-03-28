"""
topfund - Analyze fund/ETF data to find the TOP FUND to invest in.

Usage:
    python topfund.py [--tickers TICKER ...] [--period PERIOD] [--top N]

Metrics evaluated:
    - Annualized return
    - Sharpe ratio (risk-adjusted return, assumes 0% risk-free rate)
    - Maximum drawdown
    - Annualized volatility
    - Composite score (weighted combination of the above)
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

# Default universe of popular ETFs/funds across major categories
DEFAULT_TICKERS = [
    # US equity
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq-100
    "IWM",   # Russell 2000 (small-cap)
    "VTI",   # Total US market
    "VUG",   # US growth
    "VTV",   # US value
    # International equity
    "EFA",   # Developed markets ex-US
    "EEM",   # Emerging markets
    "VEA",   # Developed markets (Vanguard)
    # Sector ETFs
    "XLK",   # Technology
    "XLV",   # Healthcare
    "XLF",   # Financials
    "XLE",   # Energy
    "XLY",   # Consumer discretionary
    # Fixed income
    "AGG",   # US aggregate bond
    "BND",   # Vanguard total bond
    "TLT",   # Long-term Treasury
    # Alternatives / multi-asset
    "GLD",   # Gold
    "VNQ",   # Real estate
    "ARKK",  # Innovation (active)
]

TRADING_DAYS_PER_YEAR = 252


def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Download adjusted closing prices for *tickers* over *period*.

    Returns a DataFrame indexed by date with one column per ticker containing
    the adjusted closing price.  Tickers for which no data is available are
    silently dropped.

    Raises:
        ValueError: If Yahoo Finance returns no data for any of the tickers.
    """
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("No data returned from Yahoo Finance.")
    # yfinance returns MultiIndex columns when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    # Drop tickers with no data
    prices = prices.dropna(axis=1, how="all")
    return prices


def compute_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-fund performance metrics from a price DataFrame.

    Returns a DataFrame indexed by ticker with columns:
        annualized_return, annualized_volatility, sharpe_ratio, max_drawdown
    """
    daily_returns = prices.pct_change().dropna()

    # Annualized return (geometric)
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    n_days = len(daily_returns)
    annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1

    # Annualized volatility
    annualized_vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe ratio (risk-free rate = 0 for simplicity)
    sharpe = annualized_return / annualized_vol

    # Maximum drawdown
    def max_drawdown(price_series: pd.Series) -> float:
        rolling_max = price_series.cummax()
        drawdown = (price_series - rolling_max) / rolling_max
        return float(drawdown.min())

    max_dd = prices.apply(max_drawdown)

    metrics = pd.DataFrame(
        {
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }
    )
    metrics.index.name = "ticker"
    return metrics


def score_funds(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite score for each fund using rank-based normalization.

    Higher is better for: annualized_return, sharpe_ratio
    Lower (less negative) is better for: max_drawdown, annualized_volatility

    Weights:
        sharpe_ratio        40 %
        annualized_return   30 %
        max_drawdown        20 %
        annualized_volatility 10 %

    Returns a DataFrame that includes all columns from *metrics* plus four
    intermediate rank columns (``rank_sharpe``, ``rank_return``,
    ``rank_drawdown``, ``rank_vol``) and the final ``composite_score``
    column (float in [0, 1]).
    """
    df = metrics.copy()
    n = len(df)

    # Rank each metric (1 = best)
    df["rank_sharpe"] = df["sharpe_ratio"].rank(ascending=False)
    df["rank_return"] = df["annualized_return"].rank(ascending=False)
    df["rank_drawdown"] = df["max_drawdown"].rank(ascending=False)   # less negative = better
    df["rank_vol"] = df["annualized_volatility"].rank(ascending=True)  # lower vol = better

    # Normalise ranks to [0, 1] where 1 is best
    for col in ["rank_sharpe", "rank_return", "rank_drawdown", "rank_vol"]:
        df[col] = 1 - (df[col] - 1) / max(n - 1, 1)

    df["composite_score"] = (
        0.40 * df["rank_sharpe"]
        + 0.30 * df["rank_return"]
        + 0.20 * df["rank_drawdown"]
        + 0.10 * df["rank_vol"]
    )
    return df


def find_top_funds(
    tickers: list[str] | None = None,
    period: str = "1y",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Main entry point: fetch data, compute metrics, rank, return top *top_n* funds.

    Returns a sorted DataFrame (best first) with all metrics and composite score.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    prices = fetch_prices(tickers, period=period)
    metrics = compute_metrics(prices)
    scored = score_funds(metrics)
    ranked = scored.sort_values("composite_score", ascending=False)
    return ranked.head(top_n)


def print_results(results: pd.DataFrame) -> None:
    """Pretty-print the ranked fund table."""
    display = results[
        ["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "composite_score"]
    ].copy()

    display["annualized_return"] = display["annualized_return"].map("{:.2%}".format)
    display["annualized_volatility"] = display["annualized_volatility"].map("{:.2%}".format)
    display["sharpe_ratio"] = display["sharpe_ratio"].map("{:.2f}".format)
    display["max_drawdown"] = display["max_drawdown"].map("{:.2%}".format)
    display["composite_score"] = display["composite_score"].map("{:.3f}".format)

    display.columns = [
        "Ann. Return",
        "Ann. Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Score",
    ]
    display.index.name = "Ticker"

    print("\n" + "=" * 65)
    print("  TOP FUND RANKINGS")
    print("=" * 65)
    print(display.to_string())
    print("=" * 65)
    print(f"\n🏆  #1 TOP FUND: {results.index[0]}")
    print(
        f"    Ann. Return: {float(results['annualized_return'].iloc[0]):.2%} | "
        f"Sharpe: {float(results['sharpe_ratio'].iloc[0]):.2f} | "
        f"Max DD: {float(results['max_drawdown'].iloc[0]):.2%}"
    )
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the TOP FUND to invest in by analyzing ETF/fund performance."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help="Space-separated list of fund tickers to analyze (default: built-in universe).",
    )
    parser.add_argument(
        "--period",
        default="1y",
        choices=["3mo", "6mo", "1y", "2y", "3y", "5y"],
        help="Historical period to analyse (default: 1y).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        dest="top_n",
        help="Number of top funds to display (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"Fetching data for period={args.period} …")
    try:
        results = find_top_funds(
            tickers=args.tickers,
            period=args.period,
            top_n=args.top_n,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print_results(results)


if __name__ == "__main__":
    main()
