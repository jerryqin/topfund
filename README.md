# topfund
Analyze ETF / mutual fund data to find the **TOP FUND** to invest in.

## How it works

`topfund.py` downloads historical price data from Yahoo Finance and ranks each
fund using four performance metrics:

| Metric | Weight | Direction |
|---|---|---|
| Sharpe ratio (risk-adjusted return) | 40 % | higher is better |
| Annualised return | 30 % | higher is better |
| Maximum drawdown | 20 % | less negative is better |
| Annualised volatility | 10 % | lower is better |

Each metric is converted to a normalised rank, then combined into a single
**composite score** (0–1). The fund with the highest score is the #1 pick.

## Requirements

```
pip install -r requirements.txt
```

## Usage

```
# Analyse the built-in universe of 20 popular ETFs over the past year
python topfund.py

# Analyse a custom list of tickers over a 3-year window, show top 3
python topfund.py --tickers SPY QQQ GLD AGG TLT --period 3y --top 3
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--tickers` | built-in 20-ETF universe | Space-separated ticker symbols |
| `--period` | `1y` | `3mo` `6mo` `1y` `2y` `3y` `5y` |
| `--top` | `5` | Number of top funds to display |

### Example output

```
Fetching data for period=1y …

=================================================================
  TOP FUND RANKINGS
=================================================================
Ticker    Ann. Return  Ann. Volatility  Sharpe Ratio  Max Drawdown  Score
QQQ           28.45%            18.2%          1.56       -10.34%  0.872
XLK           26.10%            19.5%          1.34       -11.80%  0.801
SPY           18.32%            14.8%          1.24        -8.45%  0.762
VUG           22.15%            17.1%          1.30       -10.10%  0.748
VTI           16.90%            14.6%          1.16        -8.60%  0.703
=================================================================

🏆  #1 TOP FUND: QQQ
    Ann. Return: 28.45% | Sharpe: 1.56 | Max DD: -10.34%
```

## Running tests

```
pip install pytest
pytest test_topfund.py -v
```
