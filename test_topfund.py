"""Tests for topfund.py – uses synthetic price data so no network calls are needed."""

import numpy as np
import pandas as pd
import pytest

from topfund import (
    DEFAULT_TICKERS,
    TRADING_DAYS_PER_YEAR,
    compute_metrics,
    find_top_funds,
    parse_args,
    print_results,
    score_funds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(tickers: list[str], n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic price series with different drift rates per ticker."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    data = {}
    for i, ticker in enumerate(tickers):
        # Each ticker gets a slightly different drift so rankings are deterministic
        drift = 0.0005 * (i + 1)
        noise = rng.normal(drift, 0.01, n_days)
        prices = 100 * np.exp(np.cumsum(noise))
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_expected_columns(self):
        prices = _make_prices(["A", "B"])
        metrics = compute_metrics(prices)
        expected_cols = {"annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"}
        assert expected_cols == set(metrics.columns)

    def test_index_matches_tickers(self):
        tickers = ["A", "B", "C"]
        prices = _make_prices(tickers)
        metrics = compute_metrics(prices)
        assert list(metrics.index) == tickers

    def test_max_drawdown_is_non_positive(self):
        prices = _make_prices(["X", "Y", "Z"])
        metrics = compute_metrics(prices)
        assert (metrics["max_drawdown"] <= 0).all()

    def test_volatility_is_positive(self):
        prices = _make_prices(["X", "Y"])
        metrics = compute_metrics(prices)
        assert (metrics["annualized_volatility"] > 0).all()

    def test_higher_drift_yields_higher_return(self):
        """Ticker with higher drift should have a higher annualised return."""
        prices = _make_prices(["LOW", "HIGH"], seed=0)
        # Manually create deterministic prices: LOW flat, HIGH trending up
        dates = pd.bdate_range("2023-01-01", periods=252)
        low = pd.Series(100.0, index=dates)
        high = pd.Series(
            100 * (1 + 0.001) ** np.arange(252), index=dates
        )
        prices = pd.DataFrame({"LOW": low, "HIGH": high})
        metrics = compute_metrics(prices)
        assert metrics.loc["HIGH", "annualized_return"] > metrics.loc["LOW", "annualized_return"]

    def test_single_ticker(self):
        prices = _make_prices(["SOLO"])
        metrics = compute_metrics(prices)
        assert len(metrics) == 1

    def test_sharpe_positive_for_positive_return(self):
        dates = pd.bdate_range("2023-01-01", periods=252)
        prices = pd.DataFrame(
            {"UP": 100 * (1 + 0.0008) ** np.arange(252)}, index=dates
        )
        metrics = compute_metrics(prices)
        assert metrics.loc["UP", "sharpe_ratio"] > 0


# ---------------------------------------------------------------------------
# score_funds
# ---------------------------------------------------------------------------

class TestScoreFunds:
    def test_composite_score_in_0_1(self):
        prices = _make_prices(["A", "B", "C", "D"])
        metrics = compute_metrics(prices)
        scored = score_funds(metrics)
        assert (scored["composite_score"] >= 0).all()
        assert (scored["composite_score"] <= 1).all()

    def test_best_fund_has_highest_score(self):
        """Fund with high return & low volatility should rank first."""
        dates = pd.bdate_range("2023-01-01", periods=252)
        np.random.seed(7)
        # BEST: strong up-trend, low noise
        best = pd.Series(100 * (1 + 0.002) ** np.arange(252), index=dates)
        # WORST: flat with high noise
        worst = 100 + pd.Series(np.random.normal(0, 5, 252), index=dates).cumsum()
        worst = worst.clip(lower=1)
        prices = pd.DataFrame({"BEST": best, "WORST": worst})
        metrics = compute_metrics(prices)
        scored = score_funds(metrics)
        assert scored["composite_score"]["BEST"] > scored["composite_score"]["WORST"]

    def test_all_rank_columns_present(self):
        prices = _make_prices(["A", "B"])
        metrics = compute_metrics(prices)
        scored = score_funds(metrics)
        for col in ["rank_sharpe", "rank_return", "rank_drawdown", "rank_vol"]:
            assert col in scored.columns

    def test_single_fund_score_is_one(self):
        prices = _make_prices(["SOLO"])
        metrics = compute_metrics(prices)
        scored = score_funds(metrics)
        assert pytest.approx(scored.loc["SOLO", "composite_score"], abs=1e-9) == 1.0


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.tickers is None
        assert args.period == "1y"
        assert args.top_n == 5

    def test_custom_tickers(self):
        args = parse_args(["--tickers", "SPY", "QQQ"])
        assert args.tickers == ["SPY", "QQQ"]

    def test_custom_period(self):
        args = parse_args(["--period", "3y"])
        assert args.period == "3y"

    def test_custom_top(self):
        args = parse_args(["--top", "3"])
        assert args.top_n == 3


# ---------------------------------------------------------------------------
# find_top_funds (with monkey-patched fetch_prices)
# ---------------------------------------------------------------------------

class TestFindTopFunds:
    def test_returns_correct_number_of_rows(self, monkeypatch):
        tickers = ["A", "B", "C", "D", "E", "F"]
        fake_prices = _make_prices(tickers)

        import topfund
        monkeypatch.setattr(topfund, "fetch_prices", lambda t, period: fake_prices)

        result = find_top_funds(tickers=tickers, top_n=3)
        assert len(result) == 3

    def test_result_sorted_by_score(self, monkeypatch):
        tickers = ["A", "B", "C", "D"]
        fake_prices = _make_prices(tickers)

        import topfund
        monkeypatch.setattr(topfund, "fetch_prices", lambda t, period: fake_prices)

        result = find_top_funds(tickers=tickers, top_n=4)
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_uses_default_tickers_when_none(self, monkeypatch):
        fake_prices = _make_prices(DEFAULT_TICKERS[:5])

        import topfund
        captured = {}

        def mock_fetch(tickers, period):
            captured["tickers"] = tickers
            return fake_prices

        monkeypatch.setattr(topfund, "fetch_prices", mock_fetch)
        find_top_funds(tickers=None, top_n=3)
        assert captured["tickers"] == DEFAULT_TICKERS


# ---------------------------------------------------------------------------
# print_results (smoke test)
# ---------------------------------------------------------------------------

class TestPrintResults:
    def test_runs_without_error(self, monkeypatch, capsys):
        tickers = ["A", "B", "C"]
        prices = _make_prices(tickers)
        metrics = compute_metrics(prices)
        scored = score_funds(metrics).sort_values("composite_score", ascending=False)

        print_results(scored)
        captured = capsys.readouterr()
        assert "TOP FUND" in captured.out
        assert scored.index[0] in captured.out
