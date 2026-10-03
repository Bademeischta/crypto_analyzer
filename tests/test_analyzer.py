"""End-to-End-Test des Analyzers mit Fake-Datenquellen (offline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conftest import make_fear_greed, make_ohlcv
from src.analysis.analyzer import CryptoAnalyzer
from src.config import load_config, resolve_path
from src.data.cache import DiskCache
from src.data.fetcher import SymbolNotFoundError


class FakeFetcher:
    def __init__(self, config_path: Path, frames: dict[str, pd.DataFrame]):
        cfg = load_config(config_path)
        self._cache = DiskCache(resolve_path(config_path, cfg["paths"]["cache_dir"]))
        self.frames = frames
        self.http = None

    def normalize_symbol(self, symbol: str) -> str:
        sym = symbol.strip().upper()
        if not sym.isalnum():
            raise ValueError("ungültig")
        return sym.removesuffix("USDT") or sym

    def get_ohlcv(self, symbol: str, interval: str = "1d", lookback_days: float = 365) -> pd.DataFrame:
        if symbol not in self.frames:
            raise SymbolNotFoundError(f"Symbol '{symbol}USDT' nicht auf Binance gefunden.")
        df = self.frames[symbol]
        return df[df.index > df.index[-1] - pd.Timedelta(days=lookback_days)]

    def get_many_ohlcv(self, symbols, interval, lookback_days):
        frames, errors = {}, {}
        for s in symbols:
            try:
                frames[s] = self.get_ohlcv(s, interval, lookback_days)
            except ValueError as exc:
                errors[s] = str(exc)
        return frames, errors

    def get_market_data(self, symbol: str) -> dict[str, Any]:
        return {"name": f"{symbol} Coin", "symbol": symbol, "price": 1.0, "sentiment_up_pct": 60.0}

    def data_age_minutes(self, symbol: str, interval: str) -> float:
        return 1.0

    def get_trending_coins(self):
        return []

    def get_universe(self, limit: int = 100):
        return [{"symbol": s} for s in self.frames][:limit]

    @property
    def cache(self) -> DiskCache:
        return self._cache


class FakeSentiment:
    def __init__(self):
        self.fng = make_fear_greed()

    def get_fear_greed_history(self) -> pd.Series:
        return self.fng

    def get_fear_greed(self, history_days: int = 90) -> dict[str, Any]:
        return {"current_value": int(self.fng.iloc[-1]), "current_label": "Greed", "history": []}

    def get_reddit_sentiment(self, symbol: str, aliases=None) -> dict[str, Any]:
        return {"post_count": 0, "error": None}


@pytest.fixture
def analyzer(tmp_config_path):
    frames = {
        "BTC": make_ohlcv(1200, seed=1, start="2022-01-01", price=30_000),
        "ETH": make_ohlcv(1200, seed=2, start="2022-01-01", price=2_000),
        "XYZ": make_ohlcv(1200, seed=3, start="2022-01-01", price=0.001),
        "NEW": make_ohlcv(150, seed=4, start="2025-01-01"),
    }
    return CryptoAnalyzer(tmp_config_path, fetcher=FakeFetcher(tmp_config_path, frames),
                          sentiment_fetcher=FakeSentiment())


def test_full_analysis_and_model_cache(analyzer):
    messages: list[str] = []
    result = analyzer.analyze("xyzusdt", "1d", lookback_days=180, progress=messages.append)

    assert result.error is None and result.ml_error is None
    assert result.symbol == "XYZ"
    assert result.prediction is not None
    assert set(result.prediction.probabilities) == {"BEARISH", "NEUTRAL", "BULLISH"}
    assert abs(sum(result.prediction.probabilities.values()) - 1) < 3e-3  # auf 3 Stellen gerundet
    assert result.eval_metrics is not None and result.eval_metrics.n_folds >= 3
    assert result.volatility_metrics is not None
    assert result.backtest is not None
    assert result.oof is not None and not result.oof.empty
    assert (result.ohlcv.index[-1] - result.ohlcv.index[0]).days < 180
    assert {"rsi_14", "macd", "bb_upper", "corr_btc", "fng_value"} <= set(result.ohlcv.columns)
    assert any("Trainiere" in m for m in messages)
    assert result.model_info["from_cache"] is False

    cached = analyzer.analyze("XYZ", "1d", lookback_days=180)
    assert cached.model_info["from_cache"] is True
    assert cached.training_time_seconds == 0.0
    assert cached.prediction.probabilities == result.prediction.probabilities


def test_short_history_degrades_gracefully(analyzer):
    result = analyzer.analyze("NEW", "1d", lookback_days=365)
    assert result.error is None
    assert result.ml_error and "Zu wenig Historie" in result.ml_error
    assert result.prediction is None
    assert not result.ohlcv.empty and "rsi_14" in result.ohlcv.columns


def test_unknown_and_invalid_symbols(analyzer):
    assert "nicht auf Binance" in analyzer.analyze("NOPE").error
    assert analyzer.analyze("B/TC").error


def test_compare(analyzer):
    cmp = analyzer.compare(["BTC", "ETH", "NOPE"], "1d", 180)
    assert list(cmp.normalized.columns) == ["BTC", "ETH"]
    assert (cmp.normalized.iloc[0] == 100).all()
    assert cmp.correlation.shape == (2, 2)
    assert "NOPE" in cmp.errors
    assert cmp.stats.loc["BTC", "Beta zu BTC"] == pytest.approx(1.0)


def test_scan(analyzer):
    table = analyzer.scan(["BTC", "NEW", "NOPE"], "1d", max_workers=2)
    assert list(table["Symbol"]) == ["BTC", "NEW", "NOPE"]
    assert table.loc[table["Symbol"] == "BTC", "Richtungsmodell"].iloc[0] in (
        "signifikant", "schwach", "keine Vorhersagekraft")
    assert table.loc[table["Symbol"] == "NOPE", "Signal"].iloc[0] == "–"
