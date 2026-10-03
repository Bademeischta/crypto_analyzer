"""Datenschicht: Cache, HTTP-Retry, Binance/CoinGecko-Fetcher, Validator – komplett offline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
import requests

from conftest import make_ohlcv
from src.data.cache import DiskCache
from src.data.fetcher import BinanceFetcher, CoinGeckoFetcher, SymbolNotFoundError
from src.data.http import ApiError, HttpClient, RateLimitError
from src.data.validator import DataValidator

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_roundtrip_expiry_and_stale(tmp_path, monkeypatch):
    cache = DiskCache(tmp_path)
    cache.set("k", {"a": 1}, ttl=10)
    assert cache.get("k") == {"a": 1}

    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 60)
    assert cache.get("k") is None
    data, age = cache.get_stale("k")
    assert data == {"a": 1} and age >= 59
    assert cache.clear_expired() == 1


def test_cache_ignores_corrupt_files(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("k", [1, 2, 3], ttl=100)
    path = next(tmp_path.glob("*.json"))
    path.write_text("{kaputt", encoding="utf-8")
    assert cache.get("k") is None
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_frames(tmp_path):
    cache = DiskCache(tmp_path)
    df = make_ohlcv(50)
    cache.set_frame("ohlcv", df, {"x": 1})
    loaded, meta = cache.get_frame("ohlcv")
    pd.testing.assert_frame_equal(loaded, df)
    assert meta == {"x": 1}
    assert cache.stats()["entries"] == 1
    assert cache.clear_all() == 1
    assert cache.get_frame("ohlcv") is None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status: int, payload: Any = None, headers: dict[str, str] | None = None, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text or ("" if payload is None else str(payload))

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.headers: dict[str, str] = {}

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


RETRY = {"max_attempts": 3, "initial_backoff_seconds": 1.0, "backoff_multiplier": 2.0,
         "rate_limit_status_codes": [429], "transient_status_codes": [500, 502, 503, 504]}


def test_http_retries_and_respects_retry_after():
    sleeps: list[float] = []
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {"ok": True})])
    client = HttpClient(RETRY, session=session, sleep=sleeps.append)
    assert client.get_json("https://x.test/a") == {"ok": True}
    assert sleeps == [7.0]


def test_http_gives_up_with_rate_limit_error():
    session = FakeSession([FakeResponse(429)] * 3)
    client = HttpClient(RETRY, session=session, sleep=lambda s: None)
    with pytest.raises(RateLimitError):
        client.get("https://x.test/a")
    assert len(session.calls) == 3


def test_http_does_not_retry_client_errors():
    session = FakeSession([FakeResponse(400, text='{"code":-1121,"msg":"Invalid symbol."}')])
    client = HttpClient(RETRY, session=session, sleep=lambda s: None)
    with pytest.raises(ApiError) as exc:
        client.get("https://x.test/a")
    assert exc.value.status_code == 400 and "-1121" in exc.value.body
    assert len(session.calls) == 1


def test_http_retries_network_errors():
    session = FakeSession([requests.ConnectionError("boom"), FakeResponse(200, [1])])
    client = HttpClient(RETRY, session=session, sleep=lambda s: None)
    assert client.get_json("https://x.test/a") == [1]


def test_http_throttles_per_host():
    sleeps: list[float] = []
    session = FakeSession([FakeResponse(200, 1)] * 3)
    client = HttpClient(RETRY, host_rate_limits={"slow.test": 60}, session=session, sleep=sleeps.append)
    for _ in range(3):
        client.get("https://slow.test/x")
    assert len(sleeps) == 2 and all(0.9 < s <= 2.0 for s in sleeps)


# ---------------------------------------------------------------------------
# Binance
# ---------------------------------------------------------------------------

DAY_MS = 86_400_000


class FakeBinanceHttp:
    """Simuliert die Kline-API: liefert Tageskerzen im angefragten Zeitfenster."""

    def __init__(self, first_open_ms: int, fail_hosts: set[str] | None = None, invalid: bool = False):
        self.first_open_ms = first_open_ms
        self.fail_hosts = fail_hosts or set()
        self.invalid = invalid
        self.requests: list[dict[str, Any]] = []

    def get_json(self, url, params=None, timeout=None, **_):
        host = url.split("/")[2]
        if host in self.fail_hosts:
            raise ApiError("geo", 451, url)
        if self.invalid:
            raise ApiError("bad", 400, url, '{"code":-1121,"msg":"Invalid symbol."}')
        self.requests.append(dict(params))
        start = max(params["startTime"], self.first_open_ms)
        start = self.first_open_ms + -(-(start - self.first_open_ms) // DAY_MS) * DAY_MS
        rows = []
        t = start
        while t <= params["endTime"] and len(rows) < params["limit"]:
            price = 100 + (t - self.first_open_ms) / DAY_MS
            rows.append([t, str(price), str(price + 1), str(price - 1), str(price + 0.5), "10",
                         t + DAY_MS - 1, "1000", 50, "6", "600", "0"])
            t += DAY_MS
        return rows


@pytest.fixture
def binance_cfg(config):
    return config


def test_binance_parses_and_drops_unclosed_candle(binance_cfg, tmp_path):
    first = 1_700_000_000_000 - (1_700_000_000_000 % DAY_MS)
    http = FakeBinanceHttp(first)
    fetcher = BinanceFetcher(binance_cfg, DiskCache(tmp_path), http)
    now = first + 10 * DAY_MS + DAY_MS // 2  # mitten in der 11. Kerze
    df = fetcher.get_ohlcv("btc", "1d", lookback_days=30, now_ms=now)
    assert len(df) == 10  # laufende Kerze verworfen
    assert df.index[0] == pd.Timestamp(first, unit="ms", tz="UTC")
    assert df["taker_buy_base"].iloc[0] == 6.0
    assert http.requests[0]["symbol"] == "BTCUSDT"


def test_binance_incremental_refresh_only_fetches_new_candles(binance_cfg, tmp_path):
    first = 1_700_000_000_000 - (1_700_000_000_000 % DAY_MS)
    http = FakeBinanceHttp(first)
    fetcher = BinanceFetcher(binance_cfg, DiskCache(tmp_path), http)
    now = first + 100 * DAY_MS + 1
    fetcher.get_ohlcv("ETH", "1d", 50, now_ms=now)
    n_initial = len(http.requests)

    # Innerhalb der TTL: kein neuer Request
    fetcher.get_ohlcv("ETH", "1d", 50, now_ms=now + 1000)
    assert len(http.requests) == n_initial

    # Nach TTL: genau ein Request, der erst nach der letzten gespeicherten Kerze beginnt
    later = now + 3 * DAY_MS
    df = fetcher.get_ohlcv("ETH", "1d", 50, now_ms=later)
    new_requests = http.requests[n_initial:]
    assert len(new_requests) == 1
    assert new_requests[0]["startTime"] == first + 100 * DAY_MS
    assert df.index[-1] == pd.Timestamp(first + 102 * DAY_MS, unit="ms", tz="UTC")
    assert df.index.is_unique and df.index.is_monotonic_increasing


def test_binance_parallel_windows_for_long_ranges(binance_cfg, tmp_path):
    binance_cfg["api"]["binance"]["max_klines_per_request"] = 100
    first = 1_600_000_000_000 - (1_600_000_000_000 % DAY_MS)
    http = FakeBinanceHttp(first)
    fetcher = BinanceFetcher(binance_cfg, DiskCache(tmp_path), http)
    df = fetcher.get_ohlcv("SOL", "1d", 450, now_ms=first + 450 * DAY_MS + 5)
    assert len(http.requests) == 5
    # Erste Kerze öffnet 5 ms vor dem Fenster → nicht enthalten
    assert len(df) == 449 and df.index.is_unique


def test_binance_invalid_symbol_and_host_fallback(binance_cfg, tmp_path):
    first = 1_700_000_000_000 - (1_700_000_000_000 % DAY_MS)
    with pytest.raises(SymbolNotFoundError):
        BinanceFetcher(binance_cfg, DiskCache(tmp_path / "a"), FakeBinanceHttp(first, invalid=True)).get_ohlcv(
            "NOPE", "1d", 10, now_ms=first + 20 * DAY_MS)

    http = FakeBinanceHttp(first, fail_hosts={"api.binance.com"})
    df = BinanceFetcher(binance_cfg, DiskCache(tmp_path / "b"), http).get_ohlcv(
        "BTC", "1d", 10, now_ms=first + 20 * DAY_MS)
    assert not df.empty


@pytest.mark.parametrize("raw, expected", [("btc", "BTC"), (" BTCUSDT ", "BTC"), ("$pepe", "PEPE")])
def test_normalize_symbol(binance_cfg, tmp_path, raw, expected):
    assert BinanceFetcher(binance_cfg, DiskCache(tmp_path), None).normalize_symbol(raw) == expected


def test_normalize_symbol_rejects_garbage(binance_cfg, tmp_path):
    with pytest.raises(ValueError):
        BinanceFetcher(binance_cfg, DiskCache(tmp_path), None).normalize_symbol("BTC/../x")


def test_universe_filters_stablecoins(binance_cfg, tmp_path):
    now = time.time() * 1000
    raw = [
        {"symbol": "BTCUSDT", "quoteVolume": "100", "closeTime": now, "openPrice": "10", "lastPrice": "11"},
        {"symbol": "USDCUSDT", "quoteVolume": "500", "closeTime": now, "openPrice": "1", "lastPrice": "1"},
        {"symbol": "ETHBTC", "quoteVolume": "900", "closeTime": now, "openPrice": "1", "lastPrice": "1"},
        {"symbol": "OLDUSDT", "quoteVolume": "50", "closeTime": now - 10 * DAY_MS, "openPrice": "1", "lastPrice": "1"},
        {"symbol": "SOLUSDT", "quoteVolume": "300", "closeTime": now, "openPrice": "10", "lastPrice": "9"},
    ]
    rows = BinanceFetcher(binance_cfg, DiskCache(tmp_path), None)._parse_universe(raw)
    assert [r["symbol"] for r in rows] == ["SOL", "BTC"]
    assert rows[1]["change_24h_pct"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# CoinGecko
# ---------------------------------------------------------------------------


class FakeGeckoHttp:
    def __init__(self, search_result=None, fail=False):
        self.search_result = search_result or []
        self.fail = fail

    def get_json(self, url, **_):
        if self.fail:
            raise RateLimitError("429", 429, url)
        if url.endswith("/search"):
            return {"coins": self.search_result}
        return {"name": "Foo", "market_cap_rank": 5, "market_data": {"market_cap": {"usd": 1e9}},
                "sentiment_votes_up_percentage": 70.0}


def test_coingecko_resolves_exact_symbol_with_best_rank(config, tmp_path):
    http = FakeGeckoHttp([
        {"id": "foo-fake", "symbol": "FOO", "market_cap_rank": 900},
        {"id": "foobar", "symbol": "FOOBAR", "market_cap_rank": 1},
        {"id": "foo-real", "symbol": "foo", "market_cap_rank": 30},
    ])
    gecko = CoinGeckoFetcher(config, DiskCache(tmp_path), http)
    assert gecko._resolve_id("FOO") == "foo-real"
    assert gecko._resolve_id("BTC") == "bitcoin"  # aus der Config-Map


def test_coingecko_falls_back_to_stale_data(config, tmp_path, monkeypatch):
    cache = DiskCache(tmp_path)
    gecko = CoinGeckoFetcher(config, cache, FakeGeckoHttp())
    fresh = gecko.get_market_data("BTC")
    assert fresh["market_cap_usd"] == 1e9 and fresh["sentiment_up_pct"] == 70.0

    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 10 * 3600)
    stale = CoinGeckoFetcher(config, cache, FakeGeckoHttp(fail=True)).get_market_data("BTC")
    assert stale["market_cap_usd"] == 1e9 and stale["stale_minutes"] > 500


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_fills_short_gaps_and_repairs_ohlc():
    df = make_ohlcv(120)
    df = df.drop(df.index[[50, 51]])  # 2 fehlende Kerzen
    df.iloc[10, df.columns.get_loc("high")] = df["low"].iloc[10] * 0.5  # High < Low
    df = pd.concat([df, df.iloc[[5]]])  # Duplikat
    result = DataValidator(max_consecutive_gaps=3, minimum_samples=30).validate(df, "T", "1d")
    out = result.df
    assert result.is_valid
    assert len(out) == 120 and out.index.is_unique and out.index.is_monotonic_increasing
    assert result.stats["filled_bars"] == 2
    assert (out["high"] >= out[["open", "close", "low"]].max(axis=1) - 1e-12).all()
    filled = out.iloc[50]
    assert filled["volume"] == 0 and filled["open"] == filled["close"] == out["close"].iloc[49]


def test_validator_uses_segment_after_long_trading_halt():
    df = make_ohlcv(200)
    df = df.drop(df.index[60:80])  # 20 Tage Handelspause
    result = DataValidator(max_consecutive_gaps=3, minimum_samples=30).validate(df, "T", "1d")
    assert result.df.index[0] == df.index[60]
    assert len(result.df) == 120
    assert any("Handelspause" in w for w in result.warnings)


def test_validator_rejects_too_little_data():
    result = DataValidator(minimum_samples=30).validate(make_ohlcv(10), "T", "1d")
    assert not result.is_valid


def test_validator_drops_non_positive_prices():
    df = make_ohlcv(60)
    df.iloc[20, df.columns.get_loc("close")] = 0.0
    result = DataValidator(minimum_samples=30).validate(df, "T")
    assert (result.df["close"] > 0).all() and len(result.df) == 59
    assert np.isfinite(result.df[["open", "high", "low", "close"]].to_numpy()).all()
