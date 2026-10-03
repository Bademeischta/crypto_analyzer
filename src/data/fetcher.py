"""API-Fetcher für Binance (OHLCV, Ticker, Coin-Universum) und CoinGecko (Marktdaten).

Optimierungen gegenüber einem naiven Fetch:
  - **Inkrementeller OHLCV-Store**: Kerzen werden pro Paar/Intervall dauerhaft
    gespeichert. Bei einem Refresh werden nur neue Kerzen nachgeladen, statt
    jedes Mal die komplette Historie (1 Request statt bis zu 18).
  - **Parallele Batch-Requests** für große Zeiträume (z.B. 2 Jahre 1h-Kerzen).
  - **Nur abgeschlossene Kerzen**: Die aktuell laufende Kerze hat unvollständiges
    Volumen und einen vorläufigen Schlusskurs – sie würde Features verfälschen.
  - **Endpoint-Fallback**: api.binance.com → data-api.binance.vision (Market-Data-
    Mirror ohne Geo-Sperren) bei 403/451/5xx/Netzwerkfehlern.
  - **Stale-Fallback**: Fällt CoinGecko aus (429), wird der letzte bekannte
    Stand angezeigt statt gar nichts.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from src.config import Timeframe, resolve_path
from src.data.cache import DiskCache
from src.data.http import ApiError, HttpClient, RateLimitError

logger = logging.getLogger(__name__)

_MS_PER_DAY = 86_400_000

_KLINE_COLUMNS: tuple[str, ...] = (
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
)
# Spalten, die im Rest der Anwendung verwendet werden
OHLCV_COLUMNS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "quote_volume", "num_trades", "taker_buy_base",
)

# Stablecoins & Fiat: für Analyse/Scanner uninteressant (Preis ≈ konstant)
_NON_ANALYZABLE_BASES = frozenset({
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDD", "PYUSD", "USDE", "USD1",
    "BFUSD", "RLUSD", "XUSD", "AEUR", "EUR", "EURI", "GBP", "TRY", "BRL", "UST", "USTC",
    "PAXG", "WBTC", "WBETH", "BETH",
})

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,20}$")


class SymbolNotFoundError(ValueError):
    """Symbol existiert (als USDT-Paar) nicht auf Binance."""


class BinanceFetcher:
    """OHLCV-, Ticker- und Universum-Daten von der öffentlichen Binance-API.

    Args:
        config: Geladenes config.yaml als Dict.
        cache: DiskCache-Instanz.
        http: Gemeinsamer HttpClient.
    """

    def __init__(self, config: dict[str, Any], cache: DiskCache, http: HttpClient) -> None:
        cfg = config["api"]["binance"]
        self._base_urls: list[str] = [cfg["base_url"], *cfg.get("fallback_base_urls", [])]
        self._active_base = 0
        self._quote: str = cfg["quote_currency"]
        self._limit: int = cfg["max_klines_per_request"]
        self._timeout: float = cfg["request_timeout_seconds"]
        self._workers: int = cfg.get("parallel_requests", 4)
        self._endpoints = cfg
        self._cache_cfg = config["cache"]
        self._cache = cache
        self._http = http

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_symbol(self, symbol: str) -> str:
        """Normalisiert eine Nutzereingabe zu einem Basis-Symbol ('btcusdt ' → 'BTC').

        Raises:
            ValueError: Bei ungültigen Zeichen.
        """
        sym = (symbol or "").strip().upper().lstrip("$")
        if sym.endswith(self._quote) and len(sym) > len(self._quote):
            sym = sym[: -len(self._quote)]
        if not _SYMBOL_RE.match(sym):
            raise ValueError(
                f"Ungültiges Symbol '{symbol}'. Erlaubt sind nur Buchstaben und Ziffern (z.B. BTC, PEPE)."
            )
        return sym

    def trading_pair(self, symbol: str) -> str:
        """'BTC' → 'BTCUSDT'."""
        return f"{self.normalize_symbol(symbol)}{self._quote}"

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        lookback_days: float = 365,
        now_ms: int | None = None,
    ) -> pd.DataFrame:
        """Liefert abgeschlossene OHLCV-Kerzen (inkrementell gecacht).

        Args:
            symbol: Coin-Symbol ohne Quote (z.B. "BTC").
            interval: Binance-Intervall (z.B. "1d", "4h", "1h").
            lookback_days: Gewünschte Historie in Tagen.
            now_ms: Referenzzeitpunkt in ms (nur für Tests).

        Returns:
            DataFrame mit ``OHLCV_COLUMNS``, DatetimeIndex (UTC, Kerzen-Öffnungszeit).

        Raises:
            SymbolNotFoundError: Symbol nicht auf Binance handelbar.
            ApiError: API dauerhaft nicht erreichbar.
            ValueError: Keine Daten im Zeitraum.
        """
        tf = Timeframe(interval)
        pair = self.trading_pair(symbol)
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        start_ms = now_ms - int(lookback_days * _MS_PER_DAY)
        store_key = f"ohlcv_store:{pair}:{interval}"
        ttl_ms = self._ttl_seconds(interval) * 1000

        stored = self._cache.get_frame(store_key)
        changed = False
        if stored is None:
            df = self._fetch_range(pair, tf, start_ms, now_ms, now_ms)
            meta = {"covered_from_ms": start_ms, "refreshed_at_ms": now_ms}
            changed = True
        else:
            df, meta = stored
            parts = [df]
            if start_ms < meta.get("covered_from_ms", start_ms):
                # Ältere Historie fehlt → nur die Lücke nachladen
                parts.insert(0, self._fetch_range(pair, tf, start_ms, meta["covered_from_ms"] - 1, now_ms))
                meta["covered_from_ms"] = start_ms
                changed = True
            if now_ms - meta.get("refreshed_at_ms", 0) > ttl_ms:
                # Nur neue Kerzen seit der letzten gespeicherten nachladen
                from_ms = (
                    int(df.index[-1].timestamp() * 1000) + tf.milliseconds
                    if len(df) else meta["covered_from_ms"]
                )
                if from_ms <= now_ms:
                    parts.append(self._fetch_range(pair, tf, from_ms, now_ms, now_ms))
                meta["refreshed_at_ms"] = now_ms
                changed = True
            if len(parts) > 1:
                df = pd.concat([p for p in parts if not p.empty] or [df])
                df = df[~df.index.duplicated(keep="last")].sort_index()

        if changed:
            self._cache.set_frame(store_key, df, meta)

        result = df[df.index >= pd.Timestamp(start_ms, unit="ms", tz="UTC")]
        if result.empty:
            raise ValueError(
                f"Keine abgeschlossenen Kerzen für '{pair}' ({interval}) im gewählten Zeitraum. "
                f"Möglicherweise ist das Paar sehr neu oder nicht mehr gelistet."
            )
        return result.copy()

    def data_age_minutes(self, symbol: str, interval: str) -> float | None:
        """Minuten seit dem letzten erfolgreichen Refresh des OHLCV-Stores."""
        stored = self._cache.get_frame(f"ohlcv_store:{self.trading_pair(symbol)}:{interval}")
        if stored is None:
            return None
        refreshed = stored[1].get("refreshed_at_ms")
        return None if refreshed is None else (time.time() * 1000 - refreshed) / 60_000

    def get_ticker(self, symbol: str) -> dict[str, Any]:
        """24h-Ticker (Live-Preis, 24h-Änderung, Volumen) für ein Symbol.

        Returns:
            Dict mit price, change_24h_pct, high_24h, low_24h, quote_volume_24h,
            trades_24h – oder leeres Dict bei Fehler.
        """
        pair = self.trading_pair(symbol)
        key = f"binance_ticker_{pair}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            raw = self._request(self._endpoints["ticker_endpoint"], {"symbol": pair})
            result = {
                "price": float(raw["lastPrice"]),
                "change_24h_pct": float(raw["priceChangePercent"]),
                "high_24h": float(raw["highPrice"]),
                "low_24h": float(raw["lowPrice"]),
                "quote_volume_24h": float(raw["quoteVolume"]),
                "trades_24h": int(raw.get("count", 0)),
            }
            self._cache.set(key, result, self._cache_cfg["ticker_ttl_seconds"])
            return result
        except (ApiError, KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Binance-Ticker für {pair} nicht verfügbar: {exc}")
            return {}

    def get_universe(self, limit: int = 100) -> list[dict[str, Any]]:
        """Alle handelbaren USDT-Paare, sortiert nach 24h-Handelsvolumen.

        Stablecoins/Fiat werden ausgefiltert. Eine einzige API-Anfrage
        (``ticker/24hr?type=MINI``) liefert alle Paare.

        Returns:
            Liste von Dicts: symbol, price, change_24h_pct, quote_volume_24h.
        """
        key = f"binance_universe_{self._quote}"
        cached = self._cache.get(key)
        if cached is None:
            try:
                raw = self._request(self._endpoints["ticker_endpoint"], {"type": "MINI"})
                cached = self._parse_universe(raw)
                self._cache.set(key, cached, self._cache_cfg["universe_ttl_seconds"])
            except (ApiError, TypeError, ValueError) as exc:
                logger.warning(f"Binance-Universum nicht ladbar: {exc}")
                stale = self._cache.get_stale(key)
                cached = stale[0] if stale else []
        return cached[:limit]

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------

    def _parse_universe(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now_ms = time.time() * 1000
        rows = []
        for item in raw:
            pair = item.get("symbol", "")
            if not pair.endswith(self._quote):
                continue
            base = pair[: -len(self._quote)]
            if not base or base in _NON_ANALYZABLE_BASES:
                continue
            quote_vol = float(item.get("quoteVolume") or 0.0)
            # Delistete Paare haben kein Volumen bzw. einen alten closeTime
            if quote_vol <= 0 or now_ms - float(item.get("closeTime") or 0) > 2 * _MS_PER_DAY:
                continue
            open_p = float(item.get("openPrice") or 0.0)
            last_p = float(item.get("lastPrice") or 0.0)
            rows.append({
                "symbol": base,
                "price": last_p,
                "change_24h_pct": (last_p / open_p - 1) * 100 if open_p > 0 else None,
                "quote_volume_24h": quote_vol,
            })
        rows.sort(key=lambda r: r["quote_volume_24h"], reverse=True)
        return rows

    def _ttl_seconds(self, interval: str) -> int:
        minutes = Timeframe(interval).minutes
        if minutes < 60:
            return self._cache_cfg["ohlcv_short_ttl_seconds"]
        return self._cache_cfg["ohlcv_long_ttl_seconds"]

    def _fetch_range(
        self, pair: str, tf: Timeframe, start_ms: int, end_ms: int, now_ms: int
    ) -> pd.DataFrame:
        """Lädt alle Kerzen in [start_ms, end_ms] – bei Bedarf parallel in Fenstern."""
        span = self._limit * tf.milliseconds
        windows: list[tuple[int, int]] = []
        cursor = start_ms
        while cursor <= end_ms:
            windows.append((cursor, min(cursor + span - 1, end_ms)))
            cursor += span

        if not windows:
            return self._raw_to_dataframe([], now_ms)
        if len(windows) == 1:
            batches = [self._fetch_window(pair, tf.interval, *windows[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(self._workers, len(windows))) as pool:
                batches = list(pool.map(lambda w: self._fetch_window(pair, tf.interval, *w), windows))

        rows = [row for batch in batches for row in batch]
        return self._raw_to_dataframe(rows, now_ms)

    def _fetch_window(self, pair: str, interval: str, start_ms: int, end_ms: int) -> list[list[Any]]:
        params = {
            "symbol": pair,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": self._limit,
        }
        batch = self._request(self._endpoints["klines_endpoint"], params)
        if not isinstance(batch, list):
            raise ApiError(f"Unerwartete Kline-Antwort für {pair}: {type(batch).__name__}")
        return batch

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Request mit automatischem Wechsel auf Fallback-Hosts."""
        last_exc: ApiError | None = None
        n = len(self._base_urls)
        for offset in range(n):
            idx = (self._active_base + offset) % n
            url = f"{self._base_urls[idx]}{endpoint}"
            try:
                data = self._http.get_json(url, params=params, timeout=self._timeout)
                self._active_base = idx
                return data
            except RateLimitError as exc:
                last_exc = exc
            except ApiError as exc:
                if exc.status_code == 400:
                    if "-1121" in exc.body or "Invalid symbol" in exc.body:
                        pair = (params or {}).get("symbol", "?")
                        raise SymbolNotFoundError(
                            f"Symbol '{pair}' nicht auf Binance gefunden. "
                            f"Prüfe die Schreibweise (z.B. BTC, DOGE, PEPE – ohne USDT)."
                        ) from exc
                    raise
                # 403/418/451 (Geo-Sperre/Bann), 5xx oder Netzwerk → nächster Host
                last_exc = exc
            logger.warning(f"Binance-Host {self._base_urls[idx]} fehlgeschlagen: {last_exc}")
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _raw_to_dataframe(klines: list[list[Any]], now_ms: int) -> pd.DataFrame:
        """Konvertiert Binance-Rohdaten in einen typisierten DataFrame (nur geschlossene Kerzen)."""
        if not klines:
            empty = pd.DataFrame(columns=list(OHLCV_COLUMNS), dtype="float64")
            empty.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
            return empty

        df = pd.DataFrame(klines, columns=list(_KLINE_COLUMNS))
        df = df[pd.to_numeric(df["close_time"], errors="coerce") < now_ms]
        df.index = pd.DatetimeIndex(
            pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True), name="timestamp"
        )
        out = df[list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce").astype("float64")
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out


class CoinGeckoFetcher:
    """Marktdaten (Market Cap, Supply, ATH, Community-Sentiment) von CoinGecko.

    Args:
        config: Geladenes config.yaml als Dict.
        cache: DiskCache-Instanz.
        http: Gemeinsamer HttpClient.
    """

    def __init__(self, config: dict[str, Any], cache: DiskCache, http: HttpClient) -> None:
        self._cfg = config["api"]["coingecko"]
        self._cache_cfg = config["cache"]
        self._cache = cache
        self._http = http
        self._base_url = self._cfg["base_url"]
        self._id_map: dict[str, str] = {k.upper(): v for k, v in config.get("coingecko_id_map", {}).items()}

    def get_market_data(self, symbol: str) -> dict[str, Any]:
        """Market Cap, Supply, Preisänderungen, ATH und Community-Sentiment.

        Fällt bei API-Fehlern auf den letzten gecachten Stand zurück
        (Feld ``stale_minutes`` gibt dann dessen Alter an).
        """
        coin_id = self._resolve_id(symbol)
        if coin_id is None:
            return self._empty_market_data(symbol, "Coin bei CoinGecko nicht eindeutig gefunden.")

        key = f"coingecko_market_{coin_id}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false",
        }
        try:
            raw = self._http.get_json(
                f"{self._base_url}/coins/{coin_id}",
                params=params,
                timeout=self._cfg["request_timeout_seconds"],
                max_attempts=self._cfg.get("max_attempts", 2),
            )
            result = self._parse_market_data(raw, symbol, coin_id)
            self._cache.set(key, result, self._cache_cfg["metadata_ttl_seconds"])
            return result
        except (ApiError, KeyError, TypeError, ValueError) as exc:
            stale = self._cache.get_stale(key)
            if stale is not None:
                data, age = stale
                logger.info(f"CoinGecko nicht erreichbar ({exc}) – nutze Stand von vor {age / 60:.0f} Min.")
                return {**data, "stale_minutes": round(age / 60, 1)}
            logger.warning(f"CoinGecko-Fehler für '{symbol}': {exc}")
            return self._empty_market_data(symbol, f"CoinGecko nicht erreichbar: {exc}")

    def get_trending_coins(self) -> list[dict[str, Any]]:
        """Aktuell trendende Coins (CoinGecko-Suche der letzten 24h)."""
        key = "coingecko_trending"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            raw = self._http.get_json(
                f"{self._base_url}/search/trending",
                timeout=self._cfg["request_timeout_seconds"],
                max_attempts=self._cfg.get("max_attempts", 2),
            )
            coins = [
                {
                    "name": item["item"].get("name", ""),
                    "symbol": item["item"].get("symbol", "").upper(),
                    "rank": item["item"].get("market_cap_rank"),
                }
                for item in raw.get("coins", [])
            ]
            self._cache.set(key, coins, self._cache_cfg["metadata_ttl_seconds"])
            return coins
        except (ApiError, KeyError, TypeError, ValueError) as exc:
            stale = self._cache.get_stale(key)
            if stale is not None:
                return stale[0]
            logger.warning(f"CoinGecko Trending-Fehler: {exc}")
            return []

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_market_data(raw: dict[str, Any], symbol: str, coin_id: str) -> dict[str, Any]:
        md = raw.get("market_data") or {}

        def usd(field: str) -> float | None:
            value = (md.get(field) or {}).get("usd")
            return float(value) if value is not None else None

        return {
            "coingecko_id": coin_id,
            "name": raw.get("name", symbol),
            "symbol": symbol.upper(),
            "rank": raw.get("market_cap_rank"),
            "market_cap_usd": usd("market_cap"),
            "fdv_usd": usd("fully_diluted_valuation"),
            "volume_24h_usd": usd("total_volume"),
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
            "price_change_24h_pct": md.get("price_change_percentage_24h"),
            "price_change_7d_pct": md.get("price_change_percentage_7d"),
            "price_change_30d_pct": md.get("price_change_percentage_30d"),
            "price_change_1y_pct": md.get("price_change_percentage_1y"),
            "ath_usd": usd("ath"),
            "ath_change_pct": (md.get("ath_change_percentage") or {}).get("usd"),
            "ath_date": (md.get("ath_date") or {}).get("usd"),
            "sentiment_up_pct": raw.get("sentiment_votes_up_percentage"),
            "categories": [c for c in (raw.get("categories") or []) if c][:4],
        }

    def _resolve_id(self, symbol: str) -> str | None:
        """Symbol → CoinGecko-ID (Config-Map, dann Such-API mit exaktem Symbol-Match).

        Bei mehreren Coins mit gleichem Ticker gewinnt der mit dem besten
        Market-Cap-Rang. Gibt es keinen exakten Treffer, wird ``None``
        zurückgegeben statt einen falschen Coin zu raten.
        """
        upper = symbol.upper()
        if upper in self._id_map:
            return self._id_map[upper]

        key = f"coingecko_id_lookup_{upper}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached or None

        try:
            raw = self._http.get_json(
                f"{self._base_url}/search",
                params={"query": upper},
                timeout=self._cfg["request_timeout_seconds"],
                max_attempts=2,
            )
        except ApiError as exc:
            logger.warning(f"CoinGecko-ID-Suche für '{symbol}' fehlgeschlagen: {exc}")
            return None

        candidates = [c for c in raw.get("coins", []) if str(c.get("symbol", "")).upper() == upper]
        candidates.sort(key=lambda c: c.get("market_cap_rank") or 10**9)
        coin_id = candidates[0]["id"] if candidates else ""
        # Auch negative Ergebnisse cachen, um die API nicht zu fluten
        self._cache.set(key, coin_id, self._cache_cfg["coingecko_id_ttl_seconds"])
        return coin_id or None

    @staticmethod
    def _empty_market_data(symbol: str, reason: str = "") -> dict[str, Any]:
        return {
            "name": symbol.upper(),
            "symbol": symbol.upper(),
            "rank": None,
            "market_cap_usd": None,
            "volume_24h_usd": None,
            "price_change_24h_pct": None,
            "price_change_7d_pct": None,
            "unavailable_reason": reason,
        }


class DataFetcher:
    """Zentrale Schnittstelle für alle Marktdaten-Operationen.

    Args:
        config: Geladenes config.yaml als Dict.
        config_path: Pfad der config.yaml (für relative Datenpfade).
        http: Optionaler HttpClient (für Tests).
    """

    def __init__(self, config: dict[str, Any], config_path: Path, http: HttpClient | None = None) -> None:
        self._config = config
        self._cache = DiskCache(resolve_path(config_path, config["paths"]["cache_dir"]))
        api = config["api"]
        self._http = http or HttpClient(
            api["retry"],
            user_agent=api["user_agent"],
            host_rate_limits={
                _host(api["coingecko"]["base_url"]): api["coingecko"]["rate_limit_per_minute"],
                _host(api["binance"]["base_url"]): api["binance"]["rate_limit_per_minute"],
            },
        )
        self._binance = BinanceFetcher(config, self._cache, self._http)
        self._coingecko = CoinGeckoFetcher(config, self._cache, self._http)

    def normalize_symbol(self, symbol: str) -> str:
        """Nutzereingabe → Basis-Symbol (z.B. ' btcusdt' → 'BTC')."""
        return self._binance.normalize_symbol(symbol)

    def get_ohlcv(self, symbol: str, interval: str = "1d", lookback_days: float = 365) -> pd.DataFrame:
        """Abgeschlossene OHLCV-Kerzen von Binance (inkrementell gecacht)."""
        return self._binance.get_ohlcv(symbol, interval, lookback_days)

    def get_many_ohlcv(
        self, symbols: list[str], interval: str, lookback_days: float
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """Lädt mehrere Symbole parallel.

        Returns:
            Tupel (Symbol → DataFrame, Symbol → Fehlermeldung).
        """
        frames: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}

        def load(sym: str) -> tuple[str, pd.DataFrame | None, str | None]:
            try:
                return sym, self.get_ohlcv(sym, interval, lookback_days), None
            except Exception as exc:  # pro Symbol isolieren
                return sym, None, str(exc)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for sym, df, err in pool.map(load, symbols):
                if df is not None:
                    frames[sym] = df
                else:
                    errors[sym] = err or "unbekannter Fehler"
        return frames, errors

    def get_market_data(self, symbol: str) -> dict[str, Any]:
        """CoinGecko-Marktdaten, ergänzt um Live-Preis und 24h-Daten von Binance."""
        market = dict(self._coingecko.get_market_data(symbol))
        ticker = self._binance.get_ticker(symbol)
        if ticker:
            market["price"] = ticker["price"]
            market["high_24h"] = ticker["high_24h"]
            market["low_24h"] = ticker["low_24h"]
            market["binance_quote_volume_24h"] = ticker["quote_volume_24h"]
            market["trades_24h"] = ticker["trades_24h"]
            # Binance-Änderung passt exakt zum Chart → bevorzugen
            market["price_change_24h_pct"] = ticker["change_24h_pct"]
            if market.get("volume_24h_usd") is None:
                market["volume_24h_usd"] = ticker["quote_volume_24h"]
        return market

    def get_trending_coins(self) -> list[dict[str, Any]]:
        """Aktuell trendende Coins von CoinGecko."""
        return self._coingecko.get_trending_coins()

    def get_universe(self, limit: int = 100) -> list[dict[str, Any]]:
        """Top-USDT-Paare nach 24h-Volumen."""
        return self._binance.get_universe(limit)

    def data_age_minutes(self, symbol: str, interval: str) -> float | None:
        """Alter der OHLCV-Daten in Minuten."""
        return self._binance.data_age_minutes(symbol, interval)

    @property
    def cache(self) -> DiskCache:
        """Direkter Cache-Zugriff für andere Module (z.B. Sentiment-Fetcher)."""
        return self._cache

    @property
    def http(self) -> HttpClient:
        """Gemeinsamer HTTP-Client."""
        return self._http

    @property
    def config(self) -> dict[str, Any]:
        """Zugriff auf die geladene Konfiguration."""
        return self._config


def _host(url: str) -> str:
    return urlparse(url).netloc
