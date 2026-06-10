"""API-Fetcher für Binance (OHLCV), CoinGecko (Metadaten) und CoinPaprika (Backup).

Alle Fetcher implementieren exponentielles Backoff bei Rate-Limit-Fehlern (429).
Alle Daten werden vor der Rückgabe durch den DiskCache geleitet.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from src.data.cache import DiskCache

logger = logging.getLogger(__name__)

# Datumsgrenzen für Binance klines (ms-Timestamps)
_MS_PER_DAY = 86_400_000


def _load_config(config_path: Path) -> dict[str, Any]:
    """Lädt die YAML-Konfiguration."""
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _exponential_backoff(
    attempt: int,
    initial: float,
    multiplier: float,
) -> float:
    """Berechnet die Wartezeit für exponentielles Backoff.

    Args:
        attempt: Aktueller Versuch (0-basiert).
        initial: Wartezeit beim ersten Retry in Sekunden.
        multiplier: Faktor pro Versuch.

    Returns:
        Wartezeit in Sekunden.
    """
    return initial * (multiplier ** attempt)


def _request_with_retry(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 15,
    max_attempts: int = 4,
    initial_backoff: float = 1.0,
    backoff_multiplier: float = 2.0,
    rate_limit_codes: list[int] | None = None,
    transient_codes: list[int] | None = None,
) -> requests.Response:
    """HTTP-GET mit exponentiellem Backoff bei Rate-Limit und transienten Fehlern.

    Args:
        url: Ziel-URL.
        params: Query-Parameter.
        headers: HTTP-Header.
        timeout: Request-Timeout in Sekunden.
        max_attempts: Maximale Versuche (inkl. erster Versuch).
        initial_backoff: Wartezeit nach erstem Fehlschlag in Sekunden.
        backoff_multiplier: Backoff-Faktor pro Versuch.
        rate_limit_codes: HTTP-Status-Codes die als Rate-Limit behandelt werden.
        transient_codes: HTTP-Status-Codes die als transienter Fehler gelten.

    Returns:
        Erfolgreiche requests.Response.

    Raises:
        requests.HTTPError: Wenn alle Versuche fehlschlagen.
        requests.ConnectionError: Bei Netzwerkproblemen.
    """
    if rate_limit_codes is None:
        rate_limit_codes = [429]
    if transient_codes is None:
        transient_codes = [500, 502, 503, 504]

    retryable_codes = set(rate_limit_codes + transient_codes)
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)

            if response.status_code == 200:
                return response

            if response.status_code in retryable_codes:
                wait = _exponential_backoff(attempt, initial_backoff, backoff_multiplier)
                logger.warning(
                    f"HTTP {response.status_code} von {url} "
                    f"(Versuch {attempt + 1}/{max_attempts}). "
                    f"Warte {wait:.1f}s..."
                )
                time.sleep(wait)
                continue

            # Nicht-retryable Fehler sofort weiterwerfen
            response.raise_for_status()

        except requests.ConnectionError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = _exponential_backoff(attempt, initial_backoff, backoff_multiplier)
                logger.warning(
                    f"Verbindungsfehler zu {url} "
                    f"(Versuch {attempt + 1}/{max_attempts}). "
                    f"Warte {wait:.1f}s..."
                )
                time.sleep(wait)
            continue

        except requests.Timeout as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = _exponential_backoff(attempt, initial_backoff, backoff_multiplier)
                logger.warning(
                    f"Timeout bei {url} "
                    f"(Versuch {attempt + 1}/{max_attempts}). "
                    f"Warte {wait:.1f}s..."
                )
                time.sleep(wait)
            continue

    if last_exc:
        raise last_exc
    raise requests.ConnectionError(
        f"Alle {max_attempts} Versuche für {url} fehlgeschlagen. "
        f"Prüfe deine Internetverbindung."
    )


class BinanceFetcher:
    """Fetcht OHLCV-Kerzendaten von der öffentlichen Binance-API (kein API-Key nötig).

    Args:
        config: Geladenes config.yaml als Dict.
        cache: DiskCache-Instanz.
    """

    def __init__(self, config: dict[str, Any], cache: DiskCache) -> None:
        self._cfg = config["api"]["binance"]
        self._retry_cfg = config["api"]["retry"]
        self._cache_cfg = config["cache"]
        self._cache = cache
        self._base_url = self._cfg["base_url"]
        self._quote = self._cfg["quote_currency"]

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """Fetcht OHLCV-Daten für ein Symbol.

        Nutzt den Cache wenn frische Daten vorhanden sind.
        Fetcht Daten in Batches wenn lookback_days > 1000 Kerzen.

        Args:
            symbol: Coin-Symbol ohne Quote (z.B. "BTC", "DOGE").
            interval: Binance-Intervall (z.B. "1d", "4h", "1h").
            lookback_days: Wie viele Tage Geschichte laden.

        Returns:
            DataFrame mit Spalten: open, high, low, close, volume, timestamp.
            Index: DatetimeIndex (UTC).

        Raises:
            ValueError: Wenn das Symbol auf Binance nicht gefunden wird.
        """
        trading_pair = f"{symbol.upper()}{self._quote}"
        cache_key = f"binance_ohlcv_{trading_pair}_{interval}_{lookback_days}d"

        # TTL basierend auf Intervall
        ttl = (
            self._cache_cfg["ohlcv_short_ttl_seconds"]
            if interval in ("1m", "5m", "15m")
            else self._cache_cfg["ohlcv_long_ttl_seconds"]
        )

        cached = self._cache.get(cache_key)
        if cached is not None:
            return self._list_to_dataframe(cached)

        df = self._fetch_all_klines(trading_pair, interval, lookback_days)
        self._cache.set(cache_key, self._dataframe_to_list(df), ttl)
        return df

    def get_available_symbols(self) -> list[str]:
        """Gibt alle auf Binance handelbaren USDT-Paare zurück (ohne Quote).

        Returns:
            Liste von Coin-Symbolen (z.B. ["BTC", "ETH", "DOGE", ...]).
        """
        cache_key = "binance_usdt_symbols"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self._base_url}{self._cfg['exchange_info_endpoint']}"
        try:
            response = _request_with_retry(
                url,
                timeout=self._cfg["request_timeout_seconds"],
                max_attempts=self._retry_cfg["max_attempts"],
                initial_backoff=self._retry_cfg["initial_backoff_seconds"],
                backoff_multiplier=self._retry_cfg["backoff_multiplier"],
            )
            info = response.json()
            symbols = [
                s["baseAsset"]
                for s in info.get("symbols", [])
                if s.get("quoteAsset") == self._quote
                and s.get("status") == "TRADING"
            ]
            self._cache.set(cache_key, symbols, 3600)
            return symbols
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning(f"Fehler beim Laden der Binance-Symbolliste: {exc}")
            return []

    def _fetch_all_klines(
        self, trading_pair: str, interval: str, lookback_days: int
    ) -> pd.DataFrame:
        """Fetcht alle Klines in Batches (Binance-Limit: 1000 pro Request).

        Args:
            trading_pair: Z.B. "BTCUSDT".
            interval: Binance-Intervall.
            lookback_days: Anzahl Tage.

        Returns:
            Vollständiger OHLCV-DataFrame.
        """
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - lookback_days * _MS_PER_DAY
        max_per_req = self._cfg["max_klines_per_request"]

        all_klines: list[list] = []
        current_start = start_ms

        while current_start < end_ms:
            url = f"{self._base_url}{self._cfg['klines_endpoint']}"
            params: dict[str, Any] = {
                "symbol": trading_pair,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": max_per_req,
            }
            try:
                response = _request_with_retry(
                    url,
                    params=params,
                    timeout=self._cfg["request_timeout_seconds"],
                    max_attempts=self._retry_cfg["max_attempts"],
                    initial_backoff=self._retry_cfg["initial_backoff_seconds"],
                    backoff_multiplier=self._retry_cfg["backoff_multiplier"],
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 400:
                    raise ValueError(
                        f"Symbol '{trading_pair}' nicht auf Binance gefunden. "
                        f"Prüfe die Schreibweise (z.B. BTC, DOGE, SHIB)."
                    ) from exc
                raise

            batch = response.json()
            if not batch:
                break

            all_klines.extend(batch)

            # Nächster Batch beginnt nach dem letzten Timestamp dieses Batches
            last_open_time = batch[-1][0]
            current_start = last_open_time + 1

            # Kein weiterer Fetch nötig wenn weniger als max zurückgegeben
            if len(batch) < max_per_req:
                break

        if not all_klines:
            raise ValueError(
                f"Keine Daten für '{trading_pair}' im Zeitraum gefunden. "
                f"Möglicherweise existiert dieses Trading-Paar nicht."
            )

        return self._raw_to_dataframe(all_klines)

    def _raw_to_dataframe(self, klines: list[list]) -> pd.DataFrame:
        """Konvertiert Binance-Rohdaten in einen sauber typisierten DataFrame."""
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    def _dataframe_to_list(self, df: pd.DataFrame) -> list[dict]:
        """Serialisiert DataFrame für JSON-Cache."""
        df_reset = df.reset_index()
        df_reset["timestamp"] = df_reset["timestamp"].astype(str)
        return df_reset.to_dict(orient="records")

    def _list_to_dataframe(self, records: list[dict]) -> pd.DataFrame:
        """Deserialisiert DataFrame aus JSON-Cache."""
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["open", "high", "low", "close", "volume"]].sort_index()


class CoinGeckoFetcher:
    """Fetcht Marktdaten und Metadaten von der CoinGecko Free API.

    Args:
        config: Geladenes config.yaml als Dict.
        cache: DiskCache-Instanz.
    """

    def __init__(self, config: dict[str, Any], cache: DiskCache) -> None:
        self._cfg = config["api"]["coingecko"]
        self._retry_cfg = config["api"]["retry"]
        self._cache_cfg = config["cache"]
        self._cache = cache
        self._base_url = self._cfg["base_url"]
        self._id_map: dict[str, str] = config.get("coingecko_id_map", {})

    def get_market_data(self, symbol: str) -> dict[str, Any]:
        """Fetcht Market Cap, Volumen und Preis-Metadaten.

        Args:
            symbol: Coin-Symbol (z.B. "BTC").

        Returns:
            Dict mit Schlüsseln: market_cap_usd, volume_24h_usd, circulating_supply,
            price_change_24h_pct, rank. Fehlende Werte sind None.
        """
        coin_id = self._resolve_id(symbol)
        cache_key = f"coingecko_market_{coin_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self._base_url}/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
        }
        try:
            response = _request_with_retry(
                url,
                params=params,
                timeout=self._cfg["request_timeout_seconds"],
                max_attempts=self._retry_cfg["max_attempts"],
                initial_backoff=self._retry_cfg["initial_backoff_seconds"],
                backoff_multiplier=self._retry_cfg["backoff_multiplier"],
            )
            raw = response.json()
            md = raw.get("market_data", {})
            result: dict[str, Any] = {
                "market_cap_usd": md.get("market_cap", {}).get("usd"),
                "volume_24h_usd": md.get("total_volume", {}).get("usd"),
                "circulating_supply": md.get("circulating_supply"),
                "price_change_24h_pct": md.get("price_change_percentage_24h"),
                "price_change_7d_pct": md.get("price_change_percentage_7d"),
                "rank": raw.get("market_cap_rank"),
                "name": raw.get("name", symbol),
                "symbol": symbol.upper(),
            }
            self._cache.set(cache_key, result, self._cache_cfg["metadata_ttl_seconds"])
            return result
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning(f"CoinGecko-Fehler für '{symbol}': {exc}")
            return self._empty_market_data(symbol)

    def get_trending_coins(self) -> list[dict[str, Any]]:
        """Fetcht aktuell trendende Coins von CoinGecko.

        Returns:
            Liste von Dicts mit Schlüsseln: name, symbol, rank.
        """
        cache_key = "coingecko_trending"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self._base_url}/search/trending"
        try:
            response = _request_with_retry(
                url,
                timeout=self._cfg["request_timeout_seconds"],
                max_attempts=self._retry_cfg["max_attempts"],
                initial_backoff=self._retry_cfg["initial_backoff_seconds"],
                backoff_multiplier=self._retry_cfg["backoff_multiplier"],
            )
            raw = response.json()
            coins = [
                {
                    "name": item["item"].get("name", ""),
                    "symbol": item["item"].get("symbol", "").upper(),
                    "rank": item["item"].get("market_cap_rank"),
                }
                for item in raw.get("coins", [])
            ]
            self._cache.set(cache_key, coins, self._cache_cfg["metadata_ttl_seconds"])
            return coins
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning(f"CoinGecko Trending-Fehler: {exc}")
            return []

    def _resolve_id(self, symbol: str) -> str:
        """Löst ein Symbol in die CoinGecko-ID auf.

        Versucht zuerst die Config-Map, dann eine API-Suche als Fallback.

        Args:
            symbol: Coin-Symbol (z.B. "BTC").

        Returns:
            CoinGecko-ID (z.B. "bitcoin").
        """
        upper = symbol.upper()
        if upper in self._id_map:
            return self._id_map[upper]

        # Fallback: Suche über CoinGecko-Search-API
        cache_key = f"coingecko_id_lookup_{upper}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self._base_url}/search"
        try:
            response = _request_with_retry(
                url,
                params={"query": upper},
                timeout=self._cfg["request_timeout_seconds"],
                max_attempts=2,
                initial_backoff=1.0,
                backoff_multiplier=2.0,
            )
            results = response.json().get("coins", [])
            if results:
                coin_id = results[0]["id"]
                self._cache.set(cache_key, coin_id, 86400)
                return coin_id
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning(f"CoinGecko-ID-Suche für '{symbol}' fehlgeschlagen: {exc}")

        # Letzter Fallback: Symbol in Kleinbuchstaben als ID
        return symbol.lower()

    def _empty_market_data(self, symbol: str) -> dict[str, Any]:
        """Gibt leere Marktdaten zurück wenn API nicht erreichbar."""
        return {
            "market_cap_usd": None,
            "volume_24h_usd": None,
            "circulating_supply": None,
            "price_change_24h_pct": None,
            "price_change_7d_pct": None,
            "rank": None,
            "name": symbol,
            "symbol": symbol.upper(),
        }


class DataFetcher:
    """Zentrale Schnittstelle für alle Datenfetch-Operationen.

    Kombiniert BinanceFetcher und CoinGeckoFetcher hinter einer einheitlichen API.
    Wird von allen anderen Modulen als einzige Datenquelle verwendet.

    Args:
        config_path: Pfad zur config.yaml.
    """

    def __init__(self, config_path: Path) -> None:
        self._config = _load_config(config_path)
        cache_dir = config_path.parent / self._config["paths"]["cache_dir"]
        self._cache = DiskCache(cache_dir)
        self._binance = BinanceFetcher(self._config, self._cache)
        self._coingecko = CoinGeckoFetcher(self._config, self._cache)

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """OHLCV-Daten von Binance.

        Args:
            symbol: Coin-Symbol ohne Quote (z.B. "BTC").
            interval: "1h", "4h" oder "1d".
            lookback_days: Wie viele Tage Geschichte laden.

        Returns:
            Sortierter OHLCV-DataFrame mit DatetimeIndex (UTC).
        """
        return self._binance.get_ohlcv(symbol, interval, lookback_days)

    def get_market_data(self, symbol: str) -> dict[str, Any]:
        """Market Cap, Volume und Metadaten von CoinGecko.

        Args:
            symbol: Coin-Symbol.

        Returns:
            Dict mit Marktdaten.
        """
        return self._coingecko.get_market_data(symbol)

    def get_trending_coins(self) -> list[dict[str, Any]]:
        """Aktuell trendende Coins von CoinGecko.

        Returns:
            Liste von Coin-Dicts.
        """
        return self._coingecko.get_trending_coins()

    def get_available_symbols(self) -> list[str]:
        """Alle handelbaren USDT-Symbole auf Binance.

        Returns:
            Liste von Symbol-Strings.
        """
        return self._binance.get_available_symbols()

    @property
    def cache(self) -> DiskCache:
        """Direkter Cache-Zugriff für andere Module (z.B. Sentiment-Fetcher)."""
        return self._cache

    @property
    def config(self) -> dict[str, Any]:
        """Zugriff auf die geladene Konfiguration."""
        return self._config
