"""Sentiment-Daten: Fear & Greed Index (alternative.me) und Reddit-Posts.

Reddit-Zugriff erfolgt über den öffentlichen JSON-Endpoint ohne OAuth.
Laut Reddit-Nutzungsbedingungen ist nicht-kommerzieller Lesezugriff erlaubt,
solange ein identifizierbarer User-Agent gesetzt wird.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
import yaml
from pathlib import Path

from src.data.cache import DiskCache

logger = logging.getLogger(__name__)

# Einfache bullish/bearish Keyword-Listen für Memecoin-Sentiment
_BULLISH_KEYWORDS = frozenset({
    "moon", "pump", "buy", "long", "bullish", "gem", "launch",
    "breakout", "ath", "surge", "rocket", "hodl", "accumulate",
})
_BEARISH_KEYWORDS = frozenset({
    "dump", "sell", "short", "bearish", "crash", "rug", "scam",
    "dead", "exit", "correction", "fear", "panic", "rekt",
})


class SentimentFetcher:
    """Fetcht Fear & Greed Index und Reddit-Sentiment.

    Args:
        config: Geladenes config.yaml als Dict.
        cache: DiskCache-Instanz.
    """

    def __init__(self, config: dict[str, Any], cache: DiskCache) -> None:
        self._cfg_fg = config["api"]["alternative_me"]
        self._cfg_reddit = config["api"]["reddit"]
        self._cfg_retry = config["api"]["retry"]
        self._cache_cfg = config["cache"]
        self._cache = cache

    def get_fear_greed(self, history_days: int = 10) -> dict[str, Any]:
        """Fetcht den Fear & Greed Index von alternative.me.

        Args:
            history_days: Anzahl historischer Tageswerte (max 30 sinnvoll).

        Returns:
            Dict mit Schlüsseln:
              - current_value (int, 0-100)
              - current_label (str, z.B. "Extreme Fear")
              - history (list[dict] mit value, label, timestamp)
              - change_3d (float, Änderung über 3 Tage, positiv = mehr Gier)
        """
        cache_key = f"fear_greed_{history_days}d"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = self._cfg_fg["base_url"]
        params = {"limit": max(history_days, 4), "format": "json"}

        try:
            response = self._get_with_retry(url, params=params)
            raw = response.json()
            data_list = raw.get("data", [])

            if not data_list:
                return self._empty_fear_greed()

            history = [
                {
                    "value": int(item["value"]),
                    "label": item["value_classification"],
                    "timestamp": int(item["timestamp"]),
                }
                for item in data_list
            ]

            current_value = history[0]["value"]
            current_label = history[0]["label"]

            # Änderung über 3 Tage (positiv = mehr Gier)
            change_3d = 0.0
            if len(history) >= 4:
                change_3d = float(current_value - history[3]["value"])

            result = {
                "current_value": current_value,
                "current_label": current_label,
                "history": history[:history_days],
                "change_3d": change_3d,
            }

            self._cache.set(cache_key, result, self._cache_cfg["fear_greed_ttl_seconds"])
            return result

        except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
            logger.warning(f"Fear & Greed API nicht erreichbar: {exc}")
            return self._empty_fear_greed()

    def get_reddit_sentiment(self, symbol: str) -> dict[str, Any]:
        """Fetcht und analysiert Reddit-Posts für ein Coin-Symbol.

        Durchsucht r/CryptoCurrency, r/CryptoMarkets, r/SatoshiStreetBets
        nach Posts die das Symbol erwähnen. Kein OAuth nötig.

        Args:
            symbol: Coin-Symbol (z.B. "BTC", "DOGE").

        Returns:
            Dict mit Schlüsseln:
              - post_count (int): Anzahl relevanter Posts
              - bullish_score (float, 0-1): Anteil bullisher Posts
              - bearish_score (float, 0-1): Anteil bearisher Posts
              - neutral_score (float, 0-1): Anteil neutraler Posts
              - avg_upvotes (float): Durchschnittliche Upvotes relevanter Posts
              - top_titles (list[str]): Top-3 Post-Titel
              - subreddits_checked (list[str])
        """
        cache_key = f"reddit_sentiment_{symbol.upper()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        subreddits: list[str] = self._cfg_reddit["subreddits"]
        posts_per_sub: int = self._cfg_reddit["posts_per_subreddit"]
        headers = {"User-Agent": self._cfg_reddit["user_agent"]}

        all_posts: list[dict[str, Any]] = []

        for subreddit in subreddits:
            url = f"{self._cfg_reddit['base_url']}/r/{subreddit}/hot.json"
            params: dict[str, Any] = {"limit": posts_per_sub}
            try:
                # Pause zwischen Reddit-Requests um Rate-Limiting zu vermeiden
                time.sleep(0.5)
                response = self._get_with_retry(url, params=params, headers=headers)
                data = response.json()
                posts = data.get("data", {}).get("children", [])
                for post in posts:
                    pd_data = post.get("data", {})
                    all_posts.append({
                        "title": pd_data.get("title", ""),
                        "score": pd_data.get("score", 0),
                        "num_comments": pd_data.get("num_comments", 0),
                        "subreddit": subreddit,
                    })
            except (requests.RequestException, KeyError, ValueError) as exc:
                logger.warning(f"Reddit-Fehler für r/{subreddit}: {exc}")
                continue

        result = self._analyze_reddit_posts(all_posts, symbol, subreddits)
        self._cache.set(cache_key, result, self._cache_cfg["reddit_ttl_seconds"])
        return result

    def _analyze_reddit_posts(
        self,
        posts: list[dict[str, Any]],
        symbol: str,
        subreddits: list[str],
    ) -> dict[str, Any]:
        """Analysiert Posts nach Symbol-Erwähnungen und Sentiment.

        Args:
            posts: Liste aller gesammelten Posts.
            symbol: Coin-Symbol nach dem gesucht wird.
            subreddits: Liste der durchsuchten Subreddits.

        Returns:
            Sentiment-Analyse-Dict.
        """
        sym_lower = symbol.lower()
        relevant: list[dict[str, Any]] = []

        for post in posts:
            title_lower = post["title"].lower()
            if sym_lower in title_lower or f"${sym_lower}" in title_lower:
                relevant.append(post)

        if not relevant:
            return {
                "post_count": 0,
                "bullish_score": 0.0,
                "bearish_score": 0.0,
                "neutral_score": 1.0,
                "avg_upvotes": 0.0,
                "top_titles": [],
                "subreddits_checked": subreddits,
            }

        bullish_count = 0
        bearish_count = 0

        for post in relevant:
            words = set(post["title"].lower().split())
            is_bullish = bool(words & _BULLISH_KEYWORDS)
            is_bearish = bool(words & _BEARISH_KEYWORDS)
            if is_bullish and not is_bearish:
                bullish_count += 1
            elif is_bearish and not is_bullish:
                bearish_count += 1

        total = len(relevant)
        neutral_count = total - bullish_count - bearish_count

        # Nach Score (Upvotes) sortieren für Top-Titel
        sorted_posts = sorted(relevant, key=lambda p: p["score"], reverse=True)
        top_titles = [p["title"] for p in sorted_posts[:3]]
        avg_upvotes = sum(p["score"] for p in relevant) / total

        return {
            "post_count": total,
            "bullish_score": round(bullish_count / total, 3),
            "bearish_score": round(bearish_count / total, 3),
            "neutral_score": round(neutral_count / total, 3),
            "avg_upvotes": round(avg_upvotes, 1),
            "top_titles": top_titles,
            "subreddits_checked": subreddits,
        }

    def _get_with_retry(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """HTTP-GET mit exponentiellem Backoff (intern, ohne Import-Zirkularität)."""
        max_attempts = self._cfg_retry["max_attempts"]
        initial_backoff = self._cfg_retry["initial_backoff_seconds"]
        multiplier = self._cfg_retry["backoff_multiplier"]
        rate_limit_codes: list[int] = self._cfg_retry["rate_limit_status_codes"]
        transient_codes: list[int] = self._cfg_retry["transient_status_codes"]
        retryable = set(rate_limit_codes + transient_codes)
        timeout = 10

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
                if response.status_code == 200:
                    return response
                if response.status_code in retryable:
                    wait = initial_backoff * (multiplier ** attempt)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    wait = initial_backoff * (multiplier ** attempt)
                    time.sleep(wait)

        if last_exc:
            raise last_exc
        raise requests.ConnectionError(f"Alle Versuche für {url} fehlgeschlagen.")

    def _empty_fear_greed(self) -> dict[str, Any]:
        """Leeres Fear & Greed Ergebnis bei API-Ausfall."""
        return {
            "current_value": None,
            "current_label": "Nicht verfügbar",
            "history": [],
            "change_3d": None,
        }


def load_sentiment_fetcher(config_path: Path, cache: DiskCache) -> SentimentFetcher:
    """Factory-Funktion die config.yaml lädt und SentimentFetcher erstellt.

    Args:
        config_path: Pfad zur config.yaml.
        cache: DiskCache-Instanz.

    Returns:
        Initialisierter SentimentFetcher.
    """
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return SentimentFetcher(config, cache)
