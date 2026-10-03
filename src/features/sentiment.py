"""Sentiment-Daten: Fear & Greed Index (alternative.me) und Reddit-Posts.

Fear & Greed wird als **komplette Historie** geladen (ein Request, seit 2018)
und kann so als zeitlich korrektes ML-Feature dienen – statt eines einzelnen
aktuellen Werts, der rückwirkend auf alle historischen Zeilen kopiert würde
(das wäre Lookahead-Bias).

Reddit: Der anonyme JSON-Endpoint wird inzwischen häufig mit 403 blockiert.
Als Fallback wird der öffentliche RSS/Atom-Feed desselben Subreddits gelesen.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd

from src.data.cache import DiskCache
from src.data.http import ApiError, HttpClient

logger = logging.getLogger(__name__)

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
# Nach einem Rate-Limit Reddit für diese Zeit meiden (Sekunden)
_REDDIT_COOLDOWN_SECONDS = 600
_TOKEN_RE = re.compile(r"[a-z0-9']+")

_BULLISH_KEYWORDS = frozenset({
    "moon", "mooning", "pump", "pumping", "buy", "buying", "long", "bullish", "bull", "gem",
    "breakout", "ath", "surge", "surging", "rocket", "hodl", "accumulate", "accumulating",
    "rally", "rallying", "soar", "soaring", "uptrend", "undervalued", "adoption", "etf",
    "approval", "approved", "partnership", "green", "gains", "rebound",
})
_BEARISH_KEYWORDS = frozenset({
    "dump", "dumping", "sell", "selling", "short", "bearish", "bear", "crash", "crashing",
    "rug", "rugpull", "scam", "dead", "exit", "correction", "fear", "panic", "rekt",
    "plunge", "plunging", "collapse", "liquidated", "liquidation", "hack", "hacked",
    "exploit", "downtrend", "overvalued", "bubble", "lawsuit", "sec", "ban", "red", "losses",
})


class SentimentFetcher:
    """Fetcht Fear & Greed Index und Reddit-Sentiment.

    Args:
        config: Geladenes config.yaml als Dict.
        cache: DiskCache-Instanz.
        http: Gemeinsamer HttpClient.
    """

    def __init__(self, config: dict[str, Any], cache: DiskCache, http: HttpClient) -> None:
        self._cfg_fg = config["api"]["alternative_me"]
        self._cfg_reddit = config["api"]["reddit"]
        self._cache_cfg = config["cache"]
        self._cache = cache
        self._http = http
        # Circuit-Breaker: blockierte Reddit-Zugriffswege merken (Prozess-weit)
        self._reddit_lock = threading.Lock()
        self._json_blocked = False
        self._reddit_cooldown_until = 0.0

    # ------------------------------------------------------------------
    # Fear & Greed
    # ------------------------------------------------------------------

    def get_fear_greed_history(self) -> pd.Series:
        """Komplette Fear-&-Greed-Historie als tägliche Serie.

        Returns:
            Series (0–100) mit UTC-DatetimeIndex (Tagesbeginn), aufsteigend.
            Leer, wenn die API nicht erreichbar ist und kein Cache existiert.
        """
        return _records_to_series(self._fear_greed_records())

    def get_fear_greed(self, history_days: int = 90) -> dict[str, Any]:
        """Aktueller Fear & Greed Index plus Verlauf für die UI.

        Returns:
            Dict: current_value, current_label, change_1d, change_7d, change_30d,
            avg_30d, history (Liste von {date, value}).
        """
        records = self._fear_greed_records()
        series = _records_to_series(records)
        if series.empty:
            return {
                "current_value": None,
                "current_label": "Nicht verfügbar",
                "history": [],
                "change_1d": None,
                "change_7d": None,
                "change_30d": None,
                "avg_30d": None,
            }

        def change(days: int) -> float | None:
            if len(series) <= days:
                return None
            return float(series.iloc[-1] - series.iloc[-1 - days])

        labels = {int(r[0]): r[2] for r in records}
        current = int(series.iloc[-1])
        recent = series.iloc[-history_days:]
        return {
            "current_value": current,
            "current_label": labels.get(int(series.index[-1].timestamp())) or classify_fear_greed(current),
            "change_1d": change(1),
            "change_7d": change(7),
            "change_30d": change(30),
            "avg_30d": float(series.iloc[-30:].mean()),
            "history": [{"date": ts.strftime("%Y-%m-%d"), "value": int(v)} for ts, v in recent.items()],
        }

    def _fear_greed_records(self) -> list[list[Any]]:
        """Rohdaten [timestamp, value, label] – gecacht, mit Stale-Fallback."""
        key = "fear_greed_history"
        records = self._cache.get(key)
        if records is not None:
            return records
        try:
            raw = self._http.get_json(
                self._cfg_fg["base_url"],
                params={"limit": 0, "format": "json"},
                timeout=self._cfg_fg["request_timeout_seconds"],
            )
            records = [
                [int(item["timestamp"]), int(item["value"]), item.get("value_classification", "")]
                for item in raw.get("data", [])
            ]
            if records:
                self._cache.set(key, records, self._cache_cfg["fear_greed_ttl_seconds"])
            return records
        except (ApiError, KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Fear & Greed API nicht erreichbar: {exc}")
            stale = self._cache.get_stale(key)
            return stale[0] if stale else []

    # ------------------------------------------------------------------
    # Reddit
    # ------------------------------------------------------------------

    def get_reddit_sentiment(self, symbol: str, aliases: list[str] | None = None) -> dict[str, Any]:
        """Keyword-basiertes Sentiment aus Reddit-Posts, die den Coin erwähnen.

        Args:
            symbol: Ticker (z.B. "SOL"). Wird nur als eigenständiges Wort in
                GROSSBUCHSTABEN oder mit $-Präfix gezählt ("SOL", "$sol"), damit
                "sol" nicht in "solution" matcht.
            aliases: Zusätzliche Namen (z.B. ["Solana"]), case-insensitiv.

        Returns:
            Dict mit post_count, bullish/bearish/neutral_score, net_sentiment
            (−1…+1), avg_upvotes (None bei RSS), top_titles, source, error.
        """
        sym = symbol.upper()
        key = f"reddit_sentiment_{sym}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        subreddits: list[str] = self._cfg_reddit["subreddits"]
        posts = self._collect_posts(subreddits)
        errors = [p["error"] for p in posts if "error" in p]
        posts = [p for p in posts if "error" not in p]
        sources = {p["source"] for p in posts}

        result = analyze_posts(posts, sym, aliases or [])
        result["subreddits_checked"] = subreddits
        result["total_posts_scanned"] = len(posts)
        result["source"] = "/".join(sorted(sources)) or "nicht erreichbar"
        result["error"] = "; ".join(errors) if not posts and errors else None
        # Fehlschläge nur kurz cachen, damit ein späterer Versuch möglich bleibt
        ttl = self._cache_cfg["reddit_ttl_seconds"] if posts else 300
        self._cache.set(key, result, ttl)
        return result

    def _collect_posts(self, subreddits: list[str]) -> list[dict[str, Any]]:
        """Hot-Posts aller Subreddits (gemeinsam gecacht, damit jeder Coin sie wiederverwendet)."""
        key = "reddit_hot_posts"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if time.monotonic() < self._reddit_cooldown_until:
            return [{"error": "Reddit-Rate-Limit – Pause aktiv"}]

        posts: list[dict[str, Any]] = []
        for sub in subreddits:
            fetched, error = self._fetch_subreddit(sub)
            posts.extend(fetched)
            if error:
                posts.append({"error": f"r/{sub}: {error}"})
                if "429" in error:
                    with self._reddit_lock:
                        self._reddit_cooldown_until = time.monotonic() + _REDDIT_COOLDOWN_SECONDS
                    break  # weitere Anfragen würden nur ebenfalls geblockt
        if any("error" not in p for p in posts):
            self._cache.set(key, posts, self._cache_cfg["reddit_ttl_seconds"])
        return posts

    def _fetch_subreddit(self, sub: str) -> tuple[list[dict[str, Any]], str | None]:
        base = self._cfg_reddit["base_url"]
        limit = self._cfg_reddit["posts_per_subreddit"]
        timeout = self._cfg_reddit["request_timeout_seconds"]
        headers = {"User-Agent": self._cfg_reddit["user_agent"]}

        if not self._json_blocked:
            try:
                data = self._http.get_json(
                    f"{base}/r/{sub}/hot.json", params={"limit": limit}, headers=headers,
                    timeout=timeout, max_attempts=1,
                )
                posts = [
                    {
                        "title": child.get("data", {}).get("title", ""),
                        "score": child.get("data", {}).get("score", 0),
                        "subreddit": sub,
                        "source": "json",
                    }
                    for child in data.get("data", {}).get("children", [])
                ]
                return posts, None
            except ApiError as exc:
                if exc.status_code == 403:
                    self._json_blocked = True  # JSON dauerhaft gesperrt → direkt RSS nutzen
                logger.debug(f"Reddit-JSON r/{sub} fehlgeschlagen ({exc}) – versuche RSS.")

        try:
            response = self._http.get(
                f"{base}/r/{sub}/hot.rss", params={"limit": limit},
                headers={**headers, "Accept": "application/atom+xml"},
                timeout=timeout, max_attempts=1,
            )
            return parse_atom_titles(response.text, sub), None
        except (ApiError, ET.ParseError) as exc:
            logger.info(f"Reddit r/{sub} nicht erreichbar: {exc}")
            status = getattr(exc, "status_code", None)
            return [], f"HTTP {status}" if status else str(exc)[:80]


# ===========================================================================
# Reine Hilfsfunktionen (testbar ohne Netzwerk)
# ===========================================================================

def _records_to_series(records: list[list[Any]]) -> pd.Series:
    if not records:
        return pd.Series(dtype="float64", name="fear_greed")
    idx = pd.to_datetime([int(r[0]) for r in records], unit="s", utc=True)
    series = pd.Series([float(r[1]) for r in records], index=idx, name="fear_greed")
    return series[~series.index.duplicated()].sort_index()


def classify_fear_greed(value: float) -> str:
    """Klassifikation analog alternative.me."""
    if value < 25:
        return "Extreme Fear"
    if value < 46:
        return "Fear"
    if value < 55:
        return "Neutral"
    if value < 76:
        return "Greed"
    return "Extreme Greed"


def parse_atom_titles(xml_text: str, subreddit: str) -> list[dict[str, Any]]:
    """Extrahiert Post-Titel aus einem Reddit-Atom-Feed."""
    root = ET.fromstring(xml_text)
    return [
        {
            "title": (entry.findtext(f"{_ATOM_NS}title") or "").strip(),
            "score": None,
            "subreddit": subreddit,
            "source": "rss",
        }
        for entry in root.iter(f"{_ATOM_NS}entry")
    ]


def mentions(title: str, symbol: str, aliases: list[str]) -> bool:
    """True wenn der Titel den Coin erwähnt (Wortgrenzen-sicher)."""
    sym = re.escape(symbol.upper())
    # Ticker: exakt in Großbuchstaben oder mit $-Präfix (beliebige Schreibweise)
    if re.search(rf"(?<![A-Za-z0-9])(?:{sym}|\$(?i:{sym}))(?![A-Za-z0-9])", title):
        return True
    for alias in aliases:
        if alias and re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", title, re.IGNORECASE):
            return True
    return False


def score_title(title: str) -> int:
    """+1 bullish, −1 bearish, 0 neutral/gemischt."""
    tokens = set(_TOKEN_RE.findall(title.lower()))
    bull = bool(tokens & _BULLISH_KEYWORDS)
    bear = bool(tokens & _BEARISH_KEYWORDS)
    if bull and not bear:
        return 1
    if bear and not bull:
        return -1
    return 0


def analyze_posts(posts: list[dict[str, Any]], symbol: str, aliases: list[str]) -> dict[str, Any]:
    """Filtert relevante Posts und aggregiert das Keyword-Sentiment."""
    relevant = [p for p in posts if p.get("title") and mentions(p["title"], symbol, aliases)]
    if not relevant:
        return {
            "post_count": 0,
            "bullish_score": 0.0,
            "bearish_score": 0.0,
            "neutral_score": 1.0,
            "net_sentiment": 0.0,
            "avg_upvotes": None,
            "top_titles": [],
        }

    scores = [score_title(p["title"]) for p in relevant]
    total = len(relevant)
    bull = sum(1 for s in scores if s > 0)
    bear = sum(1 for s in scores if s < 0)
    upvotes = [p["score"] for p in relevant if p.get("score") is not None]
    ranked = sorted(relevant, key=lambda p: p.get("score") or 0, reverse=True)
    return {
        "post_count": total,
        "bullish_score": round(bull / total, 3),
        "bearish_score": round(bear / total, 3),
        "neutral_score": round((total - bull - bear) / total, 3),
        "net_sentiment": round((bull - bear) / total, 3),
        "avg_upvotes": round(sum(upvotes) / len(upvotes), 1) if upvotes else None,
        "top_titles": [p["title"] for p in ranked[:5]],
    }
