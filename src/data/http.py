"""Gemeinsamer HTTP-Client für alle Datenquellen.

Features:
  - Connection-Pooling über eine ``requests.Session`` (spart TLS-Handshakes)
  - Exponentielles Backoff mit Jitter bei 429/5xx und Netzwerkfehlern
  - Respektiert ``Retry-After``-Header (z.B. CoinGecko, Reddit)
  - Client-seitiges Rate-Limiting pro Host (thread-safe), damit parallele
    Requests das API-Limit gar nicht erst reißen
  - Typisierte Exceptions statt generischer ``requests``-Fehler
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

# Obergrenze für eine einzelne Wartezeit – ein Dashboard soll nicht minutenlang hängen
_MAX_SINGLE_WAIT_SECONDS = 20.0


class ApiError(RuntimeError):
    """Fehlerhafte HTTP-Antwort einer externen API.

    Attributes:
        status_code: HTTP-Status (None bei Netzwerkfehlern).
        url: Angefragte URL.
        body: Gekürzter Response-Body (für Fehlermeldungen).
    """

    def __init__(self, message: str, status_code: int | None = None, url: str = "", body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.body = body


class RateLimitError(ApiError):
    """API-Rate-Limit auch nach allen Retries überschritten (HTTP 429)."""


class HttpClient:
    """Thread-sicherer HTTP-Client mit Retry, Backoff und Rate-Limiting.

    Args:
        retry_cfg: Sektion ``api.retry`` der config.yaml.
        user_agent: User-Agent-Header für alle Requests.
        host_rate_limits: Mapping Hostname -> max. Requests pro Minute.
        session: Optional eigene Session (für Tests).
        sleep: Schlaf-Funktion (für Tests injizierbar).
    """

    def __init__(
        self,
        retry_cfg: dict[str, Any],
        user_agent: str = "CryptoAnalyzer/2.0",
        host_rate_limits: dict[str, float] | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._max_attempts = int(retry_cfg.get("max_attempts", 4))
        self._initial_backoff = float(retry_cfg.get("initial_backoff_seconds", 1.0))
        self._multiplier = float(retry_cfg.get("backoff_multiplier", 2.0))
        self._retryable = set(retry_cfg.get("rate_limit_status_codes", [429])) | set(
            retry_cfg.get("transient_status_codes", [500, 502, 503, 504])
        )
        self._sleep = sleep

        self._session = session or requests.Session()
        if session is None:
            adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
        self._session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})

        self._min_interval: dict[str, float] = {
            host: 60.0 / rpm for host, rpm in (host_rate_limits or {}).items() if rpm > 0
        }
        self._next_slot: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 15,
        max_attempts: int | None = None,
    ) -> requests.Response:
        """HTTP-GET mit Retry-Logik.

        Args:
            url: Ziel-URL.
            params: Query-Parameter.
            headers: Zusätzliche Header.
            timeout: Timeout pro Versuch in Sekunden.
            max_attempts: Überschreibt die Anzahl Versuche aus der Config.

        Returns:
            Response mit Status 200.

        Raises:
            RateLimitError: Wenn 429 auch nach allen Versuchen bestehen bleibt.
            ApiError: Bei nicht-retrybaren HTTP-Fehlern oder dauerhaften Netzwerkproblemen.
        """
        attempts = max_attempts or self._max_attempts
        host = urlparse(url).netloc
        last_error: ApiError | None = None

        for attempt in range(attempts):
            is_last = attempt == attempts - 1
            self._throttle(host)
            try:
                response = self._session.get(url, params=params, headers=headers, timeout=timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = ApiError(f"Netzwerkfehler bei {host}: {exc}", None, url)
                if is_last:
                    break
                wait = self._backoff(attempt)
                logger.warning(f"{last_error} (Versuch {attempt + 1}/{attempts}) – warte {wait:.1f}s")
                self._sleep(wait)
                continue

            if response.status_code == 200:
                return response

            body = _truncate(response.text)
            if response.status_code in self._retryable:
                error_cls = RateLimitError if response.status_code == 429 else ApiError
                last_error = error_cls(
                    f"HTTP {response.status_code} von {host}", response.status_code, url, body
                )
                if is_last:
                    break
                wait = self._retry_after(response) or self._backoff(attempt)
                logger.warning(f"{last_error} (Versuch {attempt + 1}/{attempts}) – warte {wait:.1f}s")
                self._sleep(wait)
                continue

            # Nicht-retrybar (400, 403, 404, 451, ...) → sofort melden
            raise ApiError(
                f"HTTP {response.status_code} von {host}: {body}",
                response.status_code,
                url,
                body,
            )

        assert last_error is not None
        raise last_error

    def get_json(self, url: str, **kwargs: Any) -> Any:
        """Wie ``get``, gibt aber das geparste JSON zurück.

        Raises:
            ApiError: Auch wenn die Antwort kein gültiges JSON ist.
        """
        response = self.get(url, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(f"Ungültiges JSON von {url}: {exc}", response.status_code, url) from exc

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------

    def _throttle(self, host: str) -> None:
        """Reserviert thread-sicher den nächsten freien Request-Slot für einen Host."""
        interval = self._min_interval.get(host)
        if not interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(host, 0.0))
            self._next_slot[host] = slot + interval
        wait = slot - now
        if wait > 0:
            self._sleep(wait)

    def _backoff(self, attempt: int) -> float:
        base = self._initial_backoff * (self._multiplier ** attempt)
        # Jitter verhindert, dass parallele Threads im Gleichschritt erneut anfragen
        return min(_MAX_SINGLE_WAIT_SECONDS, base * (1.0 + random.uniform(0.0, 0.25)))

    @staticmethod
    def _retry_after(response: requests.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return min(_MAX_SINGLE_WAIT_SECONDS, max(0.0, float(value)))
        except ValueError:
            return None


def _truncate(text: str, limit: int = 300) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"
