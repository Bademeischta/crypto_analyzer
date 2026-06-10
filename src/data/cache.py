"""Disk-basierter Cache mit TTL-Ablauflogik.

Jeder Cache-Eintrag ist eine JSON-Datei mit der Struktur:
  { "data": <beliebig>, "timestamp": <unix-float>, "ttl": <sekunden-int> }
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DiskCache:
    """Thread-unsicherer Disk-Cache (Streamlit läuft single-threaded, reicht aus).

    Args:
        cache_dir: Verzeichnis für Cache-Dateien. Wird ggf. angelegt.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Gibt gecachte Daten zurück oder None wenn abgelaufen / nicht vorhanden.

        Args:
            key: Beliebiger String-Schlüssel.

        Returns:
            Gecachte Daten oder None.
        """
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            with path.open(encoding="utf-8") as fh:
                entry = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Cache-Lesefehler für '{key}': {exc} – Eintrag wird ignoriert.")
            return None

        age = time.time() - entry["timestamp"]
        if age > entry["ttl"]:
            logger.debug(f"Cache-Miss (abgelaufen, Alter={age:.0f}s, TTL={entry['ttl']}s): {key}")
            return None

        logger.debug(f"Cache-Hit (Alter={age:.0f}s): {key}")
        return entry["data"]

    def set(self, key: str, data: Any, ttl: int) -> None:
        """Speichert Daten mit TTL im Cache.

        Args:
            key: Schlüssel.
            data: JSON-serialisierbare Daten.
            ttl: Time-to-live in Sekunden.
        """
        path = self._key_to_path(key)
        entry = {"data": data, "timestamp": time.time(), "ttl": ttl}
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(entry, fh, ensure_ascii=False, separators=(",", ":"))
        except OSError as exc:
            logger.warning(f"Cache-Schreibfehler für '{key}': {exc}")

    def invalidate(self, key: str) -> None:
        """Löscht einen einzelnen Cache-Eintrag.

        Args:
            key: Schlüssel des zu löschenden Eintrags.
        """
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            logger.debug(f"Cache-Eintrag gelöscht: {key}")

    def clear_expired(self) -> int:
        """Löscht alle abgelaufenen Einträge. Gibt Anzahl gelöschter Dateien zurück.

        Returns:
            Anzahl der gelöschten Cache-Einträge.
        """
        deleted = 0
        now = time.time()
        for path in self._dir.glob("*.json"):
            try:
                with path.open(encoding="utf-8") as fh:
                    entry = json.load(fh)
                if now - entry["timestamp"] > entry["ttl"]:
                    path.unlink()
                    deleted += 1
            except (json.JSONDecodeError, OSError, KeyError):
                path.unlink()
                deleted += 1
        logger.info(f"Cache bereinigt: {deleted} abgelaufene Einträge gelöscht.")
        return deleted

    def get_entry_age(self, key: str) -> float | None:
        """Gibt das Alter eines Cache-Eintrags in Sekunden zurück.

        Args:
            key: Schlüssel.

        Returns:
            Alter in Sekunden oder None wenn nicht vorhanden.
        """
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as fh:
                entry = json.load(fh)
            return time.time() - entry["timestamp"]
        except (json.JSONDecodeError, OSError, KeyError):
            return None

    def is_valid(self, key: str) -> bool:
        """Prüft ob ein Cache-Eintrag existiert und noch nicht abgelaufen ist.

        Args:
            key: Schlüssel.

        Returns:
            True wenn Eintrag gültig (nicht abgelaufen).
        """
        return self.get(key) is not None

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------

    def _key_to_path(self, key: str) -> Path:
        """Wandelt einen beliebigen Schlüssel in einen sicheren Dateinamen um.

        Verwendet SHA-256 der ersten 16 Hex-Zeichen für Eindeutigkeit
        ohne Filesystem-Probleme durch Sonderzeichen.

        Args:
            key: Beliebiger String.

        Returns:
            Absoluter Pfad zur Cache-Datei.
        """
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self._dir / f"{digest}.json"
