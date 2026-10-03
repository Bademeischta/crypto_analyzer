"""Disk-basierter Cache mit TTL-Ablauflogik.

Zwei Speicherformate:
  - JSON (``get``/``set``) für kleine, API-nahe Daten.
    Eintrag: { "data": <beliebig>, "timestamp": <unix-float>, "ttl": <sekunden> }
  - Pickle (``get_frame``/``set_frame``) für DataFrames – um Größenordnungen
    schneller als JSON-Records und verlustfrei bei dtypes/Zeitzonen.

Alle Schreibvorgänge sind atomar (temp-Datei + ``os.replace``): Ein Absturz
oder paralleler Zugriff (z.B. OneDrive-Sync) hinterlässt nie halbe Dateien.
Abgelaufene Einträge können mit ``get_stale`` trotzdem gelesen werden – so
kann die App bei API-Ausfällen auf den letzten bekannten Stand zurückfallen.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_JSON_SUFFIX = ".json"
_FRAME_SUFFIX = ".pkl"


class DiskCache:
    """Thread-sicherer Disk-Cache mit TTL.

    Args:
        cache_dir: Verzeichnis für Cache-Dateien. Wird ggf. angelegt.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @property
    def directory(self) -> Path:
        """Cache-Verzeichnis."""
        return self._dir

    # ------------------------------------------------------------------
    # JSON-Einträge
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Gibt gecachte Daten zurück oder None wenn abgelaufen / nicht vorhanden."""
        entry = self._read_json(key)
        if entry is None:
            return None
        age = time.time() - entry["timestamp"]
        if age > entry["ttl"]:
            logger.debug(f"Cache-Miss (abgelaufen, Alter={age:.0f}s): {key}")
            return None
        return entry["data"]

    def get_stale(self, key: str) -> tuple[Any, float] | None:
        """Gibt Daten auch nach TTL-Ablauf zurück (Fallback bei API-Ausfall).

        Returns:
            Tupel (Daten, Alter in Sekunden) oder None wenn nie gecacht.
        """
        entry = self._read_json(key)
        if entry is None:
            return None
        return entry["data"], time.time() - entry["timestamp"]

    def set(self, key: str, data: Any, ttl: int) -> None:
        """Speichert JSON-serialisierbare Daten mit TTL (atomar)."""
        entry = {"data": data, "timestamp": time.time(), "ttl": ttl}
        try:
            payload = json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            logger.warning(f"Cache-Eintrag '{key}' nicht serialisierbar: {exc}")
            return
        self._atomic_write(self._key_to_path(key, _JSON_SUFFIX), payload)

    def get_entry_age(self, key: str) -> float | None:
        """Alter eines JSON-Eintrags in Sekunden oder None."""
        entry = self._read_json(key)
        return None if entry is None else time.time() - entry["timestamp"]

    def is_valid(self, key: str) -> bool:
        """True wenn ein nicht abgelaufener Eintrag existiert."""
        return self.get(key) is not None

    # ------------------------------------------------------------------
    # DataFrame-Einträge (Pickle)
    # ------------------------------------------------------------------

    def get_frame(self, key: str) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        """Lädt einen gespeicherten DataFrame samt Metadaten (ohne TTL-Prüfung).

        Returns:
            Tupel (DataFrame, Metadaten) oder None.
        """
        path = self._key_to_path(key, _FRAME_SUFFIX)
        if not path.exists():
            return None
        try:
            with self._lock, path.open("rb") as fh:
                entry = pickle.load(fh)
            return entry["frame"], entry.get("meta", {})
        except Exception as exc:  # korrupte/inkompatible Datei → ignorieren
            logger.warning(f"Cache-Frame '{key}' unlesbar ({exc}) – wird neu geladen.")
            return None

    def set_frame(self, key: str, frame: pd.DataFrame, meta: dict[str, Any] | None = None) -> None:
        """Speichert einen DataFrame mit Metadaten (atomar)."""
        payload = pickle.dumps({"frame": frame, "meta": meta or {}}, protocol=pickle.HIGHEST_PROTOCOL)
        self._atomic_write(self._key_to_path(key, _FRAME_SUFFIX), payload)

    # ------------------------------------------------------------------
    # Verwaltung
    # ------------------------------------------------------------------

    def invalidate(self, key: str) -> None:
        """Löscht einen Eintrag (beide Formate)."""
        for suffix in (_JSON_SUFFIX, _FRAME_SUFFIX):
            path = self._key_to_path(key, suffix)
            with self._lock:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(f"Cache-Eintrag '{key}' nicht löschbar: {exc}")

    def clear_expired(self) -> int:
        """Löscht abgelaufene und korrupte JSON-Einträge.

        Returns:
            Anzahl gelöschter Dateien.
        """
        deleted = 0
        now = time.time()
        with self._lock:
            for path in self._dir.glob(f"*{_JSON_SUFFIX}"):
                try:
                    with path.open(encoding="utf-8") as fh:
                        entry = json.load(fh)
                    expired = now - entry["timestamp"] > entry["ttl"]
                except (json.JSONDecodeError, OSError, KeyError, TypeError):
                    expired = True
                if expired:
                    try:
                        path.unlink()
                        deleted += 1
                    except OSError:
                        pass
        logger.info(f"Cache bereinigt: {deleted} abgelaufene Einträge gelöscht.")
        return deleted

    def clear_all(self) -> int:
        """Löscht den kompletten Cache. Gibt Anzahl gelöschter Dateien zurück."""
        deleted = 0
        with self._lock:
            for path in list(self._dir.glob(f"*{_JSON_SUFFIX}")) + list(self._dir.glob(f"*{_FRAME_SUFFIX}")):
                try:
                    path.unlink()
                    deleted += 1
                except OSError:
                    pass
        return deleted

    def stats(self) -> dict[str, float]:
        """Anzahl Einträge und Gesamtgröße in MB."""
        files = list(self._dir.glob(f"*{_JSON_SUFFIX}")) + list(self._dir.glob(f"*{_FRAME_SUFFIX}"))
        size = 0
        for f in files:
            try:
                size += f.stat().st_size
            except OSError:
                pass
        return {"entries": len(files), "size_mb": round(size / 1_048_576, 2)}

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------

    def _read_json(self, key: str) -> dict[str, Any] | None:
        path = self._key_to_path(key, _JSON_SUFFIX)
        if not path.exists():
            return None
        try:
            with self._lock, path.open(encoding="utf-8") as fh:
                entry = json.load(fh)
            if not isinstance(entry, dict) or not {"data", "timestamp", "ttl"} <= entry.keys():
                raise KeyError("Unvollständiger Cache-Eintrag")
            return entry
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning(f"Cache-Lesefehler für '{key}': {exc} – Eintrag wird ignoriert.")
            return None

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with self._lock:
            try:
                tmp.write_bytes(payload)
                # os.replace kann unter Windows kurz fehlschlagen, wenn z.B. ein
                # Virenscanner oder OneDrive die Zieldatei gerade geöffnet hat.
                for attempt in range(5):
                    try:
                        os.replace(tmp, path)
                        return
                    except PermissionError:
                        if attempt == 4:
                            raise
                        time.sleep(0.05 * (attempt + 1))
            except OSError as exc:
                logger.warning(f"Cache-Schreibfehler für {path.name}: {exc}")
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def _key_to_path(self, key: str, suffix: str) -> Path:
        """Wandelt einen beliebigen Schlüssel in einen sicheren Dateinamen um.

        Verwendet die ersten 32 Hex-Zeichen des SHA-256-Hashes – eindeutig und
        frei von Sonderzeichen, die unter Windows Probleme machen würden.
        """
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self._dir / f"{digest}{suffix}"
