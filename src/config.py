"""Zentrale Konfiguration: Laden, Validieren, Logging und Zeitrahmen-Umrechnung.

Alle Module erhalten das Config-Dict aus ``load_config``. Zeitangaben in der
Config sind in *Tagen* formuliert und werden über ``Timeframe`` in Kerzen
(Bars) des jeweiligen Intervalls umgerechnet – so bleibt z.B. ein
5-Tage-Horizont auch im 1h-Chart ein 5-Tage-Horizont (= 120 Kerzen).
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_REQUIRED_SECTIONS: tuple[str, ...] = (
    "app", "paths", "api", "cache", "validation", "features", "ml", "backtest", "ui",
)

# Binance-Intervalle -> Minuten pro Kerze
INTERVAL_MINUTES: dict[str, int] = {
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
    "1w": 10080,
}


class ConfigError(ValueError):
    """Ungültige oder unvollständige Konfiguration."""


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Lädt und validiert die YAML-Konfiguration.

    Args:
        path: Pfad zur config.yaml. Default: Projekt-Wurzel.

    Returns:
        Konfiguration als Dict.

    Raises:
        ConfigError: Wenn Datei fehlt oder Pflicht-Sektionen fehlen.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"config.yaml nicht gefunden: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    missing = [s for s in _REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ConfigError(f"Fehlende Sektionen in {config_path.name}: {missing}")

    _validate(config)
    return config


def _validate(config: dict[str, Any]) -> None:
    """Plausibilitätsprüfungen, die sonst erst tief im Training auffallen würden."""
    ml = config["ml"]
    direction = ml["direction"]
    if direction["horizon_days"] <= 0:
        raise ConfigError("ml.direction.horizon_days muss > 0 sein.")
    if direction["label_mode"] not in ("fixed", "volatility_adjusted"):
        raise ConfigError("ml.direction.label_mode muss 'fixed' oder 'volatility_adjusted' sein.")
    if not 0.34 <= ml["confidence_display_threshold"] < 1.0:
        raise ConfigError("ml.confidence_display_threshold muss in [0.34, 1.0) liegen.")
    if ml.get("calibration", "blend") not in ("blend", "none"):
        raise ConfigError("ml.calibration muss 'blend' oder 'none' sein.")
    if ml["engine"] not in ("auto", "lightgbm", "hist_gb"):
        raise ConfigError("ml.engine muss 'auto', 'lightgbm' oder 'hist_gb' sein.")
    vola = ml["volatility"]
    if not 0.0 < vola["low_quantile"] < vola["high_quantile"] < 1.0:
        raise ConfigError("ml.volatility: 0 < low_quantile < high_quantile < 1 verletzt.")
    for interval in config["ui"]["available_intervals"]:
        if interval not in INTERVAL_MINUTES:
            raise ConfigError(f"Unbekanntes Intervall in ui.available_intervals: {interval}")


def per_interval(value: Any, interval: str) -> Any:
    """Liest einen Config-Wert, der skalar oder pro Intervall angegeben sein kann.

    Beispiel: ``{"1h": 90, "default": 365}`` → 90 für "1h", 365 für alle anderen.
    """
    if isinstance(value, dict):
        return value.get(interval, value.get("default"))
    return value


def resolve_path(config_path: Path, relative: str) -> Path:
    """Löst einen Config-Pfad relativ zum Verzeichnis der config.yaml auf."""
    return (config_path.parent / relative).resolve()


def config_fingerprint(config: dict[str, Any], sections: tuple[str, ...] = ("features", "ml")) -> str:
    """Stabiler Hash der ML-relevanten Config-Sektionen.

    Ändert sich ein Feature- oder ML-Parameter, ändert sich der Fingerprint –
    gespeicherte Modelle werden dadurch automatisch invalidiert.
    """
    payload = {s: config.get(s) for s in sections}
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:10]


_LOGGING_CONFIGURED = False


def setup_logging(config: dict[str, Any], config_path: Path | None = None) -> None:
    """Konfiguriert Root-Logging (Konsole + rotierende Logdatei). Idempotent."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    level = getattr(logging, str(config["app"].get("log_level", "INFO")).upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    base = config_path or DEFAULT_CONFIG_PATH
    logs_dir = resolve_path(base, config["paths"]["logs_dir"])
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "crypto_analyzer.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        root.warning("Logdatei konnte nicht angelegt werden – logge nur auf Konsole.")

    # Laute Drittbibliotheken dämpfen
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


@dataclass(frozen=True)
class Timeframe:
    """Umrechnung zwischen Kalenderzeit und Kerzen eines Binance-Intervalls.

    Attributes:
        interval: Binance-Intervall, z.B. "1h", "4h", "1d".
    """

    interval: str

    def __post_init__(self) -> None:
        if self.interval not in INTERVAL_MINUTES:
            raise ValueError(
                f"Unbekanntes Intervall '{self.interval}'. "
                f"Erlaubt: {', '.join(INTERVAL_MINUTES)}"
            )

    @property
    def minutes(self) -> int:
        """Kerzenlänge in Minuten."""
        return INTERVAL_MINUTES[self.interval]

    @property
    def milliseconds(self) -> int:
        """Kerzenlänge in Millisekunden."""
        return self.minutes * 60_000

    @property
    def delta(self) -> pd.Timedelta:
        """Kerzenlänge als Timedelta."""
        return pd.Timedelta(minutes=self.minutes)

    @property
    def bars_per_day(self) -> float:
        """Anzahl Kerzen pro Kalendertag (Krypto handelt 24/7)."""
        return 1440 / self.minutes

    @property
    def periods_per_year(self) -> float:
        """Kerzen pro Jahr (365 Tage, Krypto hat keine Handelspausen)."""
        return 365.0 * self.bars_per_day

    @property
    def annualization(self) -> float:
        """Faktor zur Annualisierung einer Pro-Kerze-Standardabweichung."""
        return math.sqrt(self.periods_per_year)

    def bars(self, days: float, minimum: int = 1) -> int:
        """Rechnet Kalendertage in Kerzen um (mindestens ``minimum``)."""
        return max(minimum, int(round(days * self.bars_per_day)))

    def describe_bars(self, bars: int) -> str:
        """Menschenlesbare Dauer von ``bars`` Kerzen (z.B. '5 Tage', '12 Stunden')."""
        total_minutes = bars * self.minutes
        if total_minutes % 1440 == 0:
            days = total_minutes // 1440
            return f"{days} Tag" if days == 1 else f"{days} Tage"
        hours = total_minutes / 60
        return f"{hours:g} Stunden"
