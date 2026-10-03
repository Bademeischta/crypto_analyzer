"""Datenqualitätsprüfung und -bereinigung für OHLCV-DataFrames.

Validierungsschritte in Reihenfolge:
  1. Schema-Prüfung (Pflicht-Spalten vorhanden, numerisch)
  2. Duplikate entfernen, chronologisch sortieren
  3. Ungültige Preise (≤ 0) entfernen, OHLC-Konsistenz reparieren
  4. Fehlende Kerzen erkennen (Reindex auf das erwartete Raster) und kurze
     Lücken forward-fillen; nach langen Handelspausen nur das jüngste
     zusammenhängende Segment verwenden
  5. Outlier-Erkennung (Volumen per IQR auf log-Skala, Returns per robustem Z-Score)
  6. Mindestanzahl Kerzen

Binance liefert fehlende Perioden nicht als NaN, sondern lässt die Zeilen
einfach weg – deshalb ist Schritt 4 (Reindex) nötig, um Lücken überhaupt
zu sehen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import Timeframe

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


@dataclass
class ValidationResult:
    """Ergebnis einer Datenvalidierung.

    Attributes:
        df: Bereinigtes DataFrame.
        warnings: Nicht-kritische Hinweise.
        errors: Kritische Fehler (Analyse nicht möglich).
        stats: Kennzahlen zur Datenqualität (für die UI).
    """

    df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True wenn keine kritischen Fehler vorliegen."""
        return not self.errors


class DataValidator:
    """Validiert und bereinigt OHLCV-Rohdaten.

    Args:
        max_consecutive_gaps: Max. aufeinanderfolgend fehlende Kerzen, die aufgefüllt werden.
        volume_iqr_multiplier: IQR-Multiplikator für Volumen-Ausreißer (log-Skala).
        return_zscore_threshold: Robuster Z-Score-Schwellenwert für Return-Ausreißer.
        minimum_samples: Mindestanzahl Kerzen für eine Analyse (Charts).
    """

    def __init__(
        self,
        max_consecutive_gaps: int = 3,
        volume_iqr_multiplier: float = 3.0,
        return_zscore_threshold: float = 6.0,
        minimum_samples: int = 30,
    ) -> None:
        self._max_gaps = max_consecutive_gaps
        self._vol_iqr_mult = volume_iqr_multiplier
        self._ret_zscore = return_zscore_threshold
        self._min_samples = minimum_samples

    def validate(self, df: pd.DataFrame, symbol: str = "", interval: str | None = None) -> ValidationResult:
        """Führt die vollständige Validierungspipeline durch.

        Args:
            df: Rohes OHLCV-DataFrame mit DatetimeIndex.
            symbol: Coin-Symbol für Meldungen.
            interval: Binance-Intervall; ermöglicht die Erkennung fehlender Kerzen.

        Returns:
            ValidationResult mit bereinigtem DataFrame.
        """
        result = ValidationResult(df=df.copy())
        prefix = f"[{symbol}] " if symbol else ""
        result.stats["raw_rows"] = len(df)

        self._check_schema(result, prefix)
        if not result.is_valid:
            return result

        self._dedupe_and_sort(result, prefix)
        self._repair_prices(result, prefix)
        if interval:
            self._fill_missing_bars(result, prefix, Timeframe(interval))
        self._detect_volume_outliers(result, prefix)
        self._detect_return_outliers(result, prefix)
        self._check_minimum_samples(result, prefix)

        result.stats["clean_rows"] = len(result.df)
        if result.warnings:
            logger.info(f"{prefix}Validierungshinweise: {'; '.join(result.warnings)}")
        if result.errors:
            logger.warning(f"{prefix}Validierungsfehler: {'; '.join(result.errors)}")
        return result

    # ------------------------------------------------------------------
    # Einzelne Validierungsschritte
    # ------------------------------------------------------------------

    def _check_schema(self, result: ValidationResult, prefix: str) -> None:
        df = result.df
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            result.errors.append(f"{prefix}Fehlende Spalten: {missing}. Erwartet: {list(REQUIRED_COLUMNS)}.")
            return
        if not isinstance(df.index, pd.DatetimeIndex):
            result.errors.append(f"{prefix}Index muss ein DatetimeIndex sein.")
            return
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        result.df = df

    def _dedupe_and_sort(self, result: ValidationResult, prefix: str) -> None:
        df = result.df
        dupes = int(df.index.duplicated().sum())
        if dupes:
            df = df[~df.index.duplicated(keep="last")]
            result.warnings.append(f"{prefix}{dupes} doppelte Zeitstempel entfernt.")
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            result.warnings.append(f"{prefix}Daten wurden chronologisch umsortiert.")
        result.df = df

    def _repair_prices(self, result: ValidationResult, prefix: str) -> None:
        df = result.df
        price_cols = ["open", "high", "low", "close"]

        invalid = (df[price_cols] <= 0).any(axis=1) | df[price_cols].isna().any(axis=1)
        n_invalid = int(invalid.sum())
        if n_invalid:
            df = df[~invalid].copy()
            result.warnings.append(f"{prefix}{n_invalid} Kerzen mit ungültigen Preisen (≤ 0/NaN) entfernt.")

        # High muss das Maximum, Low das Minimum der Kerze sein
        true_high = df[price_cols].max(axis=1)
        true_low = df[price_cols].min(axis=1)
        inconsistent = int(((df["high"] < true_high) | (df["low"] > true_low)).sum())
        if inconsistent:
            df = df.assign(high=true_high, low=true_low)
            result.warnings.append(f"{prefix}{inconsistent} inkonsistente OHLC-Kerzen korrigiert.")

        negative_vol = df["volume"] < 0
        if negative_vol.any():
            df.loc[negative_vol, "volume"] = np.nan
        df["volume"] = df["volume"].fillna(0.0)
        result.stats["zero_volume_bars"] = int((df["volume"] == 0).sum())
        result.df = df

    def _fill_missing_bars(self, result: ValidationResult, prefix: str, tf: Timeframe) -> None:
        """Erkennt fehlende Kerzen und füllt kurze Lücken auf.

        Aufgefüllte Kerzen erhalten open=high=low=close=letzter Schlusskurs und
        Volumen 0 (Standard-Konvention: "kein Handel"). Bei Lücken, die länger
        als ``max_consecutive_gaps`` sind (Handelspause, Relisting), wird nur
        das jüngste zusammenhängende Segment verwendet – Indikatoren über eine
        wochenlange Pause hinweg wären bedeutungslos.
        """
        df = result.df
        if len(df) < 2:
            return
        full_index = pd.date_range(df.index[0], df.index[-1], freq=tf.delta, tz=df.index.tz, name=df.index.name)
        missing = full_index.difference(df.index)
        result.stats["missing_bars"] = len(missing)
        if len(missing) == 0:
            return

        reindexed = df.reindex(full_index)
        is_gap = reindexed["close"].isna()
        run_id = (is_gap != is_gap.shift()).cumsum()
        run_len = is_gap.groupby(run_id).transform("sum")
        long_gap = is_gap & (run_len > self._max_gaps)

        if long_gap.any():
            last_long_gap_end = reindexed.index[long_gap][-1]
            segment = reindexed[reindexed.index > last_long_gap_end]
            dropped = int((~reindexed.loc[reindexed.index <= last_long_gap_end, "close"].isna()).sum())
            result.warnings.append(
                f"{prefix}Handelspause > {self._max_gaps} Kerzen erkannt (bis {last_long_gap_end:%Y-%m-%d %H:%M}). "
                f"Nur die {int(segment['close'].notna().sum())} Kerzen danach werden verwendet "
                f"({dropped} ältere verworfen)."
            )
            reindexed = segment.copy()
            is_gap = reindexed["close"].isna()

        n_filled = int(is_gap.sum())
        if n_filled:
            close_ff = reindexed["close"].ffill()
            for col in ("open", "high", "low", "close"):
                reindexed[col] = reindexed[col].fillna(close_ff)
            for col in reindexed.columns:
                if col not in ("open", "high", "low", "close"):
                    reindexed[col] = reindexed[col].fillna(0.0)
            result.warnings.append(f"{prefix}{n_filled} fehlende Kerzen aufgefüllt (kein Handel).")

        result.stats["filled_bars"] = n_filled
        result.df = reindexed.dropna(subset=["close"])

    def _detect_volume_outliers(self, result: ValidationResult, prefix: str) -> None:
        """Warnt bei extremem Volumen. Log-Skala, da Volumen stark rechtsschief ist."""
        vol = result.df["volume"]
        positive = vol[vol > 0]
        if len(positive) < 20:
            return
        log_vol = np.log(positive)
        q1, q3 = log_vol.quantile(0.25), log_vol.quantile(0.75)
        upper = q3 + self._vol_iqr_mult * (q3 - q1)
        n_out = int((log_vol > upper).sum())
        result.stats["volume_outliers"] = n_out
        if n_out:
            result.warnings.append(
                f"{prefix}{n_out} extreme Volumen-Spitzen erkannt (> {np.exp(upper):,.0f}). "
                f"Können auf Listings, News oder Manipulation hinweisen."
            )

    def _detect_return_outliers(self, result: ValidationResult, prefix: str) -> None:
        """Warnt bei extremen Kursbewegungen (robuster Z-Score über Median/MAD)."""
        close = result.df["close"]
        if len(close) < 20:
            return
        log_ret = np.log(close / close.shift(1)).dropna()
        median = log_ret.median()
        mad = (log_ret - median).abs().median()
        if mad == 0 or np.isnan(mad):
            return
        robust_z = 0.6745 * (log_ret - median) / mad
        n_ext = int((robust_z.abs() > self._ret_zscore).sum())
        result.stats["return_outliers"] = n_ext
        if n_ext:
            result.warnings.append(
                f"{prefix}{n_ext} extreme Kursbewegungen (|robuster Z| > {self._ret_zscore:g}). "
                f"Krypto-typisch, beeinflusst aber Volatilitäts-Features."
            )

    def _check_minimum_samples(self, result: ValidationResult, prefix: str) -> None:
        n = len(result.df)
        if n < self._min_samples:
            result.errors.append(
                f"{prefix}Nur {n} Kerzen vorhanden, mindestens {self._min_samples} werden benötigt. "
                f"Wähle einen längeren Zeitraum oder einen Coin mit mehr Handelshistorie."
            )
