"""Datenqualitätsprüfung und -bereinigung für OHLCV-DataFrames.

Validierungsschritte in Reihenfolge:
  1. Schema-Prüfung (Pflicht-Spalten vorhanden?)
  2. Chronologische Reihenfolge
  3. Forward-Fill bei Lücken (max N aufeinanderfolgende)
  4. Outlier-Erkennung (Volume IQR, Returns Z-Score)
  5. Mindestanzahl Samples
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


@dataclass
class ValidationResult:
    """Ergebnis einer Datenvalidierung.

    Attributes:
        df: Bereinigtes DataFrame.
        warnings: Liste von Warnungen (nicht kritisch).
        errors: Liste von Fehlern (kritisch, Analyse nicht möglich).
        is_valid: True wenn Daten für Analyse verwendbar sind.
    """

    df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True wenn keine kritischen Fehler vorliegen."""
        return len(self.errors) == 0


class DataValidator:
    """Validiert und bereinigt OHLCV-Rohdaten.

    Args:
        max_consecutive_gaps: Max. aufeinanderfolgende NaN-Lücken für Forward-Fill.
        volume_iqr_multiplier: Multiplikator für IQR-basierte Volumen-Outlier-Erkennung.
        return_zscore_threshold: Z-Score-Schwelle für Return-Outlier-Erkennung.
        minimum_samples: Mindestanzahl Datenpunkte für eine valide Analyse.
    """

    def __init__(
        self,
        max_consecutive_gaps: int = 3,
        volume_iqr_multiplier: float = 3.0,
        return_zscore_threshold: float = 4.0,
        minimum_samples: int = 200,
    ) -> None:
        self._max_gaps = max_consecutive_gaps
        self._vol_iqr_mult = volume_iqr_multiplier
        self._ret_zscore = return_zscore_threshold
        self._min_samples = minimum_samples

    def validate(self, df: pd.DataFrame, symbol: str = "") -> ValidationResult:
        """Führt die vollständige Validierungspipeline durch.

        Args:
            df: Rohes OHLCV-DataFrame. Muss Spalten open/high/low/close/volume haben.
            symbol: Coin-Symbol für Fehlermeldungen (optional).

        Returns:
            ValidationResult mit bereinigtem DataFrame und Warn-/Fehler-Liste.
        """
        result = ValidationResult(df=df.copy())
        prefix = f"[{symbol}] " if symbol else ""

        result = self._check_schema(result, prefix)
        if not result.is_valid:
            return result

        result = self._check_chronological_order(result, prefix)
        result = self._forward_fill_gaps(result, prefix)
        result = self._detect_volume_outliers(result, prefix)
        result = self._detect_return_outliers(result, prefix)
        result = self._check_minimum_samples(result, prefix)

        if result.warnings:
            logger.warning(f"{prefix}Validierungswarnungen: {'; '.join(result.warnings)}")
        if result.errors:
            logger.error(f"{prefix}Validierungsfehler: {'; '.join(result.errors)}")

        return result

    # ------------------------------------------------------------------
    # Einzelne Validierungsschritte
    # ------------------------------------------------------------------

    def _check_schema(self, result: ValidationResult, prefix: str) -> ValidationResult:
        """Prüft ob alle Pflicht-Spalten vorhanden sind."""
        df = result.df
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            result.errors.append(
                f"{prefix}Fehlende Spalten: {missing}. "
                f"Erwartet werden: {list(REQUIRED_COLUMNS)}."
            )
            return result

        # Spalten in float konvertieren, falls sie als object ankommen
        for col in REQUIRED_COLUMNS:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    result.errors.append(
                        f"{prefix}Spalte '{col}' kann nicht in Zahlen konvertiert werden."
                    )

        result.df = df
        return result

    def _check_chronological_order(self, result: ValidationResult, prefix: str) -> ValidationResult:
        """Stellt sicher dass der Index aufsteigend sortiert ist."""
        df = result.df
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            result.warnings.append(f"{prefix}Daten wurden chronologisch umsortiert.")
        result.df = df
        return result

    def _forward_fill_gaps(self, result: ValidationResult, prefix: str) -> ValidationResult:
        """Forward-füllt Lücken bis zum konfigurierten Maximum, danach droppen."""
        df = result.df
        total_rows_before = len(df)

        # Lücken identifizieren (NaN in close)
        null_mask = df["close"].isnull()
        if null_mask.sum() == 0:
            return result

        # Aufeinanderfolgende NaN-Gruppen messen
        consecutive_groups = (null_mask != null_mask.shift()).cumsum()
        group_sizes = null_mask.groupby(consecutive_groups).sum()
        long_gaps = (group_sizes > self._max_gaps).sum()

        # Forward-fill mit limit
        df = df.ffill(limit=self._max_gaps)

        # Verbleibende NaN droppen (Gruppen die zu lang waren)
        rows_with_nan = df[REQUIRED_COLUMNS].isnull().any(axis=1)
        df = df[~rows_with_nan]

        dropped = total_rows_before - len(df)
        if dropped > 0:
            result.warnings.append(
                f"{prefix}{dropped} Zeilen mit Datenlücken entfernt "
                f"({long_gaps} Gruppen mit >{self._max_gaps} aufeinanderfolgenden Lücken)."
            )

        result.df = df
        return result

    def _detect_volume_outliers(self, result: ValidationResult, prefix: str) -> ValidationResult:
        """Markiert extreme Volumen-Ausreißer per IQR. Dropt sie nicht, nur Warnung."""
        df = result.df
        vol = df["volume"]
        if len(vol) < 10:
            return result

        q1 = vol.quantile(0.25)
        q3 = vol.quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + self._vol_iqr_mult * iqr

        outlier_count = (vol > upper_fence).sum()
        if outlier_count > 0:
            result.warnings.append(
                f"{prefix}{outlier_count} extreme Volumen-Ausreißer erkannt "
                f"(>{upper_fence:,.0f} = {self._vol_iqr_mult}×IQR). "
                f"Können auf Listing-Events oder Manipulation hinweisen."
            )

        result.df = df
        return result

    def _detect_return_outliers(self, result: ValidationResult, prefix: str) -> ValidationResult:
        """Markiert extreme Preisveränderungen per Z-Score. Nur Warnung, kein Drop."""
        df = result.df
        if len(df) < 10:
            return result

        log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
        if len(log_ret) == 0:
            return result

        mean = log_ret.mean()
        std = log_ret.std()
        if std == 0:
            return result

        z_scores = (log_ret - mean) / std
        extreme_count = (z_scores.abs() > self._ret_zscore).sum()

        if extreme_count > 0:
            result.warnings.append(
                f"{prefix}{extreme_count} extreme Preisbewegungen erkannt "
                f"(|Z-Score| > {self._ret_zscore}). "
                f"Krypto-typisch, aber beachte mögliche Manipulation."
            )

        result.df = df
        return result

    def _check_minimum_samples(self, result: ValidationResult, prefix: str) -> ValidationResult:
        """Prüft ob genug Daten für ein aussagekräftiges ML-Training vorhanden sind."""
        n = len(result.df)
        if n < self._min_samples:
            result.errors.append(
                f"{prefix}Nur {n} Datenpunkte vorhanden, "
                f"mindestens {self._min_samples} werden für ML benötigt. "
                f"Wähle einen längeren Zeitraum oder einen Coin mit mehr Handelshistorie."
            )
        return result
