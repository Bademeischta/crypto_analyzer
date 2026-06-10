"""Inferenz: Wandelt einen Feature-Vektor in ein Signal mit Konfidenz um.

Konfidenz wird NUR angezeigt wenn max. Klassenwahrscheinlichkeit > 0.55.
Darunter gibt das System "Kein klares Signal" aus (ehrlicher als erzwungene Labels).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sprachangepasste Labels für die UI
_DIRECTION_LABELS: dict[int, str] = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
_DIRECTION_EMOJIS: dict[int, str] = {0: "🔴", 1: "🟡", 2: "🟢"}
_VOLATILITY_LABELS: dict[int, str] = {0: "NIEDRIG", 1: "MITTEL", 2: "HOCH"}
_VOLATILITY_COLORS: dict[int, str] = {0: "green", 1: "orange", 2: "red"}


@dataclass
class PredictionResult:
    """Vollständiges Vorhersage-Ergebnis für das UI.

    Attributes:
        direction_label: "BEARISH", "NEUTRAL" oder "BULLISH".
        direction_emoji: Passender Emoji.
        direction_class: Numerische Klasse (0, 1, 2).
        confidence: Konfidenz als float (0.0-1.0). None wenn unter Schwelle.
        show_signal: False wenn Konfidenz < Threshold (ehrliches "Kein Signal").
        probabilities: Dict label -> Wahrscheinlichkeit für alle 3 Klassen.
        volatility_label: "NIEDRIG", "MITTEL" oder "HOCH".
        volatility_color: CSS-Farbname.
        volatility_class: Numerische Klasse (0, 1, 2).
        horizon_days: Vorhersage-Horizont in Tagen.
        data_end_date: Datum des letzten Trainingsdatenpunkts.
        no_signal_reason: Erklärung warum kein Signal angezeigt wird (wenn show_signal=False).
    """

    direction_label: str
    direction_emoji: str
    direction_class: int
    confidence: float | None
    show_signal: bool
    probabilities: dict[str, float]
    volatility_label: str
    volatility_color: str
    volatility_class: int
    horizon_days: int
    data_end_date: str
    no_signal_reason: str = ""


class Predictor:
    """Erzeugt Vorhersagen aus trainierten LightGBM-Modellen.

    Args:
        config: Geladenes config.yaml als Dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._ml_cfg = config["ml"]
        self._confidence_threshold = config["ml"]["confidence_display_threshold"]
        self._horizon = config["ml"]["direction"]["horizon_days"]

    def predict(
        self,
        direction_model: lgb.LGBMClassifier,
        volatility_model: lgb.LGBMClassifier,
        feature_row: pd.Series,
        feature_names: list[str],
        data_end_date: str,
    ) -> PredictionResult:
        """Erstellt eine vollständige Vorhersage für einen Feature-Vektor.

        Args:
            direction_model: Trainiertes Richtungs-Klassifikationsmodell.
            volatility_model: Trainiertes Volatilitäts-Klassifikationsmodell.
            feature_row: Letzter Feature-Vektor (aus FeatureMatrix.last_row).
            feature_names: Erwartete Feature-Namen (für Reihenfolge-Sicherheit).
            data_end_date: Datum des letzten bekannten Datenpunkts.

        Returns:
            PredictionResult mit allen Informationen für das UI.
        """
        # Feature-Vektor in korrekte Reihenfolge bringen
        try:
            X = feature_row[feature_names].values.reshape(1, -1)
        except KeyError as exc:
            logger.error(f"Feature-Mismatch: {exc}")
            return self._no_signal_result(
                data_end_date,
                "Feature-Inkonsistenz: Modell neu trainieren.",
            )

        # NaN-Check: Wenn der letzte Feature-Vektor NaN enthält → kein Signal
        if np.any(np.isnan(X)):
            nan_count = np.sum(np.isnan(X))
            return self._no_signal_result(
                data_end_date,
                f"{nan_count} Features haben keinen Wert (zu wenig Daten für Indikatoren). "
                f"Lade mehr historische Daten.",
            )

        # Richtungs-Vorhersage
        dir_proba = direction_model.predict_proba(X)[0]
        dir_class = int(np.argmax(dir_proba))
        max_confidence = float(dir_proba[dir_class])

        # Konfidenz-Threshold: Unter 55% kein Signal
        show_signal = max_confidence >= self._confidence_threshold
        confidence = max_confidence if show_signal else None
        no_signal_reason = (
            ""
            if show_signal
            else (
                f"Die höchste Klassenwahrscheinlichkeit liegt bei "
                f"{max_confidence:.0%} (Schwelle: {self._confidence_threshold:.0%}). "
                f"Das Modell ist sich bei diesem Coin aktuell nicht sicher genug."
            )
        )

        probabilities = {
            _DIRECTION_LABELS[i]: round(float(p), 3)
            for i, p in enumerate(dir_proba)
        }

        # Volatilitäts-Vorhersage
        vola_proba = volatility_model.predict_proba(X)[0]
        vola_class = int(np.argmax(vola_proba))

        return PredictionResult(
            direction_label=_DIRECTION_LABELS[dir_class],
            direction_emoji=_DIRECTION_EMOJIS[dir_class],
            direction_class=dir_class,
            confidence=confidence,
            show_signal=show_signal,
            probabilities=probabilities,
            volatility_label=_VOLATILITY_LABELS[vola_class],
            volatility_color=_VOLATILITY_COLORS[vola_class],
            volatility_class=vola_class,
            horizon_days=self._horizon,
            data_end_date=data_end_date,
            no_signal_reason=no_signal_reason,
        )

    def _no_signal_result(self, data_end_date: str, reason: str) -> PredictionResult:
        """Erstellt ein 'Kein Signal'-Ergebnis mit erklärendem Grund.

        Args:
            data_end_date: Datum des letzten Datenpunkts.
            reason: Menschenlesbare Erklärung.

        Returns:
            PredictionResult mit show_signal=False.
        """
        return PredictionResult(
            direction_label="NEUTRAL",
            direction_emoji="🟡",
            direction_class=1,
            confidence=None,
            show_signal=False,
            probabilities={"BEARISH": 0.0, "NEUTRAL": 1.0, "BULLISH": 0.0},
            volatility_label="MITTEL",
            volatility_color="orange",
            volatility_class=1,
            horizon_days=self._horizon,
            data_end_date=data_end_date,
            no_signal_reason=reason,
        )
