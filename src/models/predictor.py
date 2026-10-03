"""Inferenz: Wandelt den aktuellen Feature-Vektor in ein Signal mit Kontext um.

Ein Signal wird nur angezeigt, wenn die höchste Klassenwahrscheinlichkeit
die Konfidenzschwelle erreicht. Zusätzlich wird aus der Out-of-Sample-
Historie angegeben, wie verlässlich vergleichbare Signale in der Vergangenheit
waren – eine nackte "Konfidenz" eines Baummodells ist nicht kalibriert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import Timeframe
from src.models.engine import Calibration, predict_proba_aligned

logger = logging.getLogger(__name__)

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
        confidence: Höchste Klassenwahrscheinlichkeit (None wenn unter Schwelle).
        show_signal: False wenn Konfidenz < Schwelle ("Kein klares Signal").
        probabilities: Label → Wahrscheinlichkeit.
        volatility_label / volatility_color / volatility_class: Erwartetes Vola-Regime.
        volatility_probabilities: Label → Wahrscheinlichkeit.
        horizon_bars / horizon_text: Vorhersage-Horizont.
        up_threshold_pct / down_threshold_pct: Return-Schwellen der Klassen in %.
        data_end_date: Letzte abgeschlossene Kerze.
        no_signal_reason: Erklärung bei show_signal=False.
        historical: Kennzahlen früherer Signale derselben Richtung (OOF).
        model_verdict: Urteil der Evaluierung ("signifikant", "schwach", ...).
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
    volatility_probabilities: dict[str, float]
    horizon_bars: int
    horizon_text: str
    up_threshold_pct: float
    down_threshold_pct: float
    data_end_date: str
    no_signal_reason: str = ""
    historical: dict[str, float] = field(default_factory=dict)
    model_verdict: str = ""

    @property
    def horizon_days(self) -> str:
        """Rückwärtskompatibler Alias für den Horizont-Text."""
        return self.horizon_text


class Predictor:
    """Erzeugt Vorhersagen aus trainierten Modellen.

    Args:
        config: Geladenes config.yaml als Dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._threshold = config["ml"]["confidence_display_threshold"]

    def predict(
        self,
        direction_model: Any,
        volatility_model: Any,
        feature_row: pd.Series,
        feature_names: list[str],
        required_features: list[str],
        interval: str,
        horizon_bars: int,
        thresholds: tuple[float, float],
        data_end_date: str,
        signal_stats: dict[str, dict[str, float]] | None = None,
        model_verdict: str = "",
        calibration: Calibration | None = None,
        volatility_calibration: Calibration | None = None,
        class_prior: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        volatility_prior: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ) -> PredictionResult:
        """Erstellt eine Vorhersage für den aktuellen Feature-Vektor.

        Args:
            direction_model / volatility_model: Trainierte Modelle.
            feature_row: Aktueller Feature-Vektor.
            feature_names: Feature-Reihenfolge der Modelle.
            required_features: Features, die nicht NaN sein dürfen.
            interval: Kerzen-Intervall.
            horizon_bars: Horizont in Kerzen.
            thresholds: (down, up) Return-Schwellen.
            data_end_date: Letzte abgeschlossene Kerze.
            signal_stats: OOF-Signalstatistik aus dem Evaluator.
            model_verdict: Urteil der Evaluierung.
            calibration / volatility_calibration: Kalibrierung aus dem Walk-Forward.
            class_prior / volatility_prior: Klassenhäufigkeiten im Training.
        """
        horizon_text = Timeframe(interval).describe_bars(horizon_bars)
        base = {
            "horizon_bars": horizon_bars,
            "horizon_text": horizon_text,
            "down_threshold_pct": thresholds[0] * 100,
            "up_threshold_pct": thresholds[1] * 100,
            "data_end_date": data_end_date,
            "model_verdict": model_verdict,
        }

        missing = [f for f in feature_names if f not in feature_row.index]
        if missing:
            return self._no_signal(base, f"Feature-Inkonsistenz ({len(missing)} fehlen) – Modell neu trainieren.")

        nan_required = [f for f in required_features if pd.isna(feature_row.get(f))]
        if nan_required:
            return self._no_signal(
                base,
                f"{len(nan_required)} Pflicht-Indikatoren haben noch keinen Wert "
                f"(z.B. {nan_required[0]}). Lade mehr historische Daten.",
            )

        X = feature_row[feature_names].to_numpy(dtype=float).reshape(1, -1)
        dir_proba = (calibration or Calibration()).apply(
            predict_proba_aligned(direction_model, X), np.array(class_prior)
        )[0]
        vol_proba = (volatility_calibration or Calibration()).apply(
            predict_proba_aligned(volatility_model, X), np.array(volatility_prior)
        )[0]

        dir_class = int(np.argmax(dir_proba))
        vol_class = int(np.argmax(vol_proba))
        max_conf = float(dir_proba[dir_class])
        show = max_conf >= self._threshold
        label = _DIRECTION_LABELS[dir_class]

        reason = ""
        if not show:
            reason = (
                f"Die höchste Klassenwahrscheinlichkeit liegt bei {max_conf:.0%} "
                f"(Schwelle: {self._threshold:.0%}). Das Modell ist sich aktuell nicht sicher genug."
            )

        return PredictionResult(
            direction_label=label,
            direction_emoji=_DIRECTION_EMOJIS[dir_class],
            direction_class=dir_class,
            confidence=max_conf if show else None,
            show_signal=show,
            probabilities={_DIRECTION_LABELS[i]: round(float(p), 3) for i, p in enumerate(dir_proba)},
            volatility_label=_VOLATILITY_LABELS[vol_class],
            volatility_color=_VOLATILITY_COLORS[vol_class],
            volatility_class=vol_class,
            volatility_probabilities={_VOLATILITY_LABELS[i]: round(float(p), 3) for i, p in enumerate(vol_proba)},
            no_signal_reason=reason,
            historical=(signal_stats or {}).get(label, {}) if show else {},
            **base,
        )

    @staticmethod
    def _no_signal(base: dict[str, Any], reason: str) -> PredictionResult:
        return PredictionResult(
            direction_label="NEUTRAL",
            direction_emoji="🟡",
            direction_class=1,
            confidence=None,
            show_signal=False,
            probabilities={"BEARISH": 0.0, "NEUTRAL": 0.0, "BULLISH": 0.0},
            volatility_label="–",
            volatility_color="gray",
            volatility_class=1,
            volatility_probabilities={},
            no_signal_reason=reason,
            **base,
        )
