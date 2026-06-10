"""Modell-Evaluierung: MCC, Precision, Log Loss, Baseline-Vergleich.

Der Baseline-Vergleich ("immer NEUTRAL vorhersagen") ist kritisch:
Wenn das ML-Modell schlechter ist als diese triviale Strategie,
sagt das Dashboard dem Nutzer explizit, dass das Modell unzuverlässig ist.

Ehrlichkeit hat hier Priorität vor einem "beeindruckenden" Interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    log_loss,
    matthews_corrcoef,
    precision_score,
    accuracy_score,
)

logger = logging.getLogger(__name__)

# Klassen-Label-Mapping für Ausgaben
_LABEL_MAP = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}


@dataclass
class FoldMetrics:
    """Metriken für einen einzelnen Walk-Forward-Fold.

    Attributes:
        fold: Fold-Nummer (1-basiert).
        accuracy: Gesamt-Accuracy.
        mcc: Matthews Correlation Coefficient (-1 bis +1).
        log_loss_val: Kalibrierungs-Metrik.
        precision_per_class: Precision pro Klasse.
        baseline_accuracy: Accuracy der "immer NEUTRAL"-Strategie.
        beats_baseline: True wenn Modell > Baseline-Accuracy.
        n_up: Anzahl UP-Samples im Testset.
        n_neutral: Anzahl NEUTRAL-Samples im Testset.
        n_down: Anzahl DOWN-Samples im Testset.
    """

    fold: int
    accuracy: float
    mcc: float
    log_loss_val: float
    precision_per_class: dict[str, float]
    baseline_accuracy: float
    beats_baseline: bool
    n_up: int
    n_neutral: int
    n_down: int


@dataclass
class AggregatedMetrics:
    """Aggregierte Metriken über alle Walk-Forward-Folds.

    Attributes:
        avg_accuracy: Durchschnittliche Accuracy (ehrlich anzeigen: oft 52-58%).
        avg_mcc: Durchschnittlicher MCC.
        avg_log_loss: Durchschnittlicher Log Loss.
        avg_precision_per_class: Durchschnittliche Precision pro Klasse.
        baseline_accuracy: Baseline ("immer NEUTRAL").
        beats_baseline_pct: Anteil der Folds wo Modell > Baseline (0.0-1.0).
        model_is_useful: True wenn Modell in >60% der Folds die Baseline schlägt.
        n_folds: Anzahl ausgewerteter Folds.
        fold_accuracies: Accuracy pro Fold (für Stabilitätseinschätzung).
        disclaimer: Ehrlicher Text über Modell-Limitierungen.
    """

    avg_accuracy: float
    avg_mcc: float
    avg_log_loss: float
    avg_precision_per_class: dict[str, float]
    baseline_accuracy: float
    beats_baseline_pct: float
    model_is_useful: bool
    n_folds: int
    fold_accuracies: list[float]
    disclaimer: str


class ModelEvaluator:
    """Berechnet alle Evaluierungs-Metriken aus Walk-Forward-Fold-Ergebnissen.

    Args:
        config: Geladenes config.yaml als Dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._ml_cfg = config["ml"]

    def evaluate_folds(self, fold_results: list[dict[str, Any]]) -> AggregatedMetrics:
        """Wertet alle Walk-Forward-Folds aus und aggregiert die Metriken.

        Args:
            fold_results: Liste von Fold-Dicts aus dem Trainer
                         (mit 'y_true', 'y_pred', 'y_proba', 'fold').

        Returns:
            AggregatedMetrics mit aggregierten Werten und Baseline-Vergleich.
        """
        if not fold_results:
            return self._empty_metrics()

        fold_metrics: list[FoldMetrics] = []
        for fold_data in fold_results:
            metrics = self._evaluate_single_fold(fold_data)
            fold_metrics.append(metrics)

        return self._aggregate(fold_metrics)

    def _evaluate_single_fold(self, fold_data: dict[str, Any]) -> FoldMetrics:
        """Berechnet Metriken für einen einzelnen Fold.

        Args:
            fold_data: Dict mit y_true, y_pred, y_proba, fold.

        Returns:
            FoldMetrics.
        """
        y_true = np.array(fold_data["y_true"])
        y_pred = np.array(fold_data["y_pred"])
        y_proba = np.array(fold_data["y_proba"])
        fold_num = fold_data["fold"]

        # Basis-Metriken
        acc = float(accuracy_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred))

        # Log Loss: robuster bei fehlenden Klassen (labels explizit angeben)
        try:
            ll = float(log_loss(y_true, y_proba, labels=[0, 1, 2]))
        except ValueError:
            ll = float("nan")

        # Precision pro Klasse (zero_division=0 verhindert Fehler bei fehlenden Klassen)
        prec_vals = precision_score(
            y_true, y_pred,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
        precision_per_class = {
            _LABEL_MAP[i]: round(float(p), 3)
            for i, p in enumerate(prec_vals)
        }

        # Baseline: "immer NEUTRAL vorhersagen"
        neutral_class = 1
        baseline_pred = np.full_like(y_pred, neutral_class)
        baseline_acc = float(accuracy_score(y_true, baseline_pred))
        beats_baseline = acc > baseline_acc

        # Klassenverteilung für UI-Info
        n_up = int(np.sum(y_true == 2))
        n_neutral = int(np.sum(y_true == 1))
        n_down = int(np.sum(y_true == 0))

        return FoldMetrics(
            fold=fold_num,
            accuracy=round(acc, 4),
            mcc=round(mcc, 4),
            log_loss_val=round(ll, 4) if not np.isnan(ll) else 0.0,
            precision_per_class=precision_per_class,
            baseline_accuracy=round(baseline_acc, 4),
            beats_baseline=beats_baseline,
            n_up=n_up,
            n_neutral=n_neutral,
            n_down=n_down,
        )

    def _aggregate(self, fold_metrics: list[FoldMetrics]) -> AggregatedMetrics:
        """Aggregiert Fold-Metriken zu Gesamt-Metriken.

        Args:
            fold_metrics: Liste der Einzel-Fold-Metriken.

        Returns:
            AggregatedMetrics.
        """
        accuracies = [m.accuracy for m in fold_metrics]
        mccs = [m.mcc for m in fold_metrics]
        log_losses = [m.log_loss_val for m in fold_metrics if m.log_loss_val > 0]
        baseline_accs = [m.baseline_accuracy for m in fold_metrics]
        beats_count = sum(1 for m in fold_metrics if m.beats_baseline)

        avg_acc = float(np.mean(accuracies))
        avg_mcc = float(np.mean(mccs))
        avg_ll = float(np.mean(log_losses)) if log_losses else 0.0
        avg_baseline = float(np.mean(baseline_accs))
        beats_baseline_pct = beats_count / len(fold_metrics)
        model_is_useful = beats_baseline_pct >= 0.6

        # Precision pro Klasse: Durchschnitt über Folds
        avg_prec: dict[str, float] = {}
        for label in _LABEL_MAP.values():
            vals = [m.precision_per_class.get(label, 0.0) for m in fold_metrics]
            avg_prec[label] = round(float(np.mean(vals)), 3)

        # Ehrlicher Disclaimer basierend auf tatsächlicher Performance
        disclaimer = self._generate_disclaimer(avg_acc, avg_mcc, beats_baseline_pct, model_is_useful)

        return AggregatedMetrics(
            avg_accuracy=round(avg_acc, 4),
            avg_mcc=round(avg_mcc, 4),
            avg_log_loss=round(avg_ll, 4),
            avg_precision_per_class=avg_prec,
            baseline_accuracy=round(avg_baseline, 4),
            beats_baseline_pct=round(beats_baseline_pct, 3),
            model_is_useful=model_is_useful,
            n_folds=len(fold_metrics),
            fold_accuracies=accuracies,
            disclaimer=disclaimer,
        )

    def _generate_disclaimer(
        self,
        accuracy: float,
        mcc: float,
        beats_baseline_pct: float,
        model_is_useful: bool,
    ) -> str:
        """Generiert einen ehrlichen, datengestützten Disclaimer.

        Args:
            accuracy: Durchschnittliche Accuracy.
            mcc: Durchschnittlicher MCC.
            beats_baseline_pct: Anteil Folds die die Baseline schlagen.
            model_is_useful: True wenn Modell als nützlich eingestuft wird.

        Returns:
            Menschenlesbarer Disclaimer-Text.
        """
        acc_pct = int(accuracy * 100)
        beats_pct = int(beats_baseline_pct * 100)

        if not model_is_useful:
            return (
                f"⚠️ Eingeschränkte Zuverlässigkeit: Das Modell schlägt die naive "
                f"'immer NEUTRAL'-Strategie nur in {beats_pct}% der Testperioden. "
                f"Die historische Trefferquote liegt bei {acc_pct}%. "
                f"Signale sollten mit besonderer Vorsicht interpretiert werden."
            )
        if accuracy < 0.55:
            return (
                f"ℹ️ Mäßige Zuverlässigkeit: Historische Trefferquote {acc_pct}% "
                f"(schlägt Baseline in {beats_pct}% der Perioden). "
                f"Krypto-Märkte sind schwer vorherzusagen – dies ist ein normaler Wert."
            )
        return (
            f"✅ Modell schlägt die Baseline in {beats_pct}% der Testperioden "
            f"(historische Accuracy: {acc_pct}%). "
            f"Dennoch: Vergangene Performance garantiert keine zukünftigen Ergebnisse."
        )

    def _empty_metrics(self) -> AggregatedMetrics:
        """Leere Metriken wenn keine Fold-Daten vorhanden."""
        return AggregatedMetrics(
            avg_accuracy=0.0,
            avg_mcc=0.0,
            avg_log_loss=0.0,
            avg_precision_per_class={},
            baseline_accuracy=0.0,
            beats_baseline_pct=0.0,
            model_is_useful=False,
            n_folds=0,
            fold_accuracies=[],
            disclaimer="Keine Evaluierungsdaten verfügbar.",
        )
