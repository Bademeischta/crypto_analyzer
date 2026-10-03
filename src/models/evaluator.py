"""Modell-Evaluierung auf echten Out-of-Sample-Vorhersagen (Walk-Forward-OOF).

Ehrlichkeit vor Eindruck:
  * Hauptkriterium ist der **Log-Loss-Skill** gegenüber der Prior-Baseline
    (Klassenhäufigkeiten des jeweiligen Trainingsblocks = beste Vorhersage
    ohne jede Marktinformation). Log-Loss ist eine "proper scoring rule":
    Er belohnt nur echte, gut kalibrierte Information – anders als Accuracy,
    die man z.B. durch ständiges Raten der Mehrheitsklasse aufblähen kann.
  * Signifikanz per einseitigem t-Test auf Block-Mittelwerten der
    Log-Loss-Differenzen. Da sich die h-Kerzen-Labels überlappen, sind
    benachbarte Fehler korreliert; Blöcke der Länge h machen die Stichproben
    näherungsweise unabhängig (n/h *effektive* Stichproben).
  * Zusätzlich: Accuracy vs. Mehrheitsklasse (Binomialtest), MCC,
    balancierte Accuracy.
  * Signal-Qualität: Wie oft lag das Modell richtig, *wenn* es ein Signal
    oberhalb der Konfidenzschwelle gab – und was passierte danach im Schnitt
    mit dem Kurs?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

DIRECTION_NAMES: dict[int, str] = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
VOLATILITY_NAMES: dict[int, str] = {0: "NIEDRIG", 1: "MITTEL", 2: "HOCH"}
_PROBA_COLS = ("p_down", "p_neutral", "p_up")
_LABELS = [0, 1, 2]


@dataclass
class AggregatedMetrics:
    """Out-of-Sample-Metriken des Richtungsmodells.

    Attributes:
        n_samples: Anzahl OOF-Vorhersagen.
        n_folds: Anzahl Walk-Forward-Folds.
        effective_samples: Überlappungsfreie Stichproben (n / Horizont).
        accuracy / balanced_accuracy / mcc / log_loss / brier: Klassifikationsmetriken.
        log_loss_prior: Log-Loss der Prior-Baseline (Klassenhäufigkeiten).
        skill_score: 1 − LogLoss_Modell / LogLoss_Prior (> 0 = Modell informativer).
        baseline_accuracy: Accuracy der Mehrheitsklassen-Baseline.
        neutral_baseline_accuracy: Accuracy von "immer NEUTRAL".
        beats_baseline_pct: Anteil der Folds, in denen das Modell die Prior-Baseline im Log-Loss schlägt.
        p_value: t-Test Log-Loss-Skill > 0 (einseitig, blockweise).
        accuracy_p_value: Binomialtest Accuracy > Mehrheitsklasse (einseitig).
        verdict: "signifikant", "schwach" oder "keine Vorhersagekraft".
        model_is_useful: True bei signifikanter Überlegenheit.
        confusion: 3×3-Konfusionsmatrix (Zeilen = wahr, Spalten = vorhergesagt).
        precision_per_class / recall_per_class: je Klasse.
        fold_table: Metriken je Fold.
        calibration: Zuverlässigkeitsdiagramm-Daten (vorhergesagt vs. beobachtet).
        signal_stats: Qualität der Signale über der Konfidenzschwelle je Richtung.
        unconditional_avg_return: Durchschnittlicher Horizont-Return ohne Signal.
        disclaimer: Ehrlicher, datenbasierter Einordnungstext.
    """

    n_samples: int
    n_folds: int
    effective_samples: int
    accuracy: float
    balanced_accuracy: float
    mcc: float
    log_loss: float
    brier: float
    log_loss_prior: float
    skill_score: float
    baseline_accuracy: float
    neutral_baseline_accuracy: float
    beats_baseline_pct: float
    p_value: float
    accuracy_p_value: float
    verdict: str
    model_is_useful: bool
    confusion: list[list[int]]
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    fold_table: list[dict[str, Any]]
    calibration: list[dict[str, float]]
    signal_stats: dict[str, dict[str, float]]
    unconditional_avg_return: float
    disclaimer: str

    @property
    def fold_accuracies(self) -> list[float]:
        return [f["accuracy"] for f in self.fold_table]

    @property
    def avg_accuracy(self) -> float:
        return self.accuracy

    @property
    def avg_mcc(self) -> float:
        return self.mcc


@dataclass
class VolatilityMetrics:
    """Out-of-Sample-Metriken des Volatilitätsmodells.

    Volatilität ist – anders als die Kursrichtung – nachweislich teilweise
    vorhersagbar (Volatility Clustering: ruhige Phasen folgen auf ruhige,
    hektische auf hektische).
    """

    accuracy: float
    baseline_accuracy: float
    mcc: float
    balanced_accuracy: float
    skill_score: float = 0.0
    p_value: float = float("nan")
    verdict: str = ""
    confusion: list[list[int]] = field(default_factory=list)


class ModelEvaluator:
    """Berechnet alle Evaluierungs-Metriken aus den OOF-Vorhersagen.

    Args:
        config: Geladenes config.yaml als Dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._threshold = config["ml"]["confidence_display_threshold"]
        self._alpha = config["ml"]["significance_level"]

    def evaluate(self, oof: pd.DataFrame, horizon_bars: int) -> AggregatedMetrics | None:
        """Wertet die OOF-Richtungsvorhersagen aus (None bei leeren Daten)."""
        if oof is None or oof.empty:
            return None

        y = oof["y_true"].to_numpy(dtype=int)
        pred = oof["pred"].to_numpy(dtype=int)
        proba = oof[list(_PROBA_COLS)].to_numpy(dtype=float)
        majority = oof["majority"].to_numpy(dtype=int)
        prior = _prior_matrix(oof, y)
        idx = np.arange(len(y))

        ll_model_i = -np.log(np.clip(proba[idx, y], 1e-12, 1.0))
        ll_prior_i = -np.log(np.clip(prior[idx, y], 1e-12, 1.0))
        ll_model, ll_prior = float(ll_model_i.mean()), float(ll_prior_i.mean())
        skill = 1.0 - ll_model / ll_prior if ll_prior > 0 else 0.0
        p_value = _block_ttest_p(ll_prior_i - ll_model_i, max(1, horizon_bars))

        acc = float(accuracy_score(y, pred))
        base_acc = float(np.mean(y == majority))
        neutral_acc = float(np.mean(y == 1))
        mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 else 0.0
        n_eff = max(1, len(y) // max(1, horizon_bars))
        acc_p = self._binomial_p(acc, base_acc, n_eff)

        fold_table = []
        fold_ids = oof["fold"].to_numpy()
        for fold in np.unique(fold_ids):
            pos = np.flatnonzero(fold_ids == fold)
            g_y, g_pred = y[pos], pred[pos]
            fold_table.append({
                "fold": int(fold),
                "start": oof.index[pos[0]],
                "end": oof.index[pos[-1]],
                "n": len(pos),
                "accuracy": round(float(np.mean(g_y == g_pred)), 4),
                "baseline": round(float(np.mean(g_y == majority[pos])), 4),
                "log_loss": round(float(ll_model_i[pos].mean()), 4),
                "log_loss_prior": round(float(ll_prior_i[pos].mean()), 4),
                "mcc": round(float(matthews_corrcoef(g_y, g_pred)) if len(np.unique(g_y)) > 1 else 0.0, 4),
            })
        beats_pct = float(np.mean([f["log_loss"] < f["log_loss_prior"] for f in fold_table]))

        verdict, useful = self._verdict(p_value, skill, mcc)
        return AggregatedMetrics(
            n_samples=len(y),
            n_folds=len(fold_table),
            effective_samples=n_eff,
            accuracy=round(acc, 4),
            balanced_accuracy=round(float(balanced_accuracy_score(y, pred)), 4),
            mcc=round(mcc, 4),
            log_loss=round(ll_model, 4),
            brier=round(_multiclass_brier(y, proba), 4),
            log_loss_prior=round(ll_prior, 4),
            skill_score=round(skill, 4),
            baseline_accuracy=round(base_acc, 4),
            neutral_baseline_accuracy=round(neutral_acc, 4),
            beats_baseline_pct=round(beats_pct, 3),
            p_value=round(p_value, 4) if not np.isnan(p_value) else float("nan"),
            accuracy_p_value=round(acc_p, 4) if not np.isnan(acc_p) else float("nan"),
            verdict=verdict,
            model_is_useful=useful,
            confusion=confusion_matrix(y, pred, labels=_LABELS).tolist(),
            precision_per_class=_per_class(precision_score, y, pred),
            recall_per_class=_per_class(recall_score, y, pred),
            fold_table=fold_table,
            calibration=_calibration_table(proba, y),
            signal_stats=self.signal_statistics(oof, self._threshold),
            unconditional_avg_return=float(oof["fwd_ret"].mean()),
            disclaimer=self._disclaimer(verdict, skill, acc, base_acc, beats_pct, p_value, n_eff),
        )

    def evaluate_volatility(self, oof: pd.DataFrame, horizon_bars: int = 1) -> VolatilityMetrics | None:
        """Wertet die OOF-Vorhersagen des Volatilitätsmodells aus."""
        if oof is None or oof.empty or "vol_true" not in oof:
            return None
        y = oof["vol_true"].to_numpy(dtype=int)
        pred = oof["vol_pred"].to_numpy(dtype=int)
        proba = oof[["vp_low", "vp_medium", "vp_high"]].to_numpy(dtype=float)
        prior_cols = ["vprior_low", "vprior_medium", "vprior_high"]
        if all(c in oof.columns for c in prior_cols):
            prior = oof[prior_cols].to_numpy(dtype=float)
        else:
            prior = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))
        idx = np.arange(len(y))
        ll_model_i = -np.log(np.clip(proba[idx, y], 1e-12, 1.0))
        ll_prior_i = -np.log(np.clip(prior[idx, y], 1e-12, 1.0))
        skill = 1.0 - ll_model_i.mean() / ll_prior_i.mean()
        p_value = _block_ttest_p(ll_prior_i - ll_model_i, max(1, horizon_bars))
        mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 else 0.0
        verdict, _ = self._verdict(p_value, skill, mcc)
        return VolatilityMetrics(
            accuracy=round(float(np.mean(y == pred)), 4),
            baseline_accuracy=round(float(np.mean(y == oof["vol_majority"].to_numpy(dtype=int))), 4),
            mcc=round(mcc, 4),
            balanced_accuracy=round(float(balanced_accuracy_score(y, pred)), 4),
            skill_score=round(float(skill), 4),
            p_value=round(p_value, 4) if not np.isnan(p_value) else float("nan"),
            verdict=verdict,
            confusion=confusion_matrix(y, pred, labels=_LABELS).tolist(),
        )

    @staticmethod
    def signal_statistics(oof: pd.DataFrame, threshold: float) -> dict[str, dict[str, float]]:
        """Trefferquote und Folge-Returns für Signale oberhalb der Konfidenzschwelle.

        Returns:
            {"BULLISH": {...}, "BEARISH": {...}, "NEUTRAL": {...}} mit n_signals,
            coverage (Anteil aller Zeitpunkte), hit_rate (Label korrekt),
            direction_hit_rate (Kurs bewegte sich in Signalrichtung),
            avg_return / median_return (Horizont-Return nach dem Signal).
        """
        proba = oof[list(_PROBA_COLS)].to_numpy(dtype=float)
        pred = proba.argmax(axis=1)
        confident = proba.max(axis=1) >= threshold
        y = oof["y_true"].to_numpy(dtype=int)
        fwd = oof["fwd_ret"].to_numpy(dtype=float)
        stats: dict[str, dict[str, float]] = {}
        for cls, name in DIRECTION_NAMES.items():
            mask = confident & (pred == cls)
            n = int(mask.sum())
            if n == 0:
                stats[name] = {"n_signals": 0, "coverage": 0.0}
                continue
            if cls == 2:
                direction_hits = float(np.mean(fwd[mask] > 0))
            elif cls == 0:
                direction_hits = float(np.mean(fwd[mask] < 0))
            else:
                direction_hits = float("nan")
            stats[name] = {
                "n_signals": n,
                "coverage": round(n / len(oof), 4),
                "hit_rate": round(float(np.mean(y[mask] == cls)), 4),
                "direction_hit_rate": round(direction_hits, 4) if not np.isnan(direction_hits) else float("nan"),
                "avg_return": float(np.nanmean(fwd[mask])),
                "median_return": float(np.nanmedian(fwd[mask])),
            }
        return stats

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------

    @staticmethod
    def _binomial_p(acc: float, baseline: float, n_eff: int) -> float:
        if n_eff < 10 or not 0.0 < baseline < 1.0:
            return float("nan")
        k = int(round(acc * n_eff))
        return float(binomtest(k, n_eff, baseline, alternative="greater").pvalue)

    def _verdict(self, p_value: float, skill: float, mcc: float) -> tuple[str, bool]:
        if not np.isnan(p_value) and p_value < self._alpha and skill > 0:
            return "signifikant", True
        if skill > 0 and mcc > 0:
            return "schwach", False
        return "keine Vorhersagekraft", False

    @staticmethod
    def _disclaimer(
        verdict: str, skill: float, acc: float, base: float, beats: float, p: float, n_eff: int
    ) -> str:
        p_txt = "n/a" if np.isnan(p) else f"{p:.3f}"
        core = (
            f"Log-Loss-Skill {skill:+.1%} gegenüber reinem Raten nach Klassenhäufigkeit "
            f"(p = {p_txt}, {n_eff} unabhängige Stichproben), besser in {beats:.0%} der "
            f"Testperioden. Trefferquote {acc:.0%} vs. {base:.0%} für die Mehrheitsklasse."
        )
        if verdict == "signifikant":
            return f"✅ Statistisch signifikanter Informationsgehalt. {core} Vergangenheit garantiert keine Zukunft."
        if verdict == "schwach":
            return f"ℹ️ Leichter, aber nicht signifikanter Vorsprung – kann Zufall sein. {core}"
        return (
            f"⚠️ Keine nachweisbare Vorhersagekraft für diesen Coin/Zeitraum. {core} "
            f"Signale bitte nur als Einordnung der Indikatorlage verstehen."
        )


def _prior_matrix(oof: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Prior-Wahrscheinlichkeiten je Zeile (aus dem Trainingsblock, sonst global)."""
    cols = ["prior_down", "prior_neutral", "prior_up"]
    if all(c in oof.columns for c in cols):
        return oof[cols].to_numpy(dtype=float)
    freq = np.bincount(y, minlength=3) / max(1, len(y))
    return np.tile(freq, (len(y), 1))


def _block_ttest_p(diff: np.ndarray, block: int) -> float:
    """Einseitiger t-Test (Mittelwert > 0) auf Mittelwerten nicht-überlappender Blöcke."""
    n_blocks = len(diff) // block
    if n_blocks < 10:
        return float("nan")
    means = diff[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    if np.isclose(means.std(), 0.0):
        # Keine Streuung: eindeutig besser (p≈0) bzw. nicht besser (p=1)
        return 0.0 if means.mean() > 0 else 1.0
    return float(ttest_1samp(means, 0.0, alternative="greater").pvalue)


def _per_class(metric: Any, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    values = metric(y, pred, labels=_LABELS, average=None, zero_division=0)
    return {DIRECTION_NAMES[i]: round(float(v), 3) for i, v in enumerate(values)}


def _multiclass_brier(y: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(len(_LABELS))[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _calibration_table(proba: np.ndarray, y: np.ndarray, n_bins: int = 6) -> list[dict[str, float]]:
    """Konfidenz (max. Wahrscheinlichkeit) vs. tatsächliche Trefferquote je Bin."""
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    edges = np.linspace(1 / 3, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.sum() >= 5:
            rows.append({
                "bin": f"{lo:.0%}–{hi:.0%}",
                "predicted": round(float(conf[mask].mean()), 4),
                "observed": round(float(correct[mask].mean()), 4),
                "count": int(mask.sum()),
            })
    return rows
