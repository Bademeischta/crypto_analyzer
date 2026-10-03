"""Modell-Engine: LightGBM (falls installiert) oder scikit-learn HistGradientBoosting.

Beide sind Histogram-basierte Gradient-Boosting-Verfahren mit nativer
NaN-Unterstützung (optionale Features wie Korrelationen dürfen fehlen).
LightGBM ist ~5–10× schneller, HistGradientBoosting braucht kein Zusatzpaket.

``predict_proba_aligned`` garantiert immer 3 Wahrscheinlichkeits-Spalten in
fester Klassenreihenfolge – auch wenn in einem Trainings-Fold eine Klasse
fehlte (sonst würden Spalten stillschweigend verrutschen).
"""

from __future__ import annotations

import importlib.util
import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

CLASSES: tuple[int, ...] = (0, 1, 2)

# LightGBM ≥ 4.6 vergibt auch bei NumPy-Eingaben Feature-Namen ("Column_0", …) und warnt dann
# bei jeder Vorhersage. Harmlos – einmalig und gezielt stummschalten (catch_warnings() pro Aufruf
# wäre nicht thread-sicher, die Walk-Forward-Folds laufen aber parallel).
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)


def lightgbm_available() -> bool:
    """True wenn das Paket ``lightgbm`` importierbar ist."""
    return importlib.util.find_spec("lightgbm") is not None


def resolve_engine(name: str) -> str:
    """'auto' → 'lightgbm' wenn installiert, sonst 'hist_gb'."""
    if name == "auto":
        return "lightgbm" if lightgbm_available() else "hist_gb"
    if name == "lightgbm" and not lightgbm_available():
        logger.warning("LightGBM nicht installiert – verwende HistGradientBoosting.")
        return "hist_gb"
    return name


class ConstantProbaModel:
    """Fallback-Modell, wenn ein Trainingsblock nur eine einzige Klasse enthält."""

    def __init__(self, y: np.ndarray) -> None:
        counts = np.array([(y == c).sum() for c in CLASSES], dtype=float)
        self.classes_ = np.array(CLASSES)
        self._proba = counts / counts.sum() if counts.sum() else np.full(len(CLASSES), 1 / len(CLASSES))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self._proba, (len(X), 1))


def build_classifier(model_cfg: dict[str, Any], engine: str) -> Any:
    """Erstellt einen (unfitted) Klassifikator mit den Hyperparametern aus der Config."""
    class_weight = model_cfg.get("class_weight") or None
    if engine == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=model_cfg["n_estimators"],
            learning_rate=model_cfg["learning_rate"],
            max_depth=model_cfg["max_depth"],
            num_leaves=model_cfg["num_leaves"],
            min_child_samples=model_cfg["min_samples_leaf"],
            reg_lambda=model_cfg["l2_regularization"],
            subsample=model_cfg["subsample"],
            subsample_freq=1,
            colsample_bytree=model_cfg["colsample"],
            class_weight=class_weight,
            random_state=model_cfg["random_state"],
            n_jobs=model_cfg.get("n_jobs", 1),
            verbose=-1,
        )

    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=model_cfg["n_estimators"],
        learning_rate=model_cfg["learning_rate"],
        max_depth=model_cfg["max_depth"],
        max_leaf_nodes=model_cfg["num_leaves"],
        min_samples_leaf=model_cfg["min_samples_leaf"],
        l2_regularization=model_cfg["l2_regularization"],
        class_weight=class_weight,
        early_stopping=False,
        random_state=model_cfg["random_state"],
    )


def fit_classifier(model_cfg: dict[str, Any], engine: str, X: np.ndarray, y: np.ndarray) -> Any:
    """Trainiert einen Klassifikator (oder ein Konstant-Modell bei nur einer Klasse)."""
    if len(np.unique(y)) < 2:
        return ConstantProbaModel(y)
    model = build_classifier(model_cfg, engine)
    model.fit(X, y)
    return model


def predict_proba_aligned(model: Any, X: np.ndarray) -> np.ndarray:
    """Wahrscheinlichkeiten mit genau einer Spalte pro Klasse in ``CLASSES``-Reihenfolge."""
    raw = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = [int(c) for c in model.classes_]
    if model_classes == list(CLASSES):
        return raw
    aligned = np.zeros((raw.shape[0], len(CLASSES)))
    for j, cls in enumerate(model_classes):
        aligned[:, CLASSES.index(cls)] = raw[:, j]
    return aligned


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature Scaling: T > 1 macht Wahrscheinlichkeiten vorsichtiger, T < 1 schärfer."""
    if temperature == 1.0:
        return proba
    logits = np.log(np.clip(proba, 1e-9, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class Calibration:
    """Zwei-Parameter-Kalibrierung: Temperature Scaling + Mischung mit der Prior-Verteilung.

    p_kal = w · softmax(log p / T) + (1 − w) · prior

    * T korrigiert Über-/Unterkonfidenz.
    * w (0…1) misst, wie viel echte Information das Modell liefert: Hat es
      keine, geht w → 0 und die Vorhersage fällt auf die Klassenhäufigkeiten
      zurück – ein uninformatives Modell kann so nie *schlechter* als Raten sein
      und erzeugt keine falschen, überzeugt wirkenden Signale.
    """

    temperature: float = 1.0
    weight: float = 1.0

    def apply(self, proba: np.ndarray, prior: np.ndarray) -> np.ndarray:
        scaled = apply_temperature(proba, self.temperature)
        return self.weight * scaled + (1.0 - self.weight) * np.asarray(prior, dtype=float)


def fit_calibration(proba: np.ndarray, y: np.ndarray, prior: np.ndarray) -> Calibration:
    """Findet (T, w), die den Log-Loss auf Out-of-Sample-Vorhersagen minimieren.

    Args:
        proba: Rohe Modellwahrscheinlichkeiten (n × 3).
        y: Wahre Klassen.
        prior: Prior je Zeile (n × 3) oder global (3,).
    """
    if len(y) < 30:
        return Calibration()
    idx = np.arange(len(y))
    prior = np.broadcast_to(np.asarray(prior, dtype=float), proba.shape)

    def loss(params: np.ndarray) -> float:
        cal = Calibration(float(params[0]), float(params[1]))
        p = cal.apply(proba, prior)[idx, y]
        return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))

    best = min(
        (minimize(loss, x0=np.array(x0), bounds=[(0.5, 10.0), (0.0, 1.0)], method="L-BFGS-B")
         for x0 in ((1.0, 1.0), (3.0, 0.5))),
        key=lambda r: r.fun,
    )
    return Calibration(float(best.x[0]), float(best.x[1]))
