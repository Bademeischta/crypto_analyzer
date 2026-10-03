"""Purged Walk-Forward-Training (Richtung + Volatilitätsregime).

Warum Walk-Forward? Ein zufälliger Train/Test-Split oder k-fold-CV würde
Zukunftsdaten ins Training einbeziehen (Lookahead-Bias) – bei Zeitreihen
führt das zu massiv geschönten Ergebnissen.

Warum *purged*? Das Label der Zeile t nutzt den Kurs bei t+h. Die letzten h
Trainingszeilen vor einem Testblock "kennen" also bereits Kurse aus dem
Testzeitraum. Diese Zeilen werden entfernt (Embargo = h Kerzen).

Schema (expanding, verankert am Datenende – die jüngsten Daten werden immer getestet):

    Fold 1: [Train ··········]  (Embargo)  [Test]
    Fold 2: [Train ················]  (Embargo)  [Test]
    Fold 3: [Train ······················]  (Embargo)  [Test] ← endet mit den neuesten Daten

Die Out-of-Fold-Vorhersagen (OOF) aller Testblöcke bilden eine lückenlose,
echte Out-of-Sample-Historie – Grundlage für Evaluierung und Backtest.

Kalibrierung: Die Wahrscheinlichkeiten jedes Folds werden mit einer
Kalibrierung korrigiert, die nur auf den OOF-Vorhersagen *früherer* Folds
gelernt wurde. Der erste Fold dient daher ausschließlich als Warm-up und
fließt nicht in Evaluierung und Backtest ein.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.config import Timeframe, per_interval
from src.features.pipeline import FeatureMatrix
from src.models.engine import (
    CLASSES,
    Calibration,
    fit_calibration,
    fit_classifier,
    predict_proba_aligned,
    resolve_engine,
)

logger = logging.getLogger(__name__)

_BUNDLE_VERSION = 4


@dataclass(frozen=True)
class Fold:
    """Indexgrenzen eines Walk-Forward-Folds (halboffene Intervalle [start, end))."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_idx(self) -> np.ndarray:
        return np.arange(self.train_start, self.train_end)

    @property
    def test_idx(self) -> np.ndarray:
        return np.arange(self.test_start, self.test_end)


@dataclass
class TrainingResult:
    """Ergebnis eines Walk-Forward-Trainings inkl. finaler Modelle.

    Attributes:
        direction_model: Finales Richtungsmodell (auf allen Daten trainiert).
        volatility_model: Finales Volatilitätsmodell.
        feature_names: Feature-Reihenfolge der Modelle.
        oof: Out-of-Fold-Vorhersagen (Index = Zeitstempel).
        folds: Metadaten je Fold.
        feature_importance: Permutation-Importance (Anstieg des Log-Loss) je Feature.
        engine: "lightgbm" oder "hist_gb".
        trained_at: ISO-Zeitstempel des Trainings.
        data_end: Letzter Trainings-Zeitstempel.
        n_samples: Anzahl Trainingszeilen.
        horizon_bars: Label-Horizont in Kerzen.
        fingerprint: Config-Fingerprint (Invalidierung bei Config-Änderung).
        calibration / volatility_calibration: Kalibrierung (Temperatur, Gewicht) je Modell.
        class_prior / volatility_prior: Klassenhäufigkeiten im gesamten Trainingsdatensatz.
        training_seconds: Trainingsdauer.
        from_cache: True wenn von Disk geladen.
    """

    direction_model: Any
    volatility_model: Any
    feature_names: list[str]
    oof: pd.DataFrame
    folds: list[dict[str, Any]]
    feature_importance: dict[str, float]
    engine: str
    trained_at: str
    data_end: str
    n_samples: int
    horizon_bars: int
    fingerprint: str
    calibration: Calibration = field(default_factory=Calibration)
    volatility_calibration: Calibration = field(default_factory=Calibration)
    class_prior: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    volatility_prior: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    training_seconds: float = 0.0
    from_cache: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Sekunden seit dem Training."""
        trained = datetime.fromisoformat(self.trained_at)
        return (datetime.now(UTC) - trained).total_seconds()


def compute_walk_forward_folds(
    n_samples: int,
    min_train: int,
    test_size: int,
    embargo: int,
    max_folds: int,
    rolling_window: int | None = None,
) -> list[Fold]:
    """Berechnet purged Walk-Forward-Folds, verankert am Datenende.

    Args:
        n_samples: Anzahl Zeilen.
        min_train: Mindestgröße des Trainingsblocks.
        test_size: Größe jedes Testblocks.
        embargo: Lücke zwischen Train-Ende und Test-Start (≥ Label-Horizont).
        max_folds: Maximale Anzahl Folds.
        rolling_window: Feste Trainingsfenstergröße; None = expanding.

    Returns:
        Chronologisch sortierte Folds. Testblöcke überlappen nie und liegen
        lückenlos hintereinander; der letzte endet bei ``n_samples``.
    """
    folds: list[Fold] = []
    test_end = n_samples
    while len(folds) < max_folds:
        test_start = test_end - test_size
        train_end = test_start - embargo
        train_start = 0 if rolling_window is None else max(0, train_end - rolling_window)
        if test_start <= 0 or train_end - train_start < min_train:
            break
        folds.append(Fold(train_start, train_end, test_start, test_end))
        test_end = test_start
    return folds[::-1]


class ModelTrainer:
    """Trainiert Richtungs- und Volatilitätsmodelle mit purged Walk-Forward-CV.

    Args:
        config: Geladenes config.yaml als Dict.
        models_dir: Verzeichnis für gespeicherte Modelle.
    """

    def __init__(self, config: dict[str, Any], models_dir: Path) -> None:
        self._ml_cfg = config["ml"]
        self._wf_cfg = config["ml"]["walk_forward"]
        self._model_cfg = config["ml"]["model"]
        self._engine = resolve_engine(self._ml_cfg["engine"])
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def engine(self) -> str:
        """Aktive Modell-Engine."""
        return self._engine

    def plan_folds(self, n_samples: int, interval: str, horizon_bars: int, extra_folds: int = 0) -> list[Fold]:
        """Folds für eine Datenmenge gemäß Config (Tage → Kerzen des Intervalls).

        Args:
            extra_folds: Zusätzliche (älteste) Folds, z.B. als Kalibrierungs-Warm-up.
        """
        tf = Timeframe(interval)
        wf = self._wf_cfg
        rolling = None
        if wf["mode"] == "rolling":
            rolling = tf.bars(per_interval(wf["train_window_days"], interval))
        return compute_walk_forward_folds(
            n_samples=n_samples,
            min_train=tf.bars(per_interval(wf["min_train_days"], interval)),
            test_size=tf.bars(per_interval(wf["test_window_days"], interval)),
            embargo=horizon_bars,
            max_folds=wf["max_folds"] + extra_folds,
            rolling_window=rolling,
        )

    def train(self, fm: FeatureMatrix, symbol: str, interval: str, fingerprint: str) -> TrainingResult:
        """Führt Walk-Forward-Training durch, trainiert finale Modelle und speichert sie.

        Phase 1 (parallel): Modelle je Fold trainieren, rohe OOF-Wahrscheinlichkeiten
        und Permutation-Importance berechnen. Phase 2 (sequentiell, billig):
        zeitlich saubere Kalibrierung und Zusammenbau der OOF-Tabelle.

        Raises:
            ValueError: Wenn weniger als ``min_folds`` Folds möglich sind.
        """
        t0 = time.perf_counter()
        X = fm.X.to_numpy(dtype=float)
        y_dir = fm.y_direction.to_numpy()
        y_vol = fm.y_volatility.to_numpy()
        n = len(X)

        calibrate = self._ml_cfg.get("calibration", "blend") != "none"
        warmup = 1 if calibrate else 0
        folds = self.plan_folds(n, interval, fm.horizon_bars, extra_folds=warmup)
        if len(folds) - warmup < self._wf_cfg["min_folds"]:
            tf = Timeframe(interval)
            needed_days = (
                per_interval(self._wf_cfg["min_train_days"], interval)
                + (self._wf_cfg["min_folds"] + warmup) * per_interval(self._wf_cfg["test_window_days"], interval)
            )
            raise ValueError(
                f"Zu wenig Historie für belastbares Walk-Forward-Training: {n} nutzbare Kerzen "
                f"(≈ {n / tf.bars_per_day:.0f} Tage) ergeben nur {max(0, len(folds) - warmup)} Folds, "
                f"benötigt werden {self._wf_cfg['min_folds']} (≈ {needed_days:.0f} Tage nach "
                f"Indikator-Vorlauf). Tipp: kürzeres Intervall (4h/1h) wählen – das liefert mehr Kerzen."
            )

        logger.info(
            f"[{symbol} {interval}] Walk-Forward: {n} Samples, {len(fm.feature_names)} Features, "
            f"{len(folds)} Folds, Engine={self._engine}"
        )

        # ── Phase 1: Folds parallel trainieren ─────────────────────────────
        first_importance_fold = max(warmup, len(folds) - self._wf_cfg["importance_folds"])
        seed = self._model_cfg["random_state"]

        def run_fold(i: int) -> dict[str, Any]:
            fold = folds[i]
            tr, te = fold.train_idx, fold.test_idx
            dir_model = fit_classifier(self._model_cfg, self._engine, X[tr], y_dir[tr])
            vol_model = fit_classifier(self._model_cfg, self._engine, X[tr], y_vol[tr])
            out = {
                "p_dir": predict_proba_aligned(dir_model, X[te]),
                "p_vol": predict_proba_aligned(vol_model, X[te]),
                "prior": np.bincount(y_dir[tr], minlength=3) / len(tr),
                "vol_prior": np.bincount(y_vol[tr], minlength=3) / len(tr),
                "importance": None,
            }
            if i >= first_importance_fold:
                out["importance"] = permutation_importance(
                    dir_model, X[te], y_dir[te], self._wf_cfg["permutation_repeats"],
                    np.random.default_rng(seed + i),
                )
            return out

        workers = self._parallel_workers(len(folds))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fold_out = list(pool.map(run_fold, range(len(folds))))
        else:
            fold_out = [run_fold(i) for i in range(len(folds))]

        # ── Phase 2: Kalibrierung nur mit früheren Folds (kein Lookahead) ──
        fwd = fm.forward_return.to_numpy()
        nxt = fm.next_return.to_numpy()
        oof_parts: list[pd.DataFrame] = []
        fold_meta: list[dict[str, Any]] = []

        for i, (fold, out) in enumerate(zip(folds, fold_out)):
            te = fold.test_idx
            prior, vol_prior = out["prior"], out["vol_prior"]
            cal_dir = cal_vol = Calibration()
            if calibrate and i > 0:
                cal_dir = self._fit_calibration_on(fold_out[:i], folds[:i], y_dir, "p_dir", "prior")
                cal_vol = self._fit_calibration_on(fold_out[:i], folds[:i], y_vol, "p_vol", "vol_prior")
            if i < warmup:
                continue  # Warm-up-Fold: liefert nur Kalibrierungsdaten

            p_dir = cal_dir.apply(out["p_dir"], prior)
            p_vol = cal_vol.apply(out["p_vol"], vol_prior)
            fold_no = i + 1 - warmup
            oof_parts.append(pd.DataFrame({
                "fold": fold_no,
                "p_down": p_dir[:, 0], "p_neutral": p_dir[:, 1], "p_up": p_dir[:, 2],
                "y_true": y_dir[te], "pred": p_dir.argmax(axis=1),
                "majority": int(prior.argmax()),
                "prior_down": prior[0], "prior_neutral": prior[1], "prior_up": prior[2],
                "vp_low": p_vol[:, 0], "vp_medium": p_vol[:, 1], "vp_high": p_vol[:, 2],
                "vol_true": y_vol[te], "vol_pred": p_vol.argmax(axis=1),
                "vol_majority": int(vol_prior.argmax()),
                "vprior_low": vol_prior[0], "vprior_medium": vol_prior[1], "vprior_high": vol_prior[2],
                "cal_temperature": cal_dir.temperature,
                "cal_weight": cal_dir.weight,
                "fwd_ret": fwd[te],
                "next_ret": nxt[te],
            }, index=fm.X.index[te]))
            fold_meta.append({
                "fold": fold_no,
                "train_start": str(fm.X.index[fold.train_start]),
                "train_end": str(fm.X.index[fold.train_end - 1]),
                "test_start": str(fm.X.index[fold.test_start]),
                "test_end": str(fm.X.index[fold.test_end - 1]),
                "n_train": len(fold.train_idx),
                "n_test": len(te),
                "cal_temperature": round(cal_dir.temperature, 3),
                "cal_weight": round(cal_dir.weight, 3),
            })

        # Kalibrierung für Live-Vorhersagen: auf allen OOF-Vorhersagen
        final_cal_dir = final_cal_vol = Calibration()
        if calibrate:
            final_cal_dir = self._fit_calibration_on(fold_out, folds, y_dir, "p_dir", "prior")
            final_cal_vol = self._fit_calibration_on(fold_out, folds, y_vol, "p_vol", "vol_prior")

        # Finale Modelle auf allen gelabelten Daten (parallel)
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_dir = pool.submit(fit_classifier, self._model_cfg, self._engine, X, y_dir)
            f_vol = pool.submit(fit_classifier, self._model_cfg, self._engine, X, y_vol)
            final_dir, final_vol = f_dir.result(), f_vol.result()

        importances = [o["importance"] for o in fold_out if o["importance"] is not None]
        mean_imp = np.mean(importances, axis=0) if importances else np.zeros(len(fm.feature_names))
        result = TrainingResult(
            direction_model=final_dir,
            volatility_model=final_vol,
            feature_names=list(fm.feature_names),
            oof=pd.concat(oof_parts),
            folds=fold_meta,
            feature_importance={f: float(v) for f, v in zip(fm.feature_names, mean_imp)},
            engine=self._engine,
            trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
            data_end=str(fm.X.index[-1]),
            n_samples=n,
            horizon_bars=fm.horizon_bars,
            fingerprint=fingerprint,
            calibration=final_cal_dir,
            volatility_calibration=final_cal_vol,
            class_prior=tuple(float(v) for v in np.bincount(y_dir, minlength=3) / n),
            volatility_prior=tuple(float(v) for v in np.bincount(y_vol, minlength=3) / n),
            training_seconds=round(time.perf_counter() - t0, 2),
        )
        self.save(symbol, interval, result)
        logger.info(f"[{symbol} {interval}] Training fertig in {result.training_seconds:.1f}s.")
        return result

    def _parallel_workers(self, n_folds: int) -> int:
        setting = self._ml_cfg.get("parallel_folds", "auto")
        if setting == "auto":
            setting = min(8, max(1, (os.cpu_count() or 2) - 1))
        return max(1, min(int(setting), n_folds))

    @staticmethod
    def _fit_calibration_on(
        outputs: list[dict[str, Any]], folds: list[Fold], y: np.ndarray, proba_key: str, prior_key: str
    ) -> Calibration:
        """Kalibrierung auf den rohen OOF-Vorhersagen der übergebenen Folds."""
        proba = np.vstack([o[proba_key] for o in outputs])
        labels = np.concatenate([y[f.test_idx] for f in folds])
        priors = np.vstack([np.tile(o[prior_key], (len(f.test_idx), 1)) for o, f in zip(outputs, folds)])
        return fit_calibration(proba, labels, priors)

    # ------------------------------------------------------------------
    # Persistenz
    # ------------------------------------------------------------------

    def _bundle_path(self, symbol: str, interval: str, fingerprint: str) -> Path:
        return self._models_dir / f"{symbol.upper()}_{interval}_{fingerprint}.joblib"

    def save(self, symbol: str, interval: str, result: TrainingResult) -> None:
        """Speichert Modelle + Evaluierungsdaten als ein Bundle; alte Bundles werden entfernt."""
        path = self._bundle_path(symbol, interval, result.fingerprint)
        payload = {"version": _BUNDLE_VERSION, **{k: v for k, v in result.__dict__.items() if k != "from_cache"}}
        tmp = path.with_suffix(".tmp")
        try:
            joblib.dump(payload, tmp, compress=3)
            tmp.replace(path)
        except OSError as exc:
            logger.warning(f"Modell konnte nicht gespeichert werden: {exc}")
            return
        for old in self._models_dir.glob(f"{symbol.upper()}_{interval}_*.joblib"):
            if old != path:
                old.unlink(missing_ok=True)

    def load(self, symbol: str, interval: str, fingerprint: str) -> TrainingResult | None:
        """Lädt ein gespeichertes Bundle (None wenn nicht vorhanden/inkompatibel)."""
        path = self._bundle_path(symbol, interval, fingerprint)
        if not path.exists():
            return None
        try:
            payload = joblib.load(path)
        except Exception as exc:  # Versions-Inkompatibilität, korrupte Datei, ...
            logger.warning(f"Modell-Bundle {path.name} unlesbar ({exc}) – trainiere neu.")
            return None
        if payload.pop("version", None) != _BUNDLE_VERSION:
            return None
        try:
            return TrainingResult(**payload, from_cache=True)
        except TypeError:
            return None


def permutation_importance(
    model: Any, X: np.ndarray, y: np.ndarray, n_repeats: int, rng: np.random.Generator
) -> np.ndarray:
    """Permutation-Importance auf Out-of-Sample-Daten (Anstieg des Log-Loss).

    Positiv = Feature hilft dem Modell auf ungesehenen Daten; ≈0 oder negativ =
    Feature ist nutzlos bzw. schadet. Anders als die Split-/Gain-Importance
    eines Baummodells ist das nicht zugunsten hochkardinaler Features verzerrt.
    """
    labels = list(CLASSES)
    base = log_loss(y, predict_proba_aligned(model, X), labels=labels)
    scores = np.zeros(X.shape[1])
    X_perm = X.copy()
    for j in range(X.shape[1]):
        original = X_perm[:, j].copy()
        for _ in range(n_repeats):
            X_perm[:, j] = rng.permutation(original)
            scores[j] += log_loss(y, predict_proba_aligned(model, X_perm), labels=labels) - base
        X_perm[:, j] = original
    return scores / max(n_repeats, 1)
