"""Walk-Forward-Training mit LightGBM.

KRITISCH: Der Walk-Forward-Ansatz ist die einzige korrekte Methode für
Zeitreihen-Backtesting. Ein einfacher Train/Test-Split oder k-fold-CV
würde Zukunftsdaten ins Training einbeziehen (Lookahead-Bias).

Schema:
  Fold 1: [0..179]Train  [180..209]Test
  Fold 2: [30..209]Train [210..239]Test
  Fold 3: [60..239]Train [240..269]Test
  ...
  → Jeder Fold testet auf Daten, die beim Training streng in der Zukunft lagen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.features.pipeline import FeatureMatrix

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Ergebnis eines vollständigen Walk-Forward-Trainings.

    Attributes:
        direction_model: Finales LightGBM-Modell für Richtungsklassifikation.
        volatility_model: Finales LightGBM-Modell für Volatilitätsregime.
        feature_names: Verwendete Feature-Namen.
        fold_results: Metriken pro Fold.
        feature_importance: Dict feature_name -> avg. gain importance.
        data_end_date: Datum des letzten Trainings-Datenpunkts.
        n_folds: Anzahl durchgeführter Folds.
    """

    direction_model: lgb.LGBMClassifier
    volatility_model: lgb.LGBMClassifier
    feature_names: list[str]
    fold_results: list[dict[str, Any]]
    feature_importance: dict[str, float]
    data_end_date: str
    n_folds: int


class ModelTrainer:
    """Trainiert LightGBM-Modelle mit Walk-Forward-Cross-Validation.

    Args:
        config: Geladenes config.yaml als Dict.
        models_dir: Verzeichnis zum Speichern trainierter Modelle.
    """

    def __init__(self, config: dict[str, Any], models_dir: Path) -> None:
        self._ml_cfg = config["ml"]
        self._lgb_cfg = config["ml"]["lightgbm"]
        self._wf_cfg = config["ml"]["walk_forward"]
        self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)

    def train(self, fm: FeatureMatrix, symbol: str) -> TrainingResult:
        """Führt Walk-Forward-Training durch und speichert finale Modelle.

        Args:
            fm: Feature-Matrix aus der Pipeline.
            symbol: Coin-Symbol (für Dateinamen).

        Returns:
            TrainingResult mit Modellen und Evaluierungs-Metriken.
        """
        X = fm.X
        y_dir = fm.y_direction
        y_vola = fm.y_volatility

        logger.info(
            f"[{symbol}] Starte Walk-Forward-Training "
            f"mit {len(X)} Samples, {len(fm.feature_names)} Features."
        )

        # Walk-Forward-Folds berechnen
        folds = self._compute_folds(len(X))
        if len(folds) < self._wf_cfg["min_folds"]:
            raise ValueError(
                f"Zu wenig Daten für Walk-Forward-Training. "
                f"Gefunden: {len(folds)} Folds, "
                f"benötigt: {self._wf_cfg['min_folds']}. "
                f"Lade mehr historische Daten (mindestens "
                f"{self._wf_cfg['train_window_days'] + self._wf_cfg['test_window_days'] * self._wf_cfg['min_folds']} Tage)."
            )

        fold_results: list[dict[str, Any]] = []
        all_direction_importances: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train = X.iloc[train_idx]
            y_train_dir = y_dir.iloc[train_idx]
            y_train_vola = y_vola.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test_dir = y_dir.iloc[test_idx]

            # Validation-Split vom Ende des Trainings-Blocks (letzte 10%)
            val_size = max(int(len(X_train) * 0.1), 1)
            X_tr = X_train.iloc[:-val_size]
            X_val = X_train.iloc[-val_size:]
            y_tr_dir = y_train_dir.iloc[:-val_size]
            y_val_dir = y_train_dir.iloc[-val_size:]

            dir_model = self._train_single_model(X_tr, y_tr_dir, X_val, y_val_dir)
            y_pred_dir = dir_model.predict(X_test)
            y_proba_dir = dir_model.predict_proba(X_test)

            # Datumsindex für Backtest-Engine sichern
            try:
                test_dates = X_test.index.strftime("%Y-%m-%d").tolist()
            except Exception:
                test_dates = [str(d) for d in X_test.index.tolist()]

            fold_metric = {
                "fold": fold_idx + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "y_true": y_test_dir.tolist(),
                "y_pred": y_pred_dir.tolist(),
                "y_proba": y_proba_dir.tolist(),
                "test_dates": test_dates,
            }
            fold_results.append(fold_metric)
            all_direction_importances.append(dir_model.feature_importances_)
            logger.info(f"  Fold {fold_idx + 1}/{len(folds)} abgeschlossen.")

        # Finales Modell auf ALLEN Daten trainieren (kein Validation-Split für finales Training)
        logger.info(f"[{symbol}] Trainiere finale Modelle auf vollständigem Datensatz...")
        final_dir_model = self._train_single_model(X, y_dir, None, None)
        final_vola_model = self._train_single_model(X, y_vola, None, None)

        # Feature Importance: Durchschnitt über alle Folds (stabilere Schätzung)
        avg_importances = np.mean(all_direction_importances, axis=0)
        feature_importance = {
            name: float(imp)
            for name, imp in zip(fm.feature_names, avg_importances)
        }

        # Modelle auf Disk speichern
        self._save_models(symbol, final_dir_model, final_vola_model, fm.feature_names)

        result = TrainingResult(
            direction_model=final_dir_model,
            volatility_model=final_vola_model,
            feature_names=fm.feature_names,
            fold_results=fold_results,
            feature_importance=feature_importance,
            data_end_date=fm.data_end_date,
            n_folds=len(folds),
        )
        logger.info(f"[{symbol}] Training abgeschlossen. {len(folds)} Folds.")
        return result

    def _compute_folds(self, n_samples: int) -> list[tuple[list[int], list[int]]]:
        """Berechnet Walk-Forward-Folds ohne Datenmischung.

        Folds werden chronologisch erstellt: Training immer auf Vergangenheit,
        Test immer auf anschließende Zukunft.

        Args:
            n_samples: Gesamtanzahl der Samples.

        Returns:
            Liste von (train_indices, test_indices) Tupeln.
        """
        train_w = self._wf_cfg["train_window_days"]
        test_w = self._wf_cfg["test_window_days"]
        max_folds = self._wf_cfg["max_folds"]

        folds: list[tuple[list[int], list[int]]] = []
        start = 0

        while start + train_w + test_w <= n_samples:
            train_end = start + train_w
            test_end = train_end + test_w

            train_idx = list(range(start, train_end))
            test_idx = list(range(train_end, min(test_end, n_samples)))

            if len(test_idx) > 0:
                folds.append((train_idx, test_idx))

            if len(folds) >= max_folds:
                break

            # Rollierendes Fenster: schiebe um test_w vorwärts
            start += test_w

        return folds

    def _train_single_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> lgb.LGBMClassifier:
        """Trainiert ein einzelnes LightGBM-Klassifikationsmodell.

        Args:
            X_train: Trainings-Features.
            y_train: Trainings-Labels.
            X_val: Validierungs-Features (für Early Stopping). Kann None sein.
            y_val: Validierungs-Labels.

        Returns:
            Trainiertes LGBMClassifier-Objekt.
        """
        model = lgb.LGBMClassifier(
            n_estimators=self._lgb_cfg["n_estimators"],
            learning_rate=self._lgb_cfg["learning_rate"],
            max_depth=self._lgb_cfg["max_depth"],
            num_leaves=self._lgb_cfg["num_leaves"],
            feature_fraction=self._lgb_cfg["feature_fraction"],
            subsample=self._lgb_cfg["bagging_fraction"],
            subsample_freq=self._lgb_cfg["bagging_freq"],
            min_child_samples=self._lgb_cfg["min_data_in_leaf"],
            random_state=self._lgb_cfg["random_state"],
            verbose=self._lgb_cfg["verbose"],
            n_jobs=-1,
        )

        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None and len(X_val) > 0:
            fit_params = {
                "eval_set": [(X_val, y_val)],
                "callbacks": [
                    lgb.early_stopping(
                        stopping_rounds=self._lgb_cfg["early_stopping_rounds"],
                        verbose=False,
                    )
                ],
            }

        model.fit(X_train, y_train, **fit_params)
        return model

    def _save_models(
        self,
        symbol: str,
        dir_model: lgb.LGBMClassifier,
        vola_model: lgb.LGBMClassifier,
        feature_names: list[str],
    ) -> None:
        """Speichert Modelle und Metadaten auf Disk.

        Args:
            symbol: Coin-Symbol für Dateinamen.
            dir_model: Richtungs-Modell.
            vola_model: Volatilitäts-Modell.
            feature_names: Feature-Namen für Konsistenz-Check beim Laden.
        """
        sym = symbol.upper()
        joblib.dump(dir_model, self._models_dir / f"{sym}_direction_model.joblib")
        joblib.dump(vola_model, self._models_dir / f"{sym}_volatility_model.joblib")
        joblib.dump(feature_names, self._models_dir / f"{sym}_feature_names.joblib")
        logger.info(f"Modelle gespeichert: {self._models_dir}/{sym}_*.joblib")

    def load_models(
        self, symbol: str
    ) -> tuple[lgb.LGBMClassifier, lgb.LGBMClassifier, list[str]] | None:
        """Lädt gespeicherte Modelle von Disk.

        Args:
            symbol: Coin-Symbol.

        Returns:
            Tupel (direction_model, volatility_model, feature_names) oder None wenn
            keine Modelle existieren.
        """
        sym = symbol.upper()
        dir_path = self._models_dir / f"{sym}_direction_model.joblib"
        vola_path = self._models_dir / f"{sym}_volatility_model.joblib"
        feat_path = self._models_dir / f"{sym}_feature_names.joblib"

        if not all(p.exists() for p in (dir_path, vola_path, feat_path)):
            return None

        try:
            dir_model = joblib.load(dir_path)
            vola_model = joblib.load(vola_path)
            feature_names = joblib.load(feat_path)
            return dir_model, vola_model, feature_names
        except Exception as exc:
            logger.warning(f"Fehler beim Laden der Modelle für {symbol}: {exc}")
            return None
