"""Orchestrierungs-Schicht: Koordiniert alle Komponenten zu einer Analyse.

Der Analyzer ist der einzige Einstiegspunkt für das Dashboard.
Er entscheidet ob bestehende Modelle noch frisch genug sind oder
ob ein Retrain notwendig ist.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.features.pipeline import FeaturePipeline, FeatureMatrix
from src.features.sentiment import SentimentFetcher
from src.models.evaluator import AggregatedMetrics, ModelEvaluator
from src.models.predictor import PredictionResult, Predictor
from src.models.trainer import ModelTrainer, TrainingResult

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Vollständiges Ergebnis einer Coin-Analyse.

    Attributes:
        symbol: Analysierter Coin (z.B. "BTC").
        ohlcv: OHLCV-DataFrame (nach Validierung).
        market_data: CoinGecko-Marktdaten.
        prediction: KI-Vorhersage (Richtung + Volatilität).
        eval_metrics: Historische Modell-Performance.
        feature_importance: Top-Features nach Wichtigkeit.
        sentiment: Fear & Greed + Reddit-Daten.
        warnings: Nicht-kritische Warnungen (aus Validator).
        error: Kritischer Fehler-Text oder None.
        training_time_seconds: Dauer des Trainings (0 wenn gecacht).
        data_freshness_minutes: Alter der Daten in Minuten.
    """

    symbol: str
    ohlcv: pd.DataFrame
    market_data: dict[str, Any]
    prediction: PredictionResult | None
    eval_metrics: AggregatedMetrics | None
    feature_importance: dict[str, float]
    sentiment: dict[str, Any]
    warnings: list[str]
    error: str | None
    training_time_seconds: float
    data_freshness_minutes: float


class CryptoAnalyzer:
    """Führt eine vollständige KI-gestützte Krypto-Analyse durch.

    Args:
        config_path: Pfad zur config.yaml.
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        with config_path.open(encoding="utf-8") as fh:
            self._config = yaml.safe_load(fh)

        self._fetcher = DataFetcher(config_path)
        self._validator = DataValidator(
            max_consecutive_gaps=self._config["validation"]["max_consecutive_fillable_gaps"],
            volume_iqr_multiplier=self._config["validation"]["volume_iqr_multiplier"],
            return_zscore_threshold=self._config["validation"]["return_zscore_threshold"],
            minimum_samples=self._config["validation"]["minimum_samples_for_training"],
        )
        self._pipeline = FeaturePipeline(self._config)
        self._trainer = ModelTrainer(
            self._config,
            config_path.parent / self._config["paths"]["models_dir"],
        )
        self._predictor = Predictor(self._config)
        self._evaluator = ModelEvaluator(self._config)
        self._sentiment_fetcher = SentimentFetcher(self._config, self._fetcher.cache)

    def analyze(
        self,
        symbol: str,
        interval: str = "1d",
        lookback_days: int = 365,
        force_retrain: bool = False,
    ) -> AnalysisResult:
        """Führt die vollständige Analyse eines Coins durch.

        Ablauf:
          1. OHLCV-Daten laden (mit Cache)
          2. Daten validieren
          3. Referenzdaten (BTC/ETH) für Korrelations-Features laden
          4. Sentiment laden
          5. Feature-Matrix aufbauen
          6. Modell laden oder trainieren (mit Auto-Retrain nach 24h)
          7. Vorhersage erstellen
          8. Ergebnisse zusammenführen

        Args:
            symbol: Coin-Symbol (z.B. "BTC", "DOGE").
            interval: Kerzen-Intervall ("1d", "4h", "1h").
            lookback_days: Historische Daten in Tagen.
            force_retrain: True um Retrain unabhängig vom Modell-Alter zu erzwingen.

        Returns:
            AnalysisResult mit allen Daten für das Dashboard.
        """
        warnings: list[str] = []

        # ── Schritt 1: Primäre OHLCV-Daten laden ──────────────────────────
        try:
            raw_df = self._fetcher.get_ohlcv(symbol, interval, lookback_days)
        except ValueError as exc:
            return self._error_result(symbol, str(exc))
        except Exception as exc:
            return self._error_result(
                symbol,
                f"Daten konnten nicht geladen werden: {exc}. "
                f"Prüfe deine Internetverbindung.",
            )

        # Datenfrisc(h)e berechnen
        data_age_minutes = self._get_data_age_minutes(symbol, interval, lookback_days)

        # ── Schritt 2: Daten validieren ────────────────────────────────────
        validation = self._validator.validate(raw_df, symbol)
        warnings.extend(validation.warnings)
        if not validation.is_valid:
            return self._error_result(symbol, "; ".join(validation.errors))
        df = validation.df

        # ── Schritt 3: Marktdaten von CoinGecko ───────────────────────────
        market_data = self._fetcher.get_market_data(symbol)

        # ── Schritt 4: Referenzdaten für Korrelationen ────────────────────
        reference_data = self._load_reference_data(symbol, interval, lookback_days)

        # ── Schritt 5: Sentiment-Daten ────────────────────────────────────
        sentiment = self._load_sentiment(symbol)

        # Sentiment-Features die in die Feature-Matrix fließen
        sentiment_features: dict[str, float] = {}
        fg = sentiment.get("fear_greed", {})
        if fg.get("current_value") is not None:
            sentiment_features["fear_greed"] = float(fg["current_value"])
        if fg.get("change_3d") is not None:
            sentiment_features["fear_greed_change_3d"] = float(fg["change_3d"])

        # ── Schritt 6: Feature-Matrix aufbauen ────────────────────────────
        try:
            feature_matrix = self._pipeline.build(df, reference_data, sentiment_features)
        except Exception as exc:
            logger.error(f"Feature-Pipeline fehlgeschlagen: {exc}")
            return self._error_result(
                symbol,
                f"Feature-Berechnung fehlgeschlagen: {exc}",
            )

        # ── Schritt 7: Modell laden oder trainieren ────────────────────────
        train_start = time.time()
        training_result = self._get_or_train_model(
            symbol, feature_matrix, force_retrain
        )
        train_duration = time.time() - train_start

        if training_result is None:
            return self._error_result(
                symbol,
                "Modell-Training fehlgeschlagen. Zu wenig Daten oder Feature-Fehler.",
            )

        # ── Schritt 8: Vorhersage ──────────────────────────────────────────
        prediction = self._predictor.predict(
            direction_model=training_result.direction_model,
            volatility_model=training_result.volatility_model,
            feature_row=feature_matrix.last_row,
            feature_names=feature_matrix.feature_names,
            data_end_date=feature_matrix.data_end_date,
        )

        # ── Schritt 9: Evaluierung ─────────────────────────────────────────
        eval_metrics = self._evaluator.evaluate_folds(training_result.fold_results)

        # ── Schritt 10: Feature Importance filtern (nur Top-Features) ──────
        importance = training_result.feature_importance
        total_imp = sum(importance.values()) or 1.0
        top_importance = {
            k: round(v / total_imp * 100, 2)
            for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
            if v / total_imp * 100 >= self._config["ml"]["min_feature_importance_pct"]
        }

        return AnalysisResult(
            symbol=symbol.upper(),
            ohlcv=df,
            market_data=market_data,
            prediction=prediction,
            eval_metrics=eval_metrics,
            feature_importance=top_importance,
            sentiment=sentiment,
            warnings=warnings,
            error=None,
            training_time_seconds=round(train_duration, 2),
            data_freshness_minutes=round(data_age_minutes, 1),
        )

    def get_trending_coins(self) -> list[dict[str, Any]]:
        """Delegiert an CoinGecko Trending.

        Returns:
            Liste trendender Coins.
        """
        return self._fetcher.get_trending_coins()

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _get_or_train_model(
        self,
        symbol: str,
        feature_matrix: FeatureMatrix,
        force_retrain: bool,
    ) -> TrainingResult | None:
        """Lädt gecachtes Modell oder trainiert neu wenn nötig.

        Auto-Retrain-Logik: Modell wird neu trainiert wenn:
        - Noch kein Modell vorhanden
        - Letzte Training-Datei älter als 24h (model_ttl_seconds)
        - force_retrain=True

        Args:
            symbol: Coin-Symbol.
            feature_matrix: Feature-Matrix für Training.
            force_retrain: Immer neu trainieren.

        Returns:
            TrainingResult oder None bei Fehler.
        """
        model_ttl = self._config["cache"]["model_ttl_seconds"]
        sym = symbol.upper()

        # Modell-Alter prüfen
        model_age = self._get_model_age_seconds(sym)
        needs_retrain = (
            force_retrain
            or model_age is None
            or model_age > model_ttl
        )

        if not needs_retrain:
            # Gecachte Modelle laden
            loaded = self._trainer.load_models(sym)
            if loaded is not None:
                dir_model, vola_model, feature_names = loaded
                # Wenn Feature-Namen nicht übereinstimmen → Retrain erzwingen
                if set(feature_names) == set(feature_matrix.feature_names):
                    logger.info(f"[{sym}] Gecachte Modelle geladen (Alter: {model_age:.0f}s).")
                    # Für gecachte Modelle: kein Walk-Forward, leere fold_results
                    # (Evaluierung kommt aus letztem Training)
                    return TrainingResult(
                        direction_model=dir_model,
                        volatility_model=vola_model,
                        feature_names=feature_names,
                        fold_results=[],
                        feature_importance={f: 0.0 for f in feature_names},
                        data_end_date=feature_matrix.data_end_date,
                        n_folds=0,
                    )
                logger.info(f"[{sym}] Feature-Namen geändert → Retrain nötig.")

        # Training durchführen
        try:
            result = self._trainer.train(feature_matrix, sym)
            return result
        except ValueError as exc:
            logger.error(f"Training fehlgeschlagen: {exc}")
            return None

    def _get_model_age_seconds(self, symbol: str) -> float | None:
        """Gibt das Alter des gespeicherten Modells in Sekunden zurück.

        Args:
            symbol: Coin-Symbol (uppercase).

        Returns:
            Alter in Sekunden oder None wenn kein Modell vorhanden.
        """
        import time as _time
        model_path = (
            self._config_path.parent
            / self._config["paths"]["models_dir"]
            / f"{symbol}_direction_model.joblib"
        )
        if not model_path.exists():
            return None
        return _time.time() - model_path.stat().st_mtime

    def _load_reference_data(
        self,
        symbol: str,
        interval: str,
        lookback_days: int,
    ) -> dict[str, pd.DataFrame]:
        """Lädt Referenz-OHLCV-Daten für Korrelations-Features.

        Überspringt das Symbol selbst (wenn z.B. BTC analysiert wird,
        brauchen wir keine BTC-Korrelation zu BTC).

        Args:
            symbol: Haupt-Symbol (wird ausgeschlossen).
            interval: Kerzen-Intervall.
            lookback_days: Datenmenge.

        Returns:
            Dict symbol -> DataFrame.
        """
        ref_symbols: list[str] = self._config["features"]["reference_symbols"]
        result: dict[str, pd.DataFrame] = {}

        for ref_sym in ref_symbols:
            if ref_sym.upper() == symbol.upper():
                continue
            try:
                ref_df = self._fetcher.get_ohlcv(ref_sym, interval, lookback_days)
                result[ref_sym] = ref_df
            except Exception as exc:
                logger.warning(f"Referenzdaten für {ref_sym} nicht ladbar: {exc}")

        return result

    def _load_sentiment(self, symbol: str) -> dict[str, Any]:
        """Lädt alle Sentiment-Daten.

        Args:
            symbol: Coin-Symbol.

        Returns:
            Dict mit 'fear_greed' und 'reddit' Schlüsseln.
        """
        fg_data = self._sentiment_fetcher.get_fear_greed(history_days=10)
        reddit_data = self._sentiment_fetcher.get_reddit_sentiment(symbol)
        return {"fear_greed": fg_data, "reddit": reddit_data}

    def _get_data_age_minutes(
        self, symbol: str, interval: str, lookback_days: int
    ) -> float:
        """Berechnet das Alter der gecachten Daten in Minuten.

        Args:
            symbol: Coin-Symbol.
            interval: Kerzeni-Intervall.
            lookback_days: Historische Tage.

        Returns:
            Alter in Minuten (0.0 wenn nicht bestimmbar).
        """
        cache_key = f"binance_ohlcv_{symbol.upper()}USDT_{interval}_{lookback_days}d"
        age_seconds = self._fetcher.cache.get_entry_age(cache_key)
        if age_seconds is None:
            return 0.0
        return age_seconds / 60.0

    def _error_result(self, symbol: str, error_message: str) -> AnalysisResult:
        """Erstellt ein Fehler-AnalysisResult.

        Args:
            symbol: Coin-Symbol.
            error_message: Menschenlesbarer Fehlertext.

        Returns:
            AnalysisResult mit error-Feld gesetzt.
        """
        return AnalysisResult(
            symbol=symbol.upper(),
            ohlcv=pd.DataFrame(),
            market_data={},
            prediction=None,
            eval_metrics=None,
            feature_importance={},
            sentiment={},
            warnings=[],
            error=error_message,
            training_time_seconds=0.0,
            data_freshness_minutes=0.0,
        )
