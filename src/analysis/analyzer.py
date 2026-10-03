"""Orchestrierungs-Schicht: Koordiniert alle Komponenten zu einer Analyse.

Der Analyzer ist der einzige Einstiegspunkt für Dashboard und CLI.

Ablauf einer Analyse:
  1. Alle Datenquellen **parallel** laden (Kurse, Referenz-Coins, Marktdaten,
     Fear & Greed, Reddit)
  2. Validieren & Feature-Matrix bauen
  3. Modell aus dem Cache laden oder per purged Walk-Forward neu trainieren
  4. Out-of-Sample-Evaluierung + Backtest
  5. Live-Vorhersage für die letzte abgeschlossene Kerze

Degradiert kontrolliert: Reicht die Historie nicht für ML, werden Charts,
Indikatoren und Sentiment trotzdem geliefert (``ml_error`` erklärt warum).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_CONFIG_PATH,
    Timeframe,
    config_fingerprint,
    load_config,
    per_interval,
    resolve_path,
)
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.features.pipeline import FeatureMatrix, FeaturePipeline
from src.features.sentiment import SentimentFetcher
from src.features.technical import TechnicalIndicators
from src.models.backtest import BacktestResult, performance_stats, run_backtest
from src.models.evaluator import AggregatedMetrics, ModelEvaluator, VolatilityMetrics
from src.models.predictor import PredictionResult, Predictor
from src.models.trainer import ModelTrainer, TrainingResult

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]


@dataclass
class AnalysisResult:
    """Vollständiges Ergebnis einer Coin-Analyse.

    Attributes:
        symbol: Analysierter Coin (z.B. "BTC").
        interval: Kerzen-Intervall.
        ohlcv: OHLCV + Indikatoren für den Chart-Zeitraum.
        market_data: Marktdaten (Binance-Ticker + CoinGecko).
        prediction: KI-Vorhersage oder None.
        eval_metrics: Out-of-Sample-Metriken des Richtungsmodells.
        volatility_metrics: Out-of-Sample-Metriken des Volatilitätsmodells.
        backtest: Backtest der Modellsignale mit Standardparametern.
        oof: Out-of-Fold-Vorhersagen (für interaktive Backtests im UI).
        feature_importance: Feature → Anteil an der positiven Permutation-Importance (%).
        sentiment: Fear & Greed, Reddit, CoinGecko-Community.
        warnings: Nicht-kritische Hinweise.
        error: Kritischer Fehler (keine Daten) oder None.
        ml_error: Grund, warum keine ML-Analyse möglich war, oder None.
        training_time_seconds: Trainingsdauer (0 bei Cache).
        data_freshness_minutes: Alter der Kursdaten.
        model_info: Engine, Trainingszeitpunkt, Samples, Label-Verteilung, ...
        data_quality: Kennzahlen aus der Validierung.
        timings: Laufzeiten der Einzelschritte in Sekunden.
        periods_per_year: Kerzen pro Jahr (für Annualisierung im UI).
    """

    symbol: str
    interval: str
    ohlcv: pd.DataFrame
    market_data: dict[str, Any]
    prediction: PredictionResult | None
    eval_metrics: AggregatedMetrics | None
    volatility_metrics: VolatilityMetrics | None
    backtest: BacktestResult | None
    oof: pd.DataFrame | None
    feature_importance: dict[str, float]
    sentiment: dict[str, Any]
    warnings: list[str]
    error: str | None
    ml_error: str | None = None
    training_time_seconds: float = 0.0
    data_freshness_minutes: float | None = None
    model_info: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    periods_per_year: float = 365.0


@dataclass
class ComparisonResult:
    """Ergebnis eines Coin-Vergleichs.

    Attributes:
        normalized: Kursentwicklung je Coin, normiert auf 100 zum gemeinsamen Start.
        correlation: Korrelationsmatrix der Log-Returns.
        stats: Kennzahlen je Coin (Return, Volatilität, Sharpe, Max-Drawdown, Beta).
        errors: Symbol → Fehlermeldung für nicht ladbare Coins.
    """

    normalized: pd.DataFrame
    correlation: pd.DataFrame
    stats: pd.DataFrame
    errors: dict[str, str]


class CryptoAnalyzer:
    """Führt eine vollständige KI-gestützte Krypto-Analyse durch.

    Args:
        config_path: Pfad zur config.yaml.
        fetcher: Optionaler DataFetcher (für Tests).
        sentiment_fetcher: Optionaler SentimentFetcher (für Tests).
    """

    def __init__(
        self,
        config_path: Path | None = None,
        fetcher: DataFetcher | None = None,
        sentiment_fetcher: SentimentFetcher | None = None,
    ) -> None:
        self._config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self._config = load_config(self._config_path)
        cfg = self._config

        self._fetcher = fetcher or DataFetcher(cfg, self._config_path)
        self._sentiment = sentiment_fetcher or SentimentFetcher(cfg, self._fetcher.cache, self._fetcher.http)
        self._validator = DataValidator(
            max_consecutive_gaps=cfg["validation"]["max_consecutive_fillable_gaps"],
            volume_iqr_multiplier=cfg["validation"]["volume_iqr_multiplier"],
            return_zscore_threshold=cfg["validation"]["return_zscore_threshold"],
            minimum_samples=cfg["validation"]["minimum_samples_for_chart"],
        )
        self._trainer = ModelTrainer(cfg, resolve_path(self._config_path, cfg["paths"]["models_dir"]))
        self._predictor = Predictor(cfg)
        self._evaluator = ModelEvaluator(cfg)
        self._fingerprint = config_fingerprint(cfg)

    @property
    def config(self) -> dict[str, Any]:
        """Geladene Konfiguration."""
        return self._config

    @property
    def fetcher(self) -> DataFetcher:
        """Datenzugriff (z.B. für den Coin-Vergleich im UI)."""
        return self._fetcher

    @property
    def engine(self) -> str:
        """Aktive ML-Engine."""
        return self._trainer.engine

    # ------------------------------------------------------------------
    # Hauptanalyse
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        interval: str = "1d",
        lookback_days: int = 365,
        force_retrain: bool = False,
        progress: ProgressCallback | None = None,
        include_sentiment: bool = True,
    ) -> AnalysisResult:
        """Führt die vollständige Analyse eines Coins durch.

        Args:
            symbol: Coin-Symbol (z.B. "BTC", "doge", "PEPEUSDT").
            interval: Kerzen-Intervall ("1h", "4h", "1d").
            lookback_days: Chart-Zeitraum in Tagen (Training nutzt ggf. mehr Historie).
            force_retrain: Neu trainieren, auch wenn ein frisches Modell existiert.
            progress: Callback für Fortschrittsmeldungen (UI).
            include_sentiment: Reddit/Marktdaten laden (für Scanner abschaltbar).

        Returns:
            AnalysisResult.
        """
        notify = progress or (lambda _msg: None)
        timings: dict[str, float] = {}
        warnings: list[str] = []

        try:
            sym = self._fetcher.normalize_symbol(symbol)
            tf = Timeframe(interval)
        except ValueError as exc:
            return self._error_result(str(symbol).upper(), interval, str(exc))

        # ── 1. Daten parallel laden ────────────────────────────────────────
        notify(f"Lade Kurs-, Markt- und Sentimentdaten für {sym}…")
        t0 = time.perf_counter()
        warmup_days = self._config["features"]["ema_long_long"] / tf.bars_per_day
        train_days = per_interval(self._config["ml"]["training_lookback_days"], interval)
        fetch_days = max(lookback_days + warmup_days, train_days)
        refs = [r for r in self._config["features"]["reference_symbols"] if r.upper() != sym]

        with ThreadPoolExecutor(max_workers=6) as pool:
            f_main = pool.submit(self._fetcher.get_ohlcv, sym, interval, fetch_days)
            f_refs = {r: pool.submit(self._fetcher.get_ohlcv, r, interval, fetch_days) for r in refs}
            f_fng = pool.submit(self._sentiment.get_fear_greed_history)
            f_market = pool.submit(self._fetcher.get_market_data, sym) if include_sentiment else None
            f_reddit = (
                pool.submit(self._reddit_with_aliases, sym, f_market) if include_sentiment else None
            )

            try:
                raw_df = f_main.result()
            except Exception as exc:
                # Nicht auf Reddit & Co. warten, wenn es ohnehin keine Kursdaten gibt
                pool.shutdown(wait=False, cancel_futures=True)
                if isinstance(exc, ValueError):
                    return self._error_result(sym, interval, str(exc))
                logger.exception(f"[{sym}] Kursdaten nicht ladbar")
                return self._error_result(
                    sym, interval,
                    f"Kursdaten konnten nicht geladen werden: {exc}. Prüfe deine Internetverbindung.",
                )

            reference_data: dict[str, pd.DataFrame] = {}
            for ref, fut in f_refs.items():
                try:
                    reference_data[ref] = fut.result()
                except Exception as exc:
                    warnings.append(f"Referenzdaten für {ref} nicht verfügbar: {exc}")

            fear_greed = _safe_result(f_fng, pd.Series(dtype="float64"))
            market_data = _safe_result(f_market, {"name": sym, "symbol": sym}) if f_market else {"name": sym}
            reddit = _safe_result(f_reddit, {}) if f_reddit else {}
        timings["daten"] = time.perf_counter() - t0

        # ── 2. Validieren ──────────────────────────────────────────────────
        validation = self._validator.validate(raw_df, sym, interval)
        warnings.extend(validation.warnings)
        if not validation.is_valid:
            return self._error_result(sym, interval, "; ".join(validation.errors))
        df = validation.df

        sentiment = {
            "fear_greed": self._sentiment.get_fear_greed() if not fear_greed.empty else {},
            "reddit": reddit,
            "community_up_pct": market_data.get("sentiment_up_pct"),
        }

        # ── 3. Features ────────────────────────────────────────────────────
        notify("Berechne Indikatoren und Features…")
        t0 = time.perf_counter()
        pipeline = FeaturePipeline(self._config, interval)
        try:
            fm = pipeline.build(df, reference_data, fear_greed)
        except Exception as exc:
            logger.exception(f"[{sym}] Feature-Pipeline fehlgeschlagen")
            return self._partial_result(
                sym, interval, df, lookback_days, market_data, sentiment, warnings,
                f"Feature-Berechnung fehlgeschlagen: {exc}", timings, validation.stats, tf,
            )
        timings["features"] = time.perf_counter() - t0
        chart_frame = _tail_days(fm.frame, lookback_days)

        # ── 4. Modell laden oder trainieren ───────────────────────────────
        t0 = time.perf_counter()
        try:
            training = self._get_or_train(sym, interval, fm, force_retrain, notify)
        except ValueError as exc:
            return self._partial_result(
                sym, interval, fm.frame, lookback_days, market_data, sentiment, warnings,
                str(exc), timings, validation.stats, tf,
            )
        timings["training"] = time.perf_counter() - t0

        # ── 5. Evaluierung, Backtest, Vorhersage ──────────────────────────
        notify("Evaluiere Modell und erstelle Vorhersage…")
        eval_metrics = self._evaluator.evaluate(training.oof, training.horizon_bars)
        vol_metrics = self._evaluator.evaluate_volatility(training.oof, training.horizon_bars)
        bt_cfg = self._config["backtest"]
        backtest = run_backtest(
            training.oof,
            threshold=self._config["ml"]["confidence_display_threshold"],
            periods_per_year=tf.periods_per_year,
            fee_bps=bt_cfg["fee_bps"],
            slippage_bps=bt_cfg["slippage_bps"],
            allow_short=bt_cfg["allow_short"],
            min_hold_bars=bt_cfg["min_hold_bars"],
        )
        prediction = self._predictor.predict(
            direction_model=training.direction_model,
            volatility_model=training.volatility_model,
            feature_row=fm.last_row,
            feature_names=training.feature_names,
            required_features=[f for f in fm.required_features if f in training.feature_names],
            interval=interval,
            horizon_bars=fm.horizon_bars,
            thresholds=fm.current_thresholds,
            data_end_date=fm.data_end_date,
            signal_stats=eval_metrics.signal_stats if eval_metrics else None,
            model_verdict=eval_metrics.verdict if eval_metrics else "",
            calibration=training.calibration,
            volatility_calibration=training.volatility_calibration,
            class_prior=training.class_prior,
            volatility_prior=training.volatility_prior,
        )

        return AnalysisResult(
            symbol=sym,
            interval=interval,
            ohlcv=chart_frame,
            market_data=market_data,
            prediction=prediction,
            eval_metrics=eval_metrics,
            volatility_metrics=vol_metrics,
            backtest=backtest,
            oof=training.oof,
            feature_importance=_normalize_importance(training.feature_importance),
            sentiment=sentiment,
            warnings=warnings,
            error=None,
            ml_error=None,
            training_time_seconds=0.0 if training.from_cache else training.training_seconds,
            data_freshness_minutes=self._fetcher.data_age_minutes(sym, interval),
            model_info={
                "engine": training.engine,
                "calibration": training.calibration,
                "trained_at": training.trained_at,
                "from_cache": training.from_cache,
                "n_samples": training.n_samples,
                "n_features": len(training.feature_names),
                "n_folds": len(training.folds),
                "folds": training.folds,
                "horizon_bars": training.horizon_bars,
                "horizon_text": tf.describe_bars(training.horizon_bars),
                "label_distribution": fm.label_distribution,
                "thresholds_pct": (fm.current_thresholds[0] * 100, fm.current_thresholds[1] * 100),
                "training_start": str(fm.X.index[0]),
                "training_end": str(fm.X.index[-1]),
            },
            data_quality=validation.stats,
            timings={k: round(v, 2) for k, v in timings.items()},
            periods_per_year=tf.periods_per_year,
        )

    # ------------------------------------------------------------------
    # Weitere Analysen
    # ------------------------------------------------------------------

    def compare(self, symbols: list[str], interval: str = "1d", lookback_days: int = 180) -> ComparisonResult:
        """Vergleicht mehrere Coins: relative Performance, Korrelation, Risiko-Kennzahlen."""
        clean: list[str] = []
        errors: dict[str, str] = {}
        for s in symbols:
            try:
                sym = self._fetcher.normalize_symbol(s)
                if sym not in clean:
                    clean.append(sym)
            except ValueError as exc:
                errors[s] = str(exc)

        frames, fetch_errors = self._fetcher.get_many_ohlcv(clean, interval, lookback_days)
        errors.update(fetch_errors)
        if not frames:
            empty = pd.DataFrame()
            return ComparisonResult(empty, empty, empty, errors)

        closes = pd.DataFrame({s: frames[s]["close"] for s in clean if s in frames})
        start = max(closes[c].first_valid_index() for c in closes.columns)
        closes = closes[closes.index >= start].ffill()
        normalized = closes / closes.iloc[0] * 100.0
        log_ret = np.log(closes / closes.shift(1)).dropna(how="all")

        tf = Timeframe(interval)
        benchmark = log_ret.iloc[:, 0]
        rows = []
        for sym in closes.columns:
            stats = performance_stats(closes[sym].pct_change().fillna(0.0), tf.periods_per_year)
            cov = log_ret[sym].cov(benchmark)
            var = benchmark.var()
            rows.append({
                "Symbol": sym,
                "Return": stats["total_return"],
                "Volatilität (ann.)": stats["ann_volatility"],
                "Sharpe": stats["sharpe"],
                "Max. Drawdown": stats["max_drawdown"],
                "Abstand zum Hoch": float(closes[sym].iloc[-1] / closes[sym].max() - 1.0),
                f"Beta zu {closes.columns[0]}": float(cov / var) if var > 0 else float("nan"),
            })
        return ComparisonResult(
            normalized=normalized,
            correlation=log_ret.corr(),
            stats=pd.DataFrame(rows).set_index("Symbol"),
            errors=errors,
        )

    def scan(
        self,
        symbols: list[str],
        interval: str = "1d",
        progress: Callable[[int, int, str], None] | None = None,
        max_workers: int = 3,
    ) -> pd.DataFrame:
        """Analysiert mehrere Coins und liefert eine Signal-Übersicht.

        Returns:
            DataFrame mit Signal, Konfidenz, Vola-Regime und Modellqualität je Coin.
        """
        rows: list[dict[str, Any]] = []

        def run(sym: str) -> dict[str, Any]:
            res = self.analyze(sym, interval, lookback_days=90, include_sentiment=False)
            row: dict[str, Any] = {"Symbol": res.symbol}
            if res.error or res.prediction is None:
                row["Signal"] = "–"
                row["Hinweis"] = (res.error or res.ml_error or "")[:120]
                return row
            p, m, v = res.prediction, res.eval_metrics, res.volatility_metrics
            last = res.ohlcv["close"]
            bars_7d = Timeframe(interval).bars(7)
            row.update({
                "Kurs": float(last.iloc[-1]),
                "7d %": float(last.iloc[-1] / last.iloc[-1 - bars_7d] - 1) * 100 if len(last) > bars_7d else None,
                "Signal": f"{p.direction_emoji} {p.direction_label}" if p.show_signal else "⚪ kein Signal",
                "P(UP)": p.probabilities.get("BULLISH"),
                "P(DOWN)": p.probabilities.get("BEARISH"),
                "OOS-Skill": m.skill_score if m else None,
                "p-Wert": m.p_value if m else None,
                "Richtungsmodell": m.verdict if m else "",
                "Volatilität": p.volatility_label,
                "Vola-Skill": v.skill_score if v else None,
                "Vola-Modell": v.verdict if v else "",
                "Backtest Sharpe": res.backtest.strategy["sharpe"] if res.backtest else None,
                "B&H Sharpe": res.backtest.buy_hold["sharpe"] if res.backtest else None,
                "Hinweis": "",
            })
            return row

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run, s): s for s in symbols}
            for fut in futures:
                sym = futures[fut]
                try:
                    rows.append(fut.result())
                except Exception as exc:  # ein Coin darf den Scan nicht abbrechen
                    logger.exception(f"Scan für {sym} fehlgeschlagen")
                    rows.append({"Symbol": sym, "Signal": "–", "Hinweis": str(exc)[:120]})
                done += 1
                if progress:
                    progress(done, len(symbols), sym)
        return pd.DataFrame(rows)

    def get_trending_coins(self) -> list[dict[str, Any]]:
        """Aktuell trendende Coins (CoinGecko)."""
        return self._fetcher.get_trending_coins()

    def get_universe(self, limit: int = 100) -> list[dict[str, Any]]:
        """Top-Coins nach 24h-Volumen (Binance)."""
        return self._fetcher.get_universe(limit)

    def clear_cache(self) -> int:
        """Löscht den Daten-Cache. Gibt die Anzahl gelöschter Dateien zurück."""
        return self._fetcher.cache.clear_all()

    def cache_stats(self) -> dict[str, float]:
        """Größe des Daten-Caches."""
        return self._fetcher.cache.stats()

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _get_or_train(
        self,
        symbol: str,
        interval: str,
        fm: FeatureMatrix,
        force_retrain: bool,
        notify: ProgressCallback,
    ) -> TrainingResult:
        """Lädt ein frisches, kompatibles Modell oder trainiert neu."""
        if not force_retrain:
            cached = self._trainer.load(symbol, interval, self._fingerprint)
            ttl_hours = per_interval(self._config["cache"]["model_ttl_hours"], interval)
            if (
                cached is not None
                and cached.feature_names == fm.feature_names
                and cached.age_seconds < ttl_hours * 3600
            ):
                logger.info(f"[{symbol} {interval}] Modell aus Cache (Alter {cached.age_seconds / 60:.0f} Min.).")
                return cached

        n_folds = len(self._trainer.plan_folds(len(fm.X), interval, fm.horizon_bars, extra_folds=1))
        notify(f"Trainiere KI-Modelle ({n_folds} Walk-Forward-Folds, Engine: {self._trainer.engine})…")
        return self._trainer.train(fm, symbol, interval, self._fingerprint)

    def _reddit_with_aliases(self, symbol: str, market_future: Any) -> dict[str, Any]:
        aliases: list[str] = []
        try:
            name = market_future.result().get("name") if market_future else None
            if name and name.upper() != symbol:
                aliases.append(name)
        except Exception:
            pass
        return self._sentiment.get_reddit_sentiment(symbol, aliases)

    def _partial_result(
        self,
        symbol: str,
        interval: str,
        frame: pd.DataFrame,
        lookback_days: int,
        market_data: dict[str, Any],
        sentiment: dict[str, Any],
        warnings: list[str],
        ml_error: str,
        timings: dict[str, float],
        quality: dict[str, Any],
        tf: Timeframe,
    ) -> AnalysisResult:
        """Ergebnis ohne ML (Charts, Marktdaten und Sentiment bleiben verfügbar)."""
        logger.info(f"[{symbol} {interval}] ML nicht möglich: {ml_error}")
        if "bb_mid" not in frame.columns:
            try:
                frame = TechnicalIndicators(self._config, interval).add_all(frame)
            except Exception:
                logger.exception(f"[{symbol}] Indikatoren für Chart nicht berechenbar")
        return AnalysisResult(
            symbol=symbol,
            interval=interval,
            ohlcv=_tail_days(frame, lookback_days),
            market_data=market_data,
            prediction=None,
            eval_metrics=None,
            volatility_metrics=None,
            backtest=None,
            oof=None,
            feature_importance={},
            sentiment=sentiment,
            warnings=warnings,
            error=None,
            ml_error=ml_error,
            data_freshness_minutes=self._fetcher.data_age_minutes(symbol, interval),
            data_quality=quality,
            timings={k: round(v, 2) for k, v in timings.items()},
            periods_per_year=tf.periods_per_year,
        )

    @staticmethod
    def _error_result(symbol: str, interval: str, error_message: str) -> AnalysisResult:
        return AnalysisResult(
            symbol=symbol,
            interval=interval,
            ohlcv=pd.DataFrame(),
            market_data={},
            prediction=None,
            eval_metrics=None,
            volatility_metrics=None,
            backtest=None,
            oof=None,
            feature_importance={},
            sentiment={},
            warnings=[],
            error=error_message,
        )


def _safe_result(future: Any, default: Any) -> Any:
    try:
        return future.result()
    except Exception as exc:
        logger.warning(f"Optionale Datenquelle fehlgeschlagen: {exc}")
        return default


def _tail_days(frame: pd.DataFrame, days: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    cutoff = frame.index[-1] - pd.Timedelta(days=days)
    return frame[frame.index > cutoff]


def _normalize_importance(importance: dict[str, float]) -> dict[str, float]:
    """Skaliert Permutation-Importances auf % der positiven Gesamtwichtigkeit (absteigend)."""
    positive_total = sum(v for v in importance.values() if v > 0)
    if positive_total <= 0:
        return {}
    return {
        k: round(v / positive_total * 100, 2)
        for k, v in sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    }
