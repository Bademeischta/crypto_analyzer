"""Feature-Engineering-Pipeline: Baut die vollständige Feature-Matrix für ML.

Orchestriert die technischen Indikatoren, fügt Marktstruktur-Features
(Korrelation/Beta/relative Stärke zu BTC & ETH) und die Fear-&-Greed-Historie
hinzu und erstellt die Zielvariablen.

Kritische ML-Eigenschaft – kein Lookahead-Bias:
  * Jedes Feature in Zeile t nutzt nur Informationen, die beim Schluss der
    Kerze t bekannt waren (per Test verifiziert: Anhängen zukünftiger Kerzen
    ändert keine vergangenen Feature-Werte).
  * Fear & Greed eines Tages wird erst ab dem Folgetag verwendet.
  * Klassengrenzen der Labels basieren nur auf Vergangenheitsdaten
    (kein globales Quantil über den gesamten Datensatz).

Labels:
  * Richtung (volatilitätsadjustiert): UP, wenn der Return der nächsten h
    Kerzen > k · σ_t · √h ist (σ_t = aktuelle Volatilität pro Kerze), DOWN
    bei < −k · σ_t · √h, sonst NEUTRAL. Ein fixer ±2%-Schwellenwert wäre für
    BTC zu weit und für PEPE zu eng – so sind die Klassen für jeden Coin
    sinnvoll balanciert.
  * Volatilitätsregime: realisierte Vola der nächsten h Kerzen im Vergleich
    zu den Quantilen der realisierten Vola des vergangenen Jahres.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import Timeframe, per_interval
from src.features.technical import TechnicalIndicators

logger = logging.getLogger(__name__)

DIRECTION_CLASSES = (0, 1, 2)  # DOWN, NEUTRAL, UP
VOLATILITY_CLASSES = (0, 1, 2)  # LOW, MEDIUM, HIGH


@dataclass
class FeatureMatrix:
    """Ergebnis der Feature-Pipeline.

    Attributes:
        X: Feature-Matrix (nur Zeilen mit bekanntem Label). Index: DatetimeIndex.
        y_direction: Richtungs-Label (0=DOWN, 1=NEUTRAL, 2=UP).
        y_volatility: Volatilitäts-Label (0=LOW, 1=MEDIUM, 2=HIGH).
        forward_return: Einfacher Return über den Horizont (aligned zu X).
        next_return: Einfacher Return der nächsten Kerze (aligned zu X, für Backtests).
        feature_names: Spaltennamen von X.
        required_features: Features, die für eine Vorhersage vorhanden sein müssen
            (optionale wie Korrelationen/Sentiment dürfen NaN sein).
        last_row: Aktuellster Feature-Vektor (für die Live-Vorhersage).
        last_timestamp: Öffnungszeit der letzten abgeschlossenen Kerze.
        frame: Voller Indikator-DataFrame (für Charts).
        horizon_bars: Vorhersage-Horizont in Kerzen.
        current_thresholds: (down, up) Return-Schwellen für die aktuelle Kerze.
        label_distribution: Anteil je Richtungsklasse im Trainingsdatensatz.
    """

    X: pd.DataFrame
    y_direction: pd.Series
    y_volatility: pd.Series
    forward_return: pd.Series
    next_return: pd.Series
    feature_names: list[str]
    required_features: list[str]
    last_row: pd.Series
    last_timestamp: pd.Timestamp
    frame: pd.DataFrame
    horizon_bars: int
    current_thresholds: tuple[float, float]
    label_distribution: dict[str, float] = field(default_factory=dict)

    @property
    def data_end_date(self) -> str:
        """Letzte abgeschlossene Kerze als lesbarer String."""
        return self.last_timestamp.strftime("%Y-%m-%d %H:%M UTC")


class FeaturePipeline:
    """Baut die vollständige Feature-Matrix aus OHLCV-Rohdaten.

    Args:
        config: Geladenes config.yaml als Dict.
        interval: Kerzen-Intervall der Daten.
    """

    def __init__(self, config: dict[str, Any], interval: str = "1d") -> None:
        self._cfg = config
        self._tf = Timeframe(interval)
        self._tech = TechnicalIndicators(config, interval)
        self._ml_cfg = config["ml"]
        self._feat_cfg = config["features"]

    @property
    def horizon_bars(self) -> int:
        """Vorhersage-Horizont in Kerzen des aktuellen Intervalls."""
        return self._tf.bars(self._ml_cfg["direction"]["horizon_days"])

    def build(
        self,
        df: pd.DataFrame,
        reference_data: dict[str, pd.DataFrame] | None = None,
        fear_greed: pd.Series | None = None,
    ) -> FeatureMatrix:
        """Erstellt die vollständige Feature-Matrix.

        Args:
            df: Validiertes OHLCV-DataFrame.
            reference_data: Symbol → OHLCV-DataFrame (z.B. BTC, ETH) für Marktstruktur-Features.
            fear_greed: Tägliche Fear-&-Greed-Historie (Index = Tagesbeginn UTC).

        Returns:
            FeatureMatrix.
        """
        frame = self._tech.add_all(df)
        required = self._tech.feature_columns(frame)
        optional: list[str] = []

        if reference_data:
            market = self._market_structure_features(frame, reference_data)
            frame = pd.concat([frame, market], axis=1)
            optional.extend(market.columns)

        if fear_greed is not None and not fear_greed.empty:
            fng = self._fear_greed_features(frame.index, fear_greed)
            frame = pd.concat([frame, fng], axis=1)
            optional.extend(fng.columns)

        h = self.horizon_bars
        close = frame["close"]
        log_ret = np.log(close / close.shift(1))

        y_dir, up_thr, down_thr = self._direction_target(close, log_ret, h)
        y_vol = self._volatility_target(log_ret, h)
        forward_return = close.shift(-h) / close - 1.0
        next_return = close.shift(-1) / close - 1.0

        feature_names = required + optional
        last_row = frame[feature_names].iloc[-1].copy()

        labeled_mask = y_dir.notna() & y_vol.notna() & frame[required].notna().all(axis=1)
        labeled_idx = frame.index[labeled_mask]
        max_bars = self._ml_cfg.get("max_training_bars")
        if max_bars and len(labeled_idx) > max_bars:
            labeled_idx = labeled_idx[-max_bars:]

        X = frame.loc[labeled_idx, feature_names].astype("float64")
        y_direction = y_dir.loc[labeled_idx].astype(int)
        distribution = y_direction.value_counts(normalize=True).reindex(DIRECTION_CLASSES, fill_value=0.0)

        if len(X) < 100:
            logger.warning(f"Nur {len(X)} gelabelte Zeilen – ML-Ergebnisse sind wenig belastbar.")

        return FeatureMatrix(
            X=X,
            y_direction=y_direction,
            y_volatility=y_vol.loc[labeled_idx].astype(int),
            forward_return=forward_return.loc[labeled_idx],
            next_return=next_return.loc[labeled_idx],
            feature_names=feature_names,
            required_features=required,
            last_row=last_row,
            last_timestamp=frame.index[-1],
            frame=frame,
            horizon_bars=h,
            current_thresholds=(float(down_thr.iloc[-1]), float(up_thr.iloc[-1])),
            label_distribution={
                "DOWN": float(distribution[0]), "NEUTRAL": float(distribution[1]), "UP": float(distribution[2]),
            },
        )

    # ------------------------------------------------------------------
    # Zusatz-Features
    # ------------------------------------------------------------------

    def _market_structure_features(
        self, frame: pd.DataFrame, reference_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Rollende Korrelation, Beta und relative Stärke gegenüber Referenz-Coins.

        Altcoins laufen stark mit BTC mit; eine Entkopplung (sinkende
        Korrelation, steigende relative Stärke) ist oft informativ.
        """
        w = self._feat_cfg["correlation_window"]
        p = max(self._feat_cfg["return_periods"])
        coin_ret = np.log(frame["close"] / frame["close"].shift(1))
        coin_ret_p = np.log(frame["close"] / frame["close"].shift(p))
        out: dict[str, pd.Series] = {}

        for symbol, ref_df in reference_data.items():
            if ref_df is None or ref_df.empty:
                continue
            name = symbol.lower()
            ref_close = ref_df["close"].reindex(frame.index)
            if ref_close.notna().sum() < w * 2:
                logger.info(f"Zu wenig überlappende Daten mit {symbol} – Marktstruktur-Features übersprungen.")
                continue
            ref_ret = np.log(ref_close / ref_close.shift(1))
            ref_var = ref_ret.rolling(w, min_periods=w).var().replace(0, np.nan)
            out[f"corr_{name}"] = coin_ret.rolling(w, min_periods=w).corr(ref_ret)
            out[f"beta_{name}"] = coin_ret.rolling(w, min_periods=w).cov(ref_ret) / ref_var
            out[f"rel_strength_{name}"] = coin_ret_p - np.log(ref_close / ref_close.shift(p))

        return pd.DataFrame(out, index=frame.index)

    def _fear_greed_features(self, index: pd.DatetimeIndex, fear_greed: pd.Series) -> pd.DataFrame:
        """Richtet die tägliche F&G-Historie zeitlich korrekt an den Kerzen aus.

        Der Wert eines Tages gilt erst ab dem Folgetag als bekannt (+1 Tag) und
        wird der Kerze zugeordnet, deren *Schlusszeit* danach liegt.
        """
        daily = fear_greed.sort_index()
        change_7d = daily - daily.shift(7)
        available = pd.DataFrame({"fng_value": daily, "fng_change_7d": change_7d})
        available.index = available.index + pd.Timedelta(days=1)

        # Einheitliche Auflösung, sonst verweigert merge_asof den Join (pandas ≥ 2)
        close_times = (index + self._tf.delta).astype("datetime64[ns, UTC]")
        available.index = available.index.astype("datetime64[ns, UTC]")
        aligned = pd.merge_asof(
            pd.DataFrame({"t": close_times}),
            available.rename_axis("t").reset_index(),
            on="t",
            direction="backward",
            tolerance=pd.Timedelta(days=3),  # alte Werte nach API-Lücken nicht ewig fortschreiben
        )
        aligned.index = index
        return aligned[["fng_value", "fng_change_7d"]].astype("float64")

    # ------------------------------------------------------------------
    # Zielvariablen
    # ------------------------------------------------------------------

    def _direction_target(
        self, close: pd.Series, log_ret: pd.Series, h: int
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Richtungs-Label (0/1/2) plus Up-/Down-Schwellen je Zeile.

        Die letzten h Werte sind NaN (Zukunft unbekannt).
        """
        cfg = self._ml_cfg["direction"]
        forward = close.shift(-h) / close - 1.0

        if cfg["label_mode"] == "volatility_adjusted":
            span = self._tf.bars(cfg["vol_lookback_days"], minimum=5)
            sigma = log_ret.ewm(span=span, adjust=False, min_periods=max(5, span // 2)).std()
            thr = (cfg["vol_multiplier"] * sigma * np.sqrt(h)).clip(lower=cfg["min_threshold"])
            up_thr, down_thr = thr, -thr
        else:
            up_thr = pd.Series(cfg["up_threshold"], index=close.index)
            down_thr = pd.Series(cfg["down_threshold"], index=close.index)

        label = pd.Series(np.nan, index=close.index)
        valid = forward.notna() & up_thr.notna()
        label[valid & (forward > up_thr)] = 2.0
        label[valid & (forward < down_thr)] = 0.0
        label[valid & (forward >= down_thr) & (forward <= up_thr)] = 1.0
        return label, up_thr, down_thr

    def _volatility_target(self, log_ret: pd.Series, h: int) -> pd.Series:
        """Volatilitätsregime der nächsten h Kerzen (0=LOW, 1=MEDIUM, 2=HIGH).

        Vergleicht die zukünftige realisierte Vola mit rollenden Quantilen der
        *vergangenen* realisierten Vola – die Klassengrenzen enthalten somit
        keine Zukunftsinformation.
        """
        cfg = self._ml_cfg["volatility"]
        win = max(h, 2)
        lookback = self._tf.bars(per_interval(cfg["regime_lookback_days"], self._tf.interval), minimum=20)

        past_vol = log_ret.rolling(win, min_periods=win).std()
        future_vol = past_vol.shift(-win)
        low = past_vol.rolling(lookback, min_periods=lookback // 2).quantile(cfg["low_quantile"])
        high = past_vol.rolling(lookback, min_periods=lookback // 2).quantile(cfg["high_quantile"])

        label = pd.Series(np.nan, index=log_ret.index)
        valid = future_vol.notna() & low.notna() & high.notna()
        label[valid & (future_vol <= low)] = 0.0
        label[valid & (future_vol > high)] = 2.0
        label[valid & (future_vol > low) & (future_vol <= high)] = 1.0
        return label
