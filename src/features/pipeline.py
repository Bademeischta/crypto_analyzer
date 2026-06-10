"""Feature-Engineering-Pipeline: Baut die vollständige Feature-Matrix für ML.

Orchestriert TechnicalIndicators, fügt Marktstruktur-Features (Korrelationen)
hinzu und erstellt die Zielvariablen für Richtungs- und Volatilitätsklassifikation.

Kritische ML-Eigenschaft: Keine Zukunftsdaten in Features (kein Lookahead-Bias).
Alle Features basieren ausschließlich auf Informationen, die zum Zeitpunkt t
bekannt waren. Das Ziel (Return in nächsten N Tagen) wird erst nach Feature-
Berechnung als separater Schritt erstellt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class FeatureMatrix:
    """Ergebnis der Feature-Pipeline.

    Attributes:
        X: Feature-Matrix (NaN-bereinigt). Index: DatetimeIndex.
        y_direction: Richtungs-Zielvariable (0=DOWN, 1=NEUTRAL, 2=UP).
        y_volatility: Volatilitäts-Zielvariable (0=LOW, 1=MEDIUM, 2=HIGH).
        feature_names: Spaltenbezeichnungen von X.
        last_row: Letzter Feature-Vektor (für Live-Prediction, ohne Zukunfts-Target).
        data_end_date: Letztes Datum in den Daten.
    """

    X: pd.DataFrame
    y_direction: pd.Series
    y_volatility: pd.Series
    feature_names: list[str]
    last_row: pd.Series
    data_end_date: str


class FeaturePipeline:
    """Baut die vollständige Feature-Matrix aus OHLCV-Rohdaten.

    Args:
        config: Geladenes config.yaml als Dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._tech = TechnicalIndicators(config)
        self._ml_cfg = config["ml"]
        self._feat_cfg = config["features"]

    def build(
        self,
        df: pd.DataFrame,
        reference_data: dict[str, pd.DataFrame] | None = None,
        sentiment_features: dict[str, float] | None = None,
    ) -> FeatureMatrix:
        """Erstellt die vollständige Feature-Matrix.

        Args:
            df: Validiertes OHLCV-DataFrame.
            reference_data: Dict von Symbol -> OHLCV-DataFrame für Korrelations-Features.
                           Schlüssel z.B. "BTC", "ETH".
            sentiment_features: Dict mit aktuellen Sentiment-Werten
                               (werden als konstante Features hinzugefügt).

        Returns:
            FeatureMatrix mit X, y_direction, y_volatility und letzter Zeile.
        """
        # Schritt 1: Technische Indikatoren
        df_feat = self._tech.add_all(df)

        # Schritt 2: Korrelations-Features (Marktstruktur)
        if reference_data:
            df_feat = self._add_correlation_features(df_feat, reference_data)

        # Schritt 3: Sentiment als Feature (wenn verfügbar)
        if sentiment_features:
            for key, value in sentiment_features.items():
                df_feat[f"sentiment_{key}"] = value

        # Schritt 4: Zielvariablen erstellen (NACH Features, um Lookahead zu vermeiden)
        y_direction = self._create_direction_target(df_feat)
        y_volatility = self._create_volatility_target(df_feat)

        # Schritt 5: Feature-Spalten definieren (keine Rohpreise, keine Targets)
        feature_cols = self._select_feature_columns(df_feat)

        # Schritt 6: Letzten Feature-Vektor vor dem Droppen speichern (für Live-Prediction)
        last_row = df_feat[feature_cols].iloc[-1].copy()
        last_date = str(df_feat.index[-1].date())

        # Schritt 7: Nur Zeilen behalten wo alle Features UND Targets vorhanden sind
        combined = df_feat[feature_cols].copy()
        combined["_y_dir"] = y_direction
        combined["_y_vola"] = y_volatility
        combined = combined.dropna()

        if len(combined) < 50:
            logger.warning(
                f"Nur {len(combined)} vollständige Feature-Zeilen nach NaN-Drop. "
                f"ML-Ergebnisse können unzuverlässig sein."
            )

        X = combined[feature_cols]
        y_dir = combined["_y_dir"].astype(int)
        y_vola = combined["_y_vola"].astype(int)

        return FeatureMatrix(
            X=X,
            y_direction=y_dir,
            y_volatility=y_vola,
            feature_names=list(feature_cols),
            last_row=last_row,
            data_end_date=last_date,
        )

    def _add_correlation_features(
        self,
        df: pd.DataFrame,
        reference_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Fügt rollende Korrelationen zu BTC und ETH hinzu.

        Korrelationen messen wie stark der analysierte Coin mit dem Markt mitläuft.
        Abweichung von der Norm kann ein Signal sein.

        Args:
            df: Feature-DataFrame (wird modifiziert).
            reference_data: Reference-Symbol -> OHLCV-DataFrame.

        Returns:
            DataFrame mit Korrelations-Spalten.
        """
        corr_w = self._feat_cfg["correlation_window"]
        coin_log_ret = np.log(df["close"] / df["close"].shift(1))

        for symbol, ref_df in reference_data.items():
            try:
                ref_log_ret = np.log(ref_df["close"] / ref_df["close"].shift(1))
                # Gemeinsamer Index (inner join)
                aligned = pd.concat(
                    [coin_log_ret.rename("coin"), ref_log_ret.rename("ref")],
                    axis=1,
                ).dropna()

                if len(aligned) < corr_w:
                    continue

                rolling_corr = aligned["coin"].rolling(corr_w).corr(aligned["ref"])
                # Auf den Original-Index reindexen
                df[f"corr_{symbol.lower()}_{corr_w}d"] = rolling_corr.reindex(df.index)
            except Exception as exc:
                logger.warning(f"Korrelations-Feature für {symbol} fehlgeschlagen: {exc}")

        return df

    def _create_direction_target(self, df: pd.DataFrame) -> pd.Series:
        """Erstellt die Richtungs-Zielvariable.

        Ziel: Forward-Return über N Tage.
          UP     (2): Return > up_threshold
          DOWN   (0): Return < down_threshold
          NEUTRAL(1): Dazwischen

        Das Target wird für den letzten Zeitpunkt NaN (unbekannte Zukunft).

        Args:
            df: Feature-DataFrame mit close-Spalte.

        Returns:
            Series mit Werten 0 (DOWN), 1 (NEUTRAL), 2 (UP). Letzten N Werte = NaN.
        """
        horizon = self._ml_cfg["direction"]["horizon_days"]
        up_thr = self._ml_cfg["direction"]["up_threshold"]
        down_thr = self._ml_cfg["direction"]["down_threshold"]

        # Zukunfts-Return: Return in 'horizon' Tagen VORWÄRTS
        # .shift(-horizon) verschiebt Zukunftswerte auf aktuellen Index →
        # korrekt: bei Index t steht der Return der Periode [t, t+horizon]
        forward_return = df["close"].pct_change(horizon).shift(-horizon)

        direction = pd.Series(index=df.index, dtype="float64")
        direction[forward_return > up_thr] = 2.0    # UP
        direction[forward_return < down_thr] = 0.0  # DOWN
        direction[(forward_return >= down_thr) & (forward_return <= up_thr)] = 1.0  # NEUTRAL

        return direction

    def _create_volatility_target(self, df: pd.DataFrame) -> pd.Series:
        """Erstellt die Volatilitäts-Zielvariable (LOW/MEDIUM/HIGH).

        Basiert auf der realisierte Volatilität im Vorwärtsfenster.
        Quantil-basierte Einteilung: jede Klasse hat ~33% der Fälle.

        Args:
            df: Feature-DataFrame mit close-Spalte.

        Returns:
            Series mit Werten 0 (LOW), 1 (MEDIUM), 2 (HIGH).
        """
        fwd_window = self._ml_cfg["volatility"]["lookforward_window"]
        low_q = self._ml_cfg["volatility"]["low_quantile"]
        high_q = self._ml_cfg["volatility"]["high_quantile"]

        log_ret = np.log(df["close"] / df["close"].shift(1))

        # Rollierende Std. im Vorwärtsfenster (= zukünftige Volatilität)
        # .shift(-fwd_window) + rolling(fwd_window) = Vola in nächsten fwd_window Tagen
        fwd_vola = log_ret.shift(-fwd_window).rolling(fwd_window).std()

        # Quantile für Klassengrenzen aus der Gesamtverteilung (ohne letztes Fenster)
        valid = fwd_vola.dropna()
        if len(valid) == 0:
            return pd.Series(np.nan, index=df.index)

        low_threshold = valid.quantile(low_q)
        high_threshold = valid.quantile(high_q)

        vola_class = pd.Series(np.nan, index=df.index)
        vola_class[fwd_vola <= low_threshold] = 0.0    # LOW
        vola_class[fwd_vola > high_threshold] = 2.0   # HIGH
        vola_class[
            (fwd_vola > low_threshold) & (fwd_vola <= high_threshold)
        ] = 1.0  # MEDIUM

        return vola_class

    def _select_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Wählt die Feature-Spalten aus (keine Rohpreise, keine Targets).

        Rohpreise (open, high, low, close, volume) werden explizit ausgeschlossen,
        da sie nicht-stationär sind und zu Lookahead-Bias führen können.

        Args:
            df: DataFrame mit allen berechneten Spalten.

        Returns:
            Liste der Feature-Spaltennamen.
        """
        # Explizit ausschließen: alle nicht-stationären absoluten Preisniveaus
        exclude = {
            "open", "high", "low", "close", "volume",
            "bb_upper", "bb_lower", "bb_mid",   # absolute Bollinger-Levels
            "obv_sma",                           # Zwischenwert, obv_trend reicht
        }
        # Absolute EMA-Serien ausschließen (ema_fast_9, ema_mid_21, ema_long_50 etc.)
        # Die ML-Features verwenden stattdessen normalisierte EMA-Cross-Differenzen
        ema_pattern_prefixes = ("ema_fast_", "ema_mid_", "ema_long_")
        cols = [
            col for col in df.columns
            if col not in exclude
            and not col.startswith("_")
            and not any(col.startswith(p) for p in ema_pattern_prefixes)
        ]
        return cols
