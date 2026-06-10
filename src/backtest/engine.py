"""Walk-Forward Backtesting Engine.

Simuliert eine einfache Long-Only-Strategie basierend auf den
Walk-Forward-Signalen des ML-Modells:
  - Signal BULLISH (2)  → Long-Position eingehen / halten
  - Signal BEARISH (0)  → Position schließen (Cash)
  - Signal NEUTRAL (1)  → aktuelle Position halten

Kein Lookahead-Bias: Die Signale stammen aus Walk-Forward-Folds,
bei denen das Modell ausschließlich auf Vergangenheitsdaten trainiert wurde.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Vollständiges Backtest-Ergebnis.

    Attributes:
        series: DataFrame mit Spalten portfolio, benchmark, signal, position.
        total_return_pct: Gesamt-Rendite der Strategie in %.
        bnh_return_pct: Buy-and-Hold Rendite in %.
        alpha_pct: Strategie minus Buy-and-Hold in Prozentpunkten.
        max_drawdown_pct: Maximaler Drawdown in % (negativ).
        sharpe_ratio: Annualisiertes Sharpe-Ratio (vereinfacht).
        n_days: Anzahl Handelstage im Backtest-Zeitraum.
        n_long_days: Tage mit aktiver Long-Position.
        exposure_pct: Markt-Exposure (Tage long / Gesamttage).
        n_trades: Anzahl Positions-Wechsel.
        win_rate_pct: Anteil profitabler Long-Perioden in %.
        period_label: Menschenlesbarer Zeitraum (z.B. "Jan 2024 – Jan 2025").
    """

    series: pd.DataFrame
    total_return_pct: float
    bnh_return_pct: float
    alpha_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    n_days: int
    n_long_days: int
    exposure_pct: float
    n_trades: int
    win_rate_pct: float
    period_label: str


class BacktestEngine:
    """Führt einen vereinfachten Walk-Forward-Backtest durch.

    Args:
        config: Geladenes config.yaml als Dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        bt_cfg = config.get("backtest", {})
        self._initial_capital = float(bt_cfg.get("initial_capital", 10_000))
        self._tx_cost = float(bt_cfg.get("transaction_cost_pct", 0.001))

    def run(
        self,
        ohlcv: pd.DataFrame,
        fold_results: list[dict[str, Any]],
    ) -> BacktestResult | None:
        """Simuliert die Strategie über alle Walk-Forward-Testperioden.

        Args:
            ohlcv: Bereinigtes OHLCV-DataFrame mit DatetimeIndex.
            fold_results: Fold-Dicts aus dem Trainer (y_pred + test_dates).

        Returns:
            BacktestResult oder None wenn zu wenig Daten vorhanden.
        """
        pred_df = self._build_prediction_series(fold_results)
        if pred_df is None or len(pred_df) < 5:
            logger.warning("Zu wenig Walk-Forward-Daten für Backtest.")
            return None

        close = ohlcv["close"].copy()
        aligned = self._align_with_prices(pred_df, close)
        if aligned is None or len(aligned) < 5:
            return None

        simulated = self._simulate(aligned)
        metrics = self._compute_metrics(simulated)
        return metrics

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_prediction_series(
        self, fold_results: list[dict[str, Any]]
    ) -> pd.DataFrame | None:
        """Sammelt alle Fold-Vorhersagen in einem DataFrame."""
        records = []
        for fold in fold_results:
            dates = fold.get("test_dates", [])
            preds = fold.get("y_pred", [])
            if not dates or not preds:
                continue
            for d, p in zip(dates, preds):
                records.append({"date": pd.Timestamp(d), "pred": int(p)})

        if not records:
            return None

        df = (
            pd.DataFrame(records)
            .drop_duplicates("date")
            .sort_values("date")
            .set_index("date")
        )
        return df

    def _align_with_prices(
        self, pred_df: pd.DataFrame, close: pd.Series
    ) -> pd.DataFrame | None:
        """Verknüpft Vorhersagen mit Close-Preisen."""
        # Nearest-Merge: Falls Datum nicht exakt vorhanden (z.B. Feiertag), nächsten nehmen
        merged = pred_df.copy()
        merged["close"] = close.reindex(pred_df.index, method="nearest")
        merged = merged.dropna(subset=["close"])

        if len(merged) < 5:
            return None
        return merged

    def _simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Führt die Positions-Simulation durch."""
        df = df.copy()

        # Positions-Zustand
        position = 0
        positions = []
        for sig in df["pred"]:
            if sig == 2:   # BULLISH → Long
                position = 1
            elif sig == 0: # BEARISH → Cash
                position = 0
            # NEUTRAL → halten
            positions.append(position)

        df["position"] = positions

        # Tagesrenditen
        df["daily_ret"] = df["close"].pct_change().fillna(0)

        # Strategie-Rendite: Position vom VORTAG (kein Lookahead)
        df["strategy_ret"] = df["position"].shift(1).fillna(0) * df["daily_ret"]

        # Transaktionskosten auf Positions-Wechsel
        position_changes = df["position"].diff().abs().fillna(0) > 0
        df.loc[position_changes, "strategy_ret"] -= self._tx_cost

        # Kumulatives Portfolio-Wachstum (normiert auf 100)
        df["portfolio"] = 100.0 * (1 + df["strategy_ret"]).cumprod()
        df["benchmark"] = 100.0 * (1 + df["daily_ret"]).cumprod()

        return df

    def _compute_metrics(self, df: pd.DataFrame) -> BacktestResult:
        """Berechnet alle Backtest-Metriken."""
        port = df["portfolio"]
        bnh = df["benchmark"]

        total_return = (port.iloc[-1] / 100 - 1) * 100
        bnh_return = (bnh.iloc[-1] / 100 - 1) * 100
        alpha = total_return - bnh_return

        # Max Drawdown
        peak = port.cummax()
        drawdown = (port - peak) / peak
        max_dd = float(drawdown.min()) * 100

        # Sharpe (annualisiert, vereinfacht)
        rets = df["strategy_ret"]
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(252)) if rets.std() > 0 else 0.0

        # Exposure
        n_days = len(df)
        n_long = int(df["position"].sum())
        exposure = n_long / n_days * 100 if n_days > 0 else 0.0

        # Trades zählen
        n_trades = int((df["position"].diff().abs() > 0).sum())

        # Win Rate: Anteil Long-Perioden mit positiver Rendite
        long_rets = df.loc[df["position"].shift(1).fillna(0) == 1, "strategy_ret"]
        if len(long_rets) > 0:
            win_rate = float((long_rets > 0).sum() / len(long_rets)) * 100
        else:
            win_rate = 0.0

        # Zeitraum
        try:
            start_str = df.index[0].strftime("%b %Y")
            end_str = df.index[-1].strftime("%b %Y")
            period_label = f"{start_str} – {end_str}"
        except Exception:
            period_label = "N/A"

        return BacktestResult(
            series=df[["portfolio", "benchmark", "pred", "position"]],
            total_return_pct=round(total_return, 2),
            bnh_return_pct=round(bnh_return, 2),
            alpha_pct=round(alpha, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            n_days=n_days,
            n_long_days=n_long,
            exposure_pct=round(exposure, 1),
            n_trades=n_trades,
            win_rate_pct=round(win_rate, 1),
            period_label=period_label,
        )
