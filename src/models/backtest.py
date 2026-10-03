"""Vektorisierter Backtest der Modellsignale auf Out-of-Sample-Vorhersagen.

Regeln (bewusst einfach und ohne Lookahead):
  * Die Vorhersage für Kerze t entsteht aus Daten bis zum Schluss von t.
  * Position wird zum Schlusskurs von t eingenommen und hält über Kerze t+1.
  * Long, wenn BULLISH die wahrscheinlichste Klasse ist und P(UP) ≥ Schwelle;
    Short (optional), wenn BEARISH analog; sonst flat.
  * Gebühren + Slippage fallen bei jeder Positionsänderung an (pro Seite).
  * Optionale Mindesthaltedauer reduziert Hin-und-Her-Handeln.

Verglichen wird mit Buy & Hold über exakt denselben Zeitraum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Ergebnis eines Backtests.

    Attributes:
        equity: Equity-Kurven (Start = 1.0), Spalten "Strategie" und "Buy & Hold".
        drawdown: Drawdown-Kurven (≤ 0) der beiden Equity-Kurven.
        position: Gehaltene Position je Kerze (−1, 0, +1).
        strategy: Kennzahlen der Strategie.
        buy_hold: Kennzahlen von Buy & Hold.
        params: Verwendete Parameter.
    """

    equity: pd.DataFrame
    drawdown: pd.DataFrame
    position: pd.Series
    strategy: dict[str, float]
    buy_hold: dict[str, float]
    params: dict[str, Any]


def generate_positions(
    oof: pd.DataFrame,
    threshold: float,
    allow_short: bool = False,
    min_hold_bars: int = 1,
) -> pd.Series:
    """Leitet die Zielposition je Kerze aus den OOF-Wahrscheinlichkeiten ab."""
    proba = oof[["p_down", "p_neutral", "p_up"]].to_numpy(dtype=float)
    best = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    raw = np.zeros(len(oof))
    raw[(best == 2) & (conf >= threshold)] = 1.0
    if allow_short:
        raw[(best == 0) & (conf >= threshold)] = -1.0

    if min_hold_bars > 1:
        held = np.zeros_like(raw)
        current, age = 0.0, 0
        for i, target in enumerate(raw):
            if current != 0.0 and age < min_hold_bars:
                age += 1
            elif target != current:
                current, age = target, 1
            else:
                age += 1
            held[i] = current
        raw = held
    return pd.Series(raw, index=oof.index, name="position")


def run_backtest(
    oof: pd.DataFrame,
    threshold: float,
    periods_per_year: float,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    allow_short: bool = False,
    min_hold_bars: int = 1,
) -> BacktestResult | None:
    """Simuliert die Signalstrategie gegen Buy & Hold.

    Args:
        oof: OOF-Vorhersagen mit p_down/p_neutral/p_up und next_ret.
        threshold: Mindest-Wahrscheinlichkeit für eine Position.
        periods_per_year: Kerzen pro Jahr (für Annualisierung).
        fee_bps: Handelsgebühr pro Seite in Basispunkten (10 = 0,1 %).
        slippage_bps: Geschätzte Slippage pro Seite in Basispunkten.
        allow_short: Short-Positionen bei BEARISH-Signalen erlauben.
        min_hold_bars: Mindesthaltedauer in Kerzen.

    Returns:
        BacktestResult oder None bei zu wenig Daten.
    """
    if oof is None or len(oof) < 2:
        return None

    next_ret = oof["next_ret"].astype(float).fillna(0.0)
    position = generate_positions(oof, threshold, allow_short, min_hold_bars)
    cost_rate = (fee_bps + slippage_bps) / 10_000
    turnover = position.diff().abs().fillna(position.abs())
    strat_ret = position * next_ret - turnover * cost_rate

    equity = pd.DataFrame({
        "Strategie": (1.0 + strat_ret).cumprod(),
        "Buy & Hold": (1.0 + next_ret).cumprod(),
    })
    drawdown = equity / equity.cummax() - 1.0

    strategy = performance_stats(strat_ret, periods_per_year)
    strategy.update(_trade_stats(position, position * next_ret, cost_rate))
    strategy["exposure"] = float((position != 0).mean())
    strategy["costs_paid"] = float((turnover * cost_rate).sum())

    return BacktestResult(
        equity=equity,
        drawdown=drawdown,
        position=position,
        strategy=strategy,
        buy_hold=performance_stats(next_ret, periods_per_year),
        params={
            "threshold": threshold,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "allow_short": allow_short,
            "min_hold_bars": min_hold_bars,
            "start": str(oof.index[0]),
            "end": str(oof.index[-1]),
        },
    )


def performance_stats(returns: pd.Series, periods_per_year: float) -> dict[str, float]:
    """Standard-Performancekennzahlen einer Return-Reihe (pro Kerze)."""
    r = returns.astype(float).fillna(0.0)
    n = len(r)
    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0) if n else 0.0
    years = n / periods_per_year if periods_per_year else 0.0
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1.0 else float("nan")
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))) if n else 0.0
    ann = np.sqrt(periods_per_year)
    max_dd = float((equity / equity.cummax() - 1.0).min()) if n else 0.0
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_volatility": std * ann,
        "sharpe": float(r.mean() / std * ann) if std > 0 else 0.0,
        "sortino": float(r.mean() / downside * ann) if downside > 0 else 0.0,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 and not np.isnan(cagr) else float("nan"),
    }


def _trade_stats(position: pd.Series, gross_ret: pd.Series, cost_rate: float) -> dict[str, float]:
    """Anzahl Trades, Trefferquote und Ø-Rendite je Trade (Trade = zusammenhängende Position).

    Ein- und Ausstiegskosten werden dem jeweiligen Trade zugerechnet; ein am
    Ende noch offener Trade wird ohne Ausstiegskosten bewertet.
    """
    trades: list[float] = []
    current, growth = 0.0, 1.0
    for p, g in zip(position.to_numpy(), gross_ret.to_numpy()):
        if p != current:
            if current != 0.0:
                trades.append(growth * (1.0 - cost_rate) - 1.0)
            current = p
            growth = 1.0 - cost_rate if p != 0.0 else 1.0
        if p != 0.0:
            growth *= 1.0 + g
    if current != 0.0:
        trades.append(growth - 1.0)
    if not trades:
        return {"n_trades": 0, "win_rate": float("nan"), "avg_trade": float("nan")}
    arr = np.array(trades)
    return {"n_trades": len(arr), "win_rate": float(np.mean(arr > 0)), "avg_trade": float(arr.mean())}
