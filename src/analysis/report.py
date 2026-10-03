"""Text-Zusammenfassung einer Analyse – gemeinsam genutzt von CLI und Colab-Notebook."""

from __future__ import annotations

import math

from src.analysis.analyzer import AnalysisResult


def _fmt_pct(value: float | None, ratio: bool = True, signed: bool = True) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    v = value * (100 if ratio else 1)
    return f"{v:+.2f}%" if signed else f"{v:.2f}%"


def format_summary(result: AnalysisResult) -> str:
    """Mehrzeilige Zusammenfassung: Kurs, Signal, Modellqualität, Backtest."""
    if result.error:
        return f"FEHLER: {result.error}"

    p, m, v, bt = result.prediction, result.eval_metrics, result.volatility_metrics, result.backtest
    md = result.market_data
    lines = [
        f"═══ {md.get('name', result.symbol)} ({result.symbol}) · {result.interval} ═══",
        f"Preis: {md.get('price', '–')}  ·  24h: {_fmt_pct(md.get('price_change_24h_pct'), ratio=False)}"
        f"  ·  Rang: #{md.get('rank') or '–'}",
    ]
    if result.ml_error or p is None:
        lines += ["", f"KI-Analyse nicht möglich: {result.ml_error}"]
        return "\n".join(lines)

    signal = f"{p.direction_emoji} {p.direction_label} ({p.confidence:.0%})" if p.show_signal else "⚪ kein klares Signal"
    lines += [
        "",
        f"Signal ({p.horizon_text}): {signal}",
        "  Wahrscheinlichkeiten: " + ", ".join(f"{k} {val:.0%}" for k, val in p.probabilities.items()),
        f"  Schwellen: BULLISH > {p.up_threshold_pct:+.2f}%, BEARISH < {p.down_threshold_pct:+.2f}%",
        f"  Volatilität: {p.volatility_label} ("
        + ", ".join(f"{k} {val:.0%}" for k, val in p.volatility_probabilities.items()) + ")",
    ]
    if m:
        lines += [
            "",
            f"Modellqualität (out-of-sample, {m.n_folds} Folds, {m.n_samples} Vorhersagen)",
            f"  Richtung:    {m.verdict:<22} Skill {m.skill_score:+.1%}  p={m.p_value:.3f}  "
            f"Treffer {m.accuracy:.0%} vs. {m.baseline_accuracy:.0%}  MCC {m.mcc:+.3f}",
        ]
        if v:
            lines.append(f"  Volatilität: {v.verdict:<22} Skill {v.skill_score:+.1%}  p={v.p_value:.3f}  MCC {v.mcc:+.3f}")
    if bt:
        s, b = bt.strategy, bt.buy_hold
        lines += [
            "",
            f"Backtest {bt.params['start'][:10]} → {bt.params['end'][:10]} (nach Kosten)",
            f"  Strategie:  Rendite {_fmt_pct(s['total_return'])}  Sharpe {s['sharpe']:.2f}  "
            f"MaxDD {_fmt_pct(s['max_drawdown'], signed=False)}  investiert {s['exposure']:.0%}  Trades {s['n_trades']}",
            f"  Buy & Hold: Rendite {_fmt_pct(b['total_return'])}  Sharpe {b['sharpe']:.2f}  "
            f"MaxDD {_fmt_pct(b['max_drawdown'], signed=False)}",
        ]
    lines += ["", f"Laufzeit: {sum(result.timings.values()):.1f}s {result.timings}", "⚠️  Keine Finanzberatung."]
    return "\n".join(lines)
