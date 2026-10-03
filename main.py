"""Einstiegspunkt für den Crypto Analyzer.

Dashboard:
    streamlit run main.py          (oder einfach: python main.py)

Kommandozeile:
    python main.py check                       Abhängigkeiten & API-Erreichbarkeit prüfen
    python main.py analyze BTC                 Analyse im Terminal (--interval 4h --retrain --json)
    python main.py scan --top 10               Signal-Scanner über die liquidesten Coins
    python main.py clear-cache                 API-Cache löschen
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _check_python_version() -> None:
    if sys.version_info < (3, 11):  # noqa: UP036 – Absicherung für Aufrufe mit altem Python
        print(
            f"FEHLER: Python 3.11+ erforderlich (installiert: {sys.version_info.major}.{sys.version_info.minor}).\n"
            f"Download: https://python.org/downloads"
        )
        sys.exit(1)


def _utf8_console() -> None:
    """Windows-Konsolen (cp1252) können keine Emojis – auf UTF-8 umstellen."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _running_in_streamlit() -> bool:
    """True, wenn das Skript von Streamlit ausgeführt wird (auch in AppTest)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


# ===========================================================================
# CLI-Befehle
# ===========================================================================

def cmd_check(_: argparse.Namespace) -> int:
    """Prüft Pakete und API-Erreichbarkeit."""
    print("Crypto Analyzer – Systemcheck")
    print("=" * 48)
    required = [
        ("streamlit", "streamlit"), ("pandas", "pandas"), ("numpy", "numpy"), ("scikit-learn", "sklearn"),
        ("scipy", "scipy"), ("plotly", "plotly"), ("joblib", "joblib"), ("PyYAML", "yaml"), ("requests", "requests"),
    ]
    ok = True
    for label, module in required:
        try:
            mod = importlib.import_module(module)
            print(f"  ✅ {label:<14} {getattr(mod, '__version__', '')}")
        except ImportError as exc:
            print(f"  ❌ {label:<14} {exc}")
            ok = False
    try:
        import lightgbm

        print(f"  ✅ {'lightgbm':<14} {lightgbm.__version__} (optional, schnellere Engine)")
    except ImportError:
        print(f"  ➖ {'lightgbm':<14} nicht installiert (optional – HistGradientBoosting wird genutzt)")

    if not ok:
        print("=" * 48)
        print("Fehlende Pakete installieren: pip install -r requirements.txt")
        return 1

    import requests

    from src.config import load_config

    config = load_config()
    print("\nAPI-Erreichbarkeit")
    endpoints = [
        ("Binance", config["api"]["binance"]["base_url"] + "/api/v3/ping"),
        ("Binance Mirror", config["api"]["binance"]["fallback_base_urls"][0] + "/api/v3/ping"),
        ("CoinGecko", config["api"]["coingecko"]["base_url"] + "/ping"),
        ("Fear & Greed", config["api"]["alternative_me"]["base_url"] + "?limit=1"),
    ]
    for label, url in endpoints:
        try:
            status = requests.get(url, timeout=8, headers={"User-Agent": config["api"]["user_agent"]}).status_code
            icon = "✅" if status == 200 else "⚠️"
            print(f"  {icon} {label:<14} HTTP {status}")
        except requests.RequestException as exc:
            print(f"  ❌ {label:<14} {type(exc).__name__}")
    print("=" * 48)
    print("Alles bereit. Start: streamlit run main.py")
    return 0


def _fmt_pct(value: float | None, ratio: bool = True) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value * (100 if ratio else 1):+.2f}%"


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyse eines Coins im Terminal."""
    from src.analysis.analyzer import CryptoAnalyzer

    analyzer = CryptoAnalyzer()
    result = analyzer.analyze(
        args.symbol, args.interval, args.days, force_retrain=args.retrain,
        progress=None if args.json else (lambda msg: print(f"  … {msg}")),
    )
    if result.error:
        print(f"FEHLER: {result.error}", file=sys.stderr)
        return 1

    p, m, v, bt = result.prediction, result.eval_metrics, result.volatility_metrics, result.backtest
    if args.json:
        payload: dict[str, Any] = {
            "symbol": result.symbol,
            "interval": result.interval,
            "price": result.market_data.get("price"),
            "ml_error": result.ml_error,
            "prediction": None if p is None else {
                "signal": p.direction_label if p.show_signal else None,
                "probabilities": p.probabilities,
                "volatility": p.volatility_label,
                "volatility_probabilities": p.volatility_probabilities,
                "horizon": p.horizon_text,
                "thresholds_pct": [p.down_threshold_pct, p.up_threshold_pct],
            },
            "model": None if m is None else {
                "verdict": m.verdict, "skill": m.skill_score, "p_value": m.p_value, "accuracy": m.accuracy,
                "baseline_accuracy": m.baseline_accuracy, "mcc": m.mcc, "folds": m.n_folds,
                "volatility_skill": v.skill_score if v else None, "volatility_verdict": v.verdict if v else None,
            },
            "backtest": None if bt is None else {"strategy": bt.strategy, "buy_hold": bt.buy_hold},
            "timings": result.timings,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=lambda o: None if o != o else str(o)))
        return 0

    md = result.market_data
    print()
    print(f"═══ {md.get('name', result.symbol)} ({result.symbol}) · {result.interval} ═══")
    print(f"Preis: {md.get('price', '–')}  ·  24h: {_fmt_pct(md.get('price_change_24h_pct'), ratio=False)}"
          f"  ·  Rang: #{md.get('rank') or '–'}")
    if result.ml_error:
        print(f"\nKI-Analyse nicht möglich: {result.ml_error}")
        return 0
    print(f"\nSignal ({p.horizon_text}): "
          + (f"{p.direction_emoji} {p.direction_label} ({p.confidence:.0%})" if p.show_signal else "⚪ kein klares Signal"))
    print("  Wahrscheinlichkeiten: " + ", ".join(f"{k} {val:.0%}" for k, val in p.probabilities.items()))
    print(f"  Schwellen: BULLISH > {p.up_threshold_pct:+.2f}%, BEARISH < {p.down_threshold_pct:+.2f}%")
    print(f"  Volatilität: {p.volatility_label} (" + ", ".join(f"{k} {val:.0%}" for k, val in p.volatility_probabilities.items()) + ")")
    if m:
        print(f"\nModellqualität (out-of-sample, {m.n_folds} Folds, {m.n_samples} Vorhersagen)")
        print(f"  Richtung:   {m.verdict:<22} Skill {m.skill_score:+.1%}  p={m.p_value:.3f}  "
              f"Treffer {m.accuracy:.0%} vs. {m.baseline_accuracy:.0%}  MCC {m.mcc:+.3f}")
        if v:
            print(f"  Volatilität: {v.verdict:<21} Skill {v.skill_score:+.1%}  p={v.p_value:.3f}  MCC {v.mcc:+.3f}")
    if bt:
        s, b = bt.strategy, bt.buy_hold
        print(f"\nBacktest {bt.params['start'][:10]} → {bt.params['end'][:10]} (nach Kosten)")
        print(f"  Strategie:  Rendite {_fmt_pct(s['total_return'])}  Sharpe {s['sharpe']:.2f}  "
              f"MaxDD {_fmt_pct(s['max_drawdown'])}  investiert {s['exposure']:.0%}  Trades {s['n_trades']}")
        print(f"  Buy & Hold: Rendite {_fmt_pct(b['total_return'])}  Sharpe {b['sharpe']:.2f}  MaxDD {_fmt_pct(b['max_drawdown'])}")
    print(f"\nLaufzeit: {sum(result.timings.values()):.1f}s {result.timings}")
    print("⚠️  Keine Finanzberatung.")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Signal-Scanner über die liquidesten Coins."""
    import pandas as pd

    from src.analysis.analyzer import CryptoAnalyzer

    analyzer = CryptoAnalyzer()
    symbols = [s.upper() for s in args.symbols] if args.symbols else [r["symbol"] for r in analyzer.get_universe(args.top)]
    print(f"Scanne {len(symbols)} Coins ({args.interval})…")
    table = analyzer.scan(symbols, args.interval, progress=lambda d, t, s: print(f"  {d}/{t} {s}"))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(table.drop(columns=["Hinweis"], errors="ignore").to_string(index=False))
    return 0


def cmd_clear_cache(_: argparse.Namespace) -> int:
    from src.analysis.analyzer import CryptoAnalyzer

    print(f"{CryptoAnalyzer().clear_cache()} Cache-Dateien gelöscht.")
    return 0


def cmd_ui(_: argparse.Namespace) -> int:
    """Startet das Streamlit-Dashboard."""
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crypto-analyzer", description="KI-gestützte Krypto-Analyse (nur Bildungszwecke).")
    parser.add_argument("--check", action="store_true", help=argparse.SUPPRESS)  # Alias aus v1
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("ui", help="Dashboard starten (Standard)").set_defaults(func=cmd_ui)
    sub.add_parser("check", help="Abhängigkeiten und APIs prüfen").set_defaults(func=cmd_check)

    p_an = sub.add_parser("analyze", help="Einen Coin analysieren")
    p_an.add_argument("symbol", help="z.B. BTC, ETH, PEPE")
    p_an.add_argument("--interval", "-i", default="1d", choices=["1h", "4h", "1d"])
    p_an.add_argument("--days", "-d", type=int, default=365, help="Chart-Zeitraum in Tagen")
    p_an.add_argument("--retrain", action="store_true", help="Modell neu trainieren")
    p_an.add_argument("--json", action="store_true", help="Ausgabe als JSON")
    p_an.set_defaults(func=cmd_analyze)

    p_sc = sub.add_parser("scan", help="Mehrere Coins scannen")
    p_sc.add_argument("--top", type=int, default=10, help="Top-N Coins nach Volumen")
    p_sc.add_argument("--interval", "-i", default="1d", choices=["1h", "4h", "1d"])
    p_sc.add_argument("symbols", nargs="*", help="Optional: eigene Symbolliste")
    p_sc.set_defaults(func=cmd_scan)

    sub.add_parser("clear-cache", help="API-Cache löschen").set_defaults(func=cmd_clear_cache)
    return parser


def main(argv: list[str] | None = None) -> int:
    _check_python_version()
    _utf8_console()
    args = build_parser().parse_args(argv)
    if args.check:
        return cmd_check(args)
    if not getattr(args, "func", None):
        return cmd_ui(args)

    import logging

    from src.config import load_config, setup_logging

    setup_logging(load_config())
    logging.getLogger().handlers[0].setLevel(logging.WARNING)  # Konsole ruhig halten
    return args.func(args)


if _running_in_streamlit():
    from src.ui.dashboard import render_dashboard

    render_dashboard()
elif __name__ == "__main__":
    sys.exit(main())
