"""Streamlit-Hauptdashboard.

Aufbau:
  Sidebar  – Coin-Auswahl (Top-Coins nach Volumen + freie Eingabe), Intervall,
             Zeitraum, Neu-Training, Cache-Verwaltung, Trending Coins
  Kopf     – Pflicht-Hinweis, Kennzahlen (Preis, Market Cap, …), Datenhinweise
  Tabs     – Chart & Signal · Indikatoren · KI-Modell · Backtest · Sentiment ·
             Vergleich · Scanner

Teure Operationen (Daten, Training) laufen im Analyzer mit Disk-Cache; das
Ergebnis wird zusätzlich pro Session zwischengespeichert, damit Klicks auf
Widgets nicht neu rechnen.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.analyzer import AnalysisResult, CryptoAnalyzer
from src.config import DEFAULT_CONFIG_PATH, Timeframe, load_config, setup_logging
from src.models.backtest import run_backtest
from src.ui import components as ui

logger = logging.getLogger(__name__)

_DIRECTION_ORDER = ["BULLISH", "NEUTRAL", "BEARISH"]
_VOLA_ORDER = ["HOCH", "MITTEL", "NIEDRIG"]


# ===========================================================================
# Ressourcen & Caching
# ===========================================================================

@st.cache_resource(show_spinner=False)
def _get_analyzer() -> CryptoAnalyzer:
    """Ein Analyzer pro Server-Prozess (hält HTTP-Session & Caches)."""
    config = load_config(DEFAULT_CONFIG_PATH)
    setup_logging(config, DEFAULT_CONFIG_PATH)
    return CryptoAnalyzer(DEFAULT_CONFIG_PATH)


@st.cache_data(ttl=900, show_spinner=False)
def _universe(limit: int) -> list[str]:
    try:
        return [row["symbol"] for row in _get_analyzer().get_universe(limit)]
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def _trending() -> list[dict[str, Any]]:
    try:
        return _get_analyzer().get_trending_coins()
    except Exception:
        return []


def _run_analysis(symbol: str, interval: str, lookback: int, force: bool, ttl: int) -> AnalysisResult:
    """Analyse mit Session-Cache und Fortschrittsanzeige."""
    cache: dict[tuple, tuple[float, AnalysisResult]] = st.session_state.setdefault("analysis_cache", {})
    key = (symbol, interval, lookback)
    hit = cache.get(key)
    if hit and not force and time.time() - hit[0] < ttl:
        return hit[1]

    with st.status(f"Analysiere {symbol} ({interval})…", expanded=False) as status:
        result = _get_analyzer().analyze(
            symbol, interval, lookback, force_retrain=force, progress=lambda msg: status.update(label=msg)
        )
        state = "error" if result.error else "complete"
        duration = sum(result.timings.values()) if result.timings else 0
        status.update(label=f"{symbol} analysiert in {duration:.1f}s", state=state)
    if not result.error:
        cache[key] = (time.time(), result)
    return result


# ===========================================================================
# Sidebar
# ===========================================================================

def _set_symbol(value: str | None) -> None:
    if value:
        st.session_state["symbol"] = value


def _apply_quick_pick() -> None:
    _set_symbol(st.session_state.get("quick_pick"))
    st.session_state["quick_pick"] = None  # Auswahl zurücksetzen, Selectbox ist die Wahrheit


def _render_sidebar(config: dict[str, Any], analyzer: CryptoAnalyzer) -> tuple[str, str, int, bool]:
    ui_cfg = config["ui"]
    with st.sidebar:
        st.title(f"{ui_cfg['page_icon']} Crypto Analyzer")
        st.caption(f"Version {config['app']['version']} · ML-Engine: {analyzer.engine}")

        popular = ui_cfg["popular_symbols"]
        options = list(dict.fromkeys(popular + _universe(ui_cfg["universe_size"])))
        st.session_state.setdefault("symbol", ui_cfg["default_symbol"])
        if st.session_state["symbol"] not in options:
            options.insert(0, st.session_state["symbol"])

        st.selectbox(
            "Coin",
            options,
            key="symbol",
            accept_new_options=True,
            help="Top-Coins nach Binance-Volumen – oder beliebiges Symbol eintippen (z.B. WIF, BONK).",
        )
        st.pills(
            "Schnellwahl", popular, key="quick_pick", label_visibility="collapsed",
            on_change=_apply_quick_pick,
        )

        interval = st.segmented_control(
            "Intervall", ui_cfg["available_intervals"], default=ui_cfg["default_interval"],
            key="interval", required=True,
            help="1d = Tageskerzen (empfohlen) · 4h/1h = mehr Datenpunkte, kürzere Muster",
        ) or ui_cfg["default_interval"]
        lookback = st.select_slider(
            "Chart-Zeitraum",
            options=[30, 60, 90, 180, 365, 730, 1095, 1460],
            value=ui_cfg["default_lookback_days"],
            format_func=lambda d: f"{d} Tage" if d < 365 else f"{d / 365:g} Jahr(e)",
            help="Nur die Anzeige. Das Modell nutzt automatisch die maximal sinnvolle Historie.",
        )

        st.divider()
        force = st.button("🔄 Modell neu trainieren", width="stretch",
                          help="Erzwingt ein Neutraining, auch wenn das gespeicherte Modell noch frisch ist.")
        col_a, col_b = st.columns(2)
        if col_a.button("🧹 Cache leeren", width="stretch", help="Löscht gespeicherte API-Daten"):
            n = analyzer.clear_cache()
            st.session_state.pop("analysis_cache", None)
            _universe.clear()
            _trending.clear()
            st.toast(f"{n} Cache-Dateien gelöscht.")
        stats = analyzer.cache_stats()
        col_b.caption(f"Cache: {stats['entries']} Dateien · {stats['size_mb']} MB")

        trending = _trending()
        if trending:
            st.divider()
            st.subheader("🔥 Trending")
            available = set(options)
            for coin in trending[:7]:
                sym = coin.get("symbol", "")
                rank = f"#{coin['rank']} · " if coin.get("rank") else ""
                if sym in available:
                    st.button(f"{sym} – {coin.get('name', '')}", key=f"trend_{sym}", width="stretch",
                              on_click=_set_symbol, args=(sym,), type="tertiary")
                else:
                    st.caption(f"{rank}**{sym}** – {coin.get('name', '')} (nicht auf Binance)")

    return str(st.session_state["symbol"]).strip().upper(), interval, int(lookback), force


# ===========================================================================
# Tabs
# ===========================================================================

def _signals_from_oof(result: AnalysisResult, threshold: float) -> pd.DataFrame:
    """Historische Out-of-Sample-Signale über der Schwelle (für Chart-Marker)."""
    oof = result.oof
    if oof is None or oof.empty:
        return pd.DataFrame()
    proba = oof[["p_down", "p_neutral", "p_up"]].to_numpy()
    best = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    labels = np.array(["BEARISH", "NEUTRAL", "BULLISH"])[best]
    mask = (conf >= threshold) & (best != 1)
    return pd.DataFrame({"signal": labels[mask], "confidence": conf[mask], "fwd_ret": oof["fwd_ret"].to_numpy()[mask]},
                        index=oof.index[mask])


def _tab_chart(result: AnalysisResult, config: dict[str, Any]) -> None:
    chart_col, signal_col = st.columns([2.3, 1], gap="large")
    threshold = config["ml"]["confidence_display_threshold"]

    with chart_col:
        c1, c2 = st.columns(2)
        show_signals = c1.toggle("Historische Modellsignale", value=result.oof is not None,
                                 disabled=result.oof is None,
                                 help="Out-of-Sample-Signale aus dem Walk-Forward (▲ bullish, ▼ bearish)")
        log_scale = c2.toggle("Log-Skala", value=False)
        signals = _signals_from_oof(result, threshold) if show_signals else None
        st.plotly_chart(ui.price_chart(result.ohlcv, result.symbol, config["features"], signals, log_scale),
                        width="stretch")

    with signal_col:
        st.subheader("KI-Einschätzung")
        p = result.prediction
        if p is None:
            st.info(f"Keine KI-Analyse möglich: {result.ml_error or 'unbekannter Grund'}")
            return
        ui.render_signal_card(p)
        m = result.eval_metrics
        if m:
            ui.render_verdict_badge(m.verdict, "Richtungsmodell")
        st.markdown("")
        ui.render_probability_bars(p.probabilities, ui.SIGNAL_COLORS, _DIRECTION_ORDER)
        st.caption(
            f"**BULLISH** = Kurs steigt in {p.horizon_text} um mehr als **{p.up_threshold_pct:+.1f}%**, "
            f"**BEARISH** = fällt um mehr als **{abs(p.down_threshold_pct):.1f}%**, sonst NEUTRAL. "
            f"Schwellen passen sich der aktuellen Volatilität an."
        )

        st.divider()
        st.markdown(f"**Erwartete Volatilität ({p.horizon_text})**")
        v = result.volatility_metrics
        if v:
            ui.render_verdict_badge(v.verdict, "Vola-Modell")
            st.markdown("")
        ui.render_probability_bars(p.volatility_probabilities, ui.VOLA_COLORS, _VOLA_ORDER)
        st.caption("Relativ zur Volatilität des Coins im Referenzzeitraum. "
                   "Volatilität ist deutlich besser vorhersagbar als die Kursrichtung.")
        st.caption(f"Basis: letzte abgeschlossene Kerze {p.data_end_date}")


def _tab_indicators(result: AnalysisResult, config: dict[str, Any]) -> None:
    st.plotly_chart(ui.indicator_panel(result.ohlcv, config["features"]), width="stretch")
    st.markdown("**Aktuelle Werte (letzte abgeschlossene Kerze)**")
    st.dataframe(ui.indicator_summary(result.ohlcv, config["features"]), hide_index=True, width="stretch")
    st.caption("Einordnungen sind klassische Faustregeln der technischen Analyse – keine Handlungsempfehlung.")


def _tab_model(result: AnalysisResult) -> None:
    m, v, info = result.eval_metrics, result.volatility_metrics, result.model_info
    if m is None:
        st.info(f"Keine Modell-Evaluierung verfügbar. {result.ml_error or ''}")
        return

    st.markdown(f"#### Out-of-Sample-Bewertung ({m.n_folds} Walk-Forward-Testperioden, {m.n_samples:,} Vorhersagen)")
    st.markdown(m.disclaimer)

    c = st.columns(5)
    c[0].metric("Log-Loss-Skill", ui.fmt_pct(m.skill_score), border=True,
                help="Informationsgewinn gegenüber Raten nach Klassenhäufigkeit. > 0 = Modell weiß mehr als der Zufall.")
    c[1].metric("p-Wert", "n/a" if np.isnan(m.p_value) else f"{m.p_value:.3f}", border=True,
                help="Wahrscheinlichkeit, einen mindestens so guten Skill rein zufällig zu sehen. < 0.05 = signifikant.")
    c[2].metric("Trefferquote", ui.fmt_pct(m.accuracy, 0, signed=False),
                ui.fmt_pct(m.accuracy - m.baseline_accuracy, 1) + " vs. Baseline", border=True,
                help=f"Baseline = immer die häufigste Klasse raten ({m.baseline_accuracy:.0%}).")
    c[3].metric("MCC", f"{m.mcc:+.3f}", border=True,
                help="Matthews-Korrelation: 0 = Zufall, +1 = perfekt. Robust gegenüber Klassen-Ungleichgewicht.")
    if v:
        c[4].metric("Vola-Skill", ui.fmt_pct(v.skill_score), f"p = {v.p_value:.3f}" if not np.isnan(v.p_value) else None,
                    delta_color="off", border=True, help="Log-Loss-Skill des Volatilitätsmodells.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Log-Loss je Testperiode** (niedriger = besser)")
        st.plotly_chart(ui.fold_chart(m), width="stretch")
    with right:
        st.markdown("**Kalibrierung** – stimmen die Wahrscheinlichkeiten?")
        st.plotly_chart(ui.calibration_chart(m.calibration), width="stretch")
        if len(m.calibration) <= 1:
            st.caption("Alle Vorhersagen liegen nahe den reinen Klassenhäufigkeiten – das Modell traut sich "
                       "(zu Recht) keine starken Aussagen zu.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Konfusionsmatrix** (Zeilen-%)")
        st.plotly_chart(ui.confusion_chart(m.confusion, ["BEARISH", "NEUTRAL", "BULLISH"]), width="stretch")
    with right:
        st.markdown("**Was geschah nach Signalen?**")
        table = ui.signal_stats_table(m.signal_stats, m.unconditional_avg_return)
        st.dataframe(table, hide_index=True, width="stretch")
        if len(table) == 1:
            st.caption("Im Testzeitraum lag keine Vorhersage über der Konfidenzschwelle.")
        st.caption(f"Signal = Wahrscheinlichkeit ≥ Konfidenzschwelle. Horizont: {info.get('horizon_text', '')}. "
                   f"„Kurs in Richtung“ = Anteil der Signale, nach denen der Kurs tatsächlich stieg (BULLISH) "
                   f"bzw. fiel (BEARISH).")

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Feature-Wichtigkeit** (Permutation, out-of-sample)")
        if result.feature_importance:
            st.plotly_chart(ui.feature_importance_chart(result.feature_importance), width="stretch")
            st.caption("Blau = Feature verbessert Vorhersagen auf ungesehenen Daten, rot = verschlechtert sie.")
        else:
            st.info("Kein Feature liefert messbaren Out-of-Sample-Mehrwert.")
    with right:
        st.markdown("**Modell-Steckbrief**")
        cal = info.get("calibration")
        dist = info.get("label_distribution", {})
        trained = info.get("trained_at", "")
        facts = {
            "Engine": info.get("engine"),
            "Trainiert": trained.replace("T", " ")[:16] + " UTC" + (" (Cache)" if info.get("from_cache") else ""),
            "Trainingsdaten": f"{info.get('training_start', '')[:10]} → {info.get('training_end', '')[:10]}",
            "Samples / Features": f"{info.get('n_samples', 0):,} / {info.get('n_features', 0)}",
            "Horizont": info.get("horizon_text"),
            "Label-Verteilung": " / ".join(f"{k} {val:.0%}" for k, val in dist.items()),
            "Kalibrierung": f"T = {cal.temperature:.2f}, Modellgewicht = {cal.weight:.0%}" if cal else "–",
            "Trainingsdauer": f"{result.training_time_seconds:.1f}s" if result.training_time_seconds else "aus Cache",
        }
        st.dataframe(pd.DataFrame(facts.items(), columns=["", "Wert"]), hide_index=True, width="stretch")
        st.caption(
            "Modellgewicht < 100 %: Die Kalibrierung mischt die Vorhersage mit den reinen Klassenhäufigkeiten, "
            "weil das Modell out-of-sample nur begrenzt informativ war. 0 % = das Modell weiß nichts."
        )


def _tab_backtest(result: AnalysisResult, config: dict[str, Any]) -> None:
    if result.oof is None or result.oof.empty:
        st.info(f"Kein Backtest möglich. {result.ml_error or ''}")
        return
    st.markdown(
        "Simuliert die Modellsignale auf den **Out-of-Sample-Vorhersagen** des Walk-Forward – "
        "jede Entscheidung basiert nur auf Daten, die zu diesem Zeitpunkt bekannt waren."
    )
    bt_cfg = config["backtest"]
    c = st.columns(4)
    threshold = c[0].slider("Konfidenzschwelle", 0.34, 0.80, float(config["ml"]["confidence_display_threshold"]), 0.01,
                            help="Position nur, wenn die Signalwahrscheinlichkeit mindestens so hoch ist.")
    fee = c[1].number_input("Kosten pro Seite (bps)", 0.0, 100.0, float(bt_cfg["fee_bps"] + bt_cfg["slippage_bps"]), 1.0,
                            help="Gebühren + Slippage in Basispunkten (10 bps = 0,1 %).")
    hold = c[2].number_input("Mindesthaltedauer (Kerzen)", 1, 200, int(bt_cfg["min_hold_bars"]), 1)
    short = c[3].toggle("Short erlauben", value=bool(bt_cfg["allow_short"]))

    bt = run_backtest(result.oof, threshold, result.periods_per_year, fee_bps=fee, slippage_bps=0.0,
                      allow_short=short, min_hold_bars=int(hold))
    if bt is None:
        st.info("Zu wenig Daten für einen Backtest.")
        return

    s, b = bt.strategy, bt.buy_hold
    if s["n_trades"] == 0:
        st.info(
            f"Bei einer Schwelle von {threshold:.0%} gab es im Testzeitraum kein einziges Signal – das Modell war "
            "out-of-sample nie so sicher. Senke die Schwelle, um zu sehen, wie sich schwächere Signale geschlagen hätten."
        )
    k = st.columns(4)
    k[0].metric("Rendite Strategie", ui.fmt_pct(s["total_return"]), ui.fmt_pct(s["total_return"] - b["total_return"]) + " vs. B&H",
                border=True)
    k[1].metric("Sharpe", f"{s['sharpe']:.2f}", f"{s['sharpe'] - b['sharpe']:+.2f} vs. B&H", border=True)
    k[2].metric("Max. Drawdown", ui.fmt_pct(s["max_drawdown"], signed=False),
                ui.fmt_pct(s["max_drawdown"] - b["max_drawdown"]) + " vs. B&H", border=True)
    k[3].metric("Zeit investiert", ui.fmt_pct(s["exposure"], 0, signed=False), f"{s['n_trades']} Trades",
                delta_color="off", border=True)

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(ui.equity_chart(bt, log_scale=st.toggle("Log-Skala", key="bt_log")), width="stretch")
    with right:
        table = ui.backtest_table(bt)
        st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table))
        st.caption(f"Zeitraum: {bt.params['start'][:10]} → {bt.params['end'][:10]}")
    st.caption(
        "⚠️ Backtests überschätzen reale Ergebnisse systematisch (Parameterwahl im Nachhinein, "
        "Liquidität, Ausführung). Ein Backtest ist kein Beleg für zukünftige Gewinne."
    )


def _tab_sentiment(result: AnalysisResult) -> None:
    sent = result.sentiment or {}
    fg = sent.get("fear_greed") or {}
    left, right = st.columns(2)
    with left:
        st.subheader("Fear & Greed Index")
        if fg.get("current_value") is not None:
            st.plotly_chart(ui.fear_greed_gauge(fg["current_value"], fg.get("current_label", "")), width="stretch")
            c = st.columns(3)
            for col, (label, key) in zip(c, (("Δ 1 Tag", "change_1d"), ("Δ 7 Tage", "change_7d"), ("Δ 30 Tage", "change_30d"))):
                val = fg.get(key)
                col.metric(label, f"{val:+.0f}" if val is not None else "–")
            if fg.get("history"):
                st.plotly_chart(ui.fear_greed_history_chart(fg["history"]), width="stretch")
            st.caption("Contrarian-Lesart: Extreme Angst ging historisch oft Erholungen voraus, extreme Gier "
                       "Korrekturen – keine verlässliche Regel. Der Index fließt als zeitversetztes Feature ins Modell ein.")
        else:
            st.info("Fear & Greed Index aktuell nicht verfügbar.")

    with right:
        community = sent.get("community_up_pct")
        if community is not None:
            st.subheader("CoinGecko-Community")
            st.metric("Bullische Stimmen", f"{community:.0f}%", help="Anteil positiver Nutzer-Votes auf CoinGecko (24h)")
            st.progress(min(1.0, community / 100))

        st.subheader(f"Reddit zu {result.symbol}")
        reddit = sent.get("reddit") or {}
        if reddit.get("post_count", 0) > 0:
            c = st.columns(3)
            c[0].metric("Erwähnungen", reddit["post_count"], help=f"von {reddit.get('total_posts_scanned', '?')} Hot-Posts")
            c[1].metric("Netto-Stimmung", f"{reddit['net_sentiment']:+.2f}", help="−1 = nur bärisch, +1 = nur bullisch")
            c[2].metric("Ø Upvotes", f"{reddit['avg_upvotes']:.0f}" if reddit.get("avg_upvotes") is not None else "–")
            st.progress(reddit["bullish_score"], text=f"Bullish {reddit['bullish_score']:.0%}")
            st.progress(reddit["bearish_score"], text=f"Bearish {reddit['bearish_score']:.0%}")
            for title in reddit.get("top_titles", []):
                st.markdown(f"- {ui.escape_md(title)}")
            st.caption(f"Quelle: {reddit.get('source')} · "
                       f"{', '.join('r/' + s for s in reddit.get('subreddits_checked', []))} · Keyword-Analyse, kein NLP.")
        elif reddit.get("error"):
            st.info(f"Reddit aktuell nicht erreichbar ({reddit['error']}).")
        else:
            st.info(f"Keine aktuellen Hot-Posts erwähnen {result.symbol}.")


def _tab_compare(result: AnalysisResult, config: dict[str, Any], interval: str) -> None:
    analyzer = _get_analyzer()
    options = list(dict.fromkeys([result.symbol] + config["ui"]["popular_symbols"] + _universe(config["ui"]["universe_size"])))
    defaults = list(dict.fromkeys([result.symbol, "BTC", "ETH"]))
    c1, c2 = st.columns([3, 1])
    symbols = c1.multiselect("Coins", options, default=defaults, max_selections=8, accept_new_options=True)
    days = c2.select_slider("Zeitraum", [30, 90, 180, 365, 730], value=180, key="cmp_days",
                            format_func=lambda d: f"{d} Tage")
    if len(symbols) < 2:
        st.info("Mindestens zwei Coins auswählen.")
        return

    key = (tuple(symbols), interval, days)
    cache = st.session_state.setdefault("compare_cache", {})
    if key not in cache:
        with st.spinner("Lade Vergleichsdaten…"):
            cache[key] = analyzer.compare(symbols, interval, days)
    cmp = cache[key]
    for sym, err in cmp.errors.items():
        st.warning(f"{sym}: {err}")
    if cmp.normalized.empty:
        return

    st.plotly_chart(ui.comparison_chart(cmp.normalized), width="stretch")
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            cmp.stats, width="stretch",
            column_config={
                col: st.column_config.NumberColumn(format="percent")
                for col in ("Return", "Volatilität (ann.)", "Max. Drawdown", "Abstand zum Hoch")
            } | {c: st.column_config.NumberColumn(format="%.2f") for c in cmp.stats.columns if c.startswith(("Sharpe", "Beta"))},
        )
    with right:
        st.plotly_chart(ui.correlation_heatmap(cmp.correlation), width="stretch")
    st.caption("Korrelation der Log-Returns: Werte nahe 1 bedeuten, dass die Coins fast im Gleichschritt laufen "
               "(wenig Diversifikation).")


def _tab_scanner(config: dict[str, Any], interval: str) -> None:
    st.markdown(
        "Analysiert die liquidesten Coins mit demselben Modell-Setup und zeigt Signale **zusammen mit der "
        "gemessenen Modellqualität** – ein Signal ohne signifikanten Skill ist nur eine Indikator-Einordnung."
    )
    c1, c2 = st.columns([1, 3])
    n = c1.number_input("Anzahl Coins", 3, 40, int(config["ui"]["scanner_size"]), 1)
    run = c2.button(f"🔭 Top {n} ({interval}) scannen", type="primary")

    key = (interval, int(n))
    cache = st.session_state.setdefault("scan_cache", {})
    if run:
        symbols = _universe(int(n)) or config["ui"]["popular_symbols"][: int(n)]
        bar = st.progress(0.0, text="Starte Scan…")

        def progress(done: int, total: int, sym: str) -> None:
            bar.progress(done / total, text=f"{done}/{total} · {sym} fertig")

        started = time.time()
        cache[key] = (_get_analyzer().scan(symbols, interval, progress=progress), time.time())
        bar.progress(1.0, text=f"Fertig in {time.time() - started:.0f}s")

    if key not in cache:
        st.caption("Erster Scan trainiert für jeden Coin ein Modell (≈ 2–5 s pro Coin), danach aus dem Cache.")
        return
    table, ts = cache[key]
    st.caption(f"Stand: {datetime.fromtimestamp(ts, tz=UTC):%Y-%m-%d %H:%M} UTC")
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Kurs": st.column_config.NumberColumn(format="%.6g"),
            "7d %": st.column_config.NumberColumn(format="%+.1f%%"),
            "P(UP)": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent"),
            "P(DOWN)": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent"),
            "OOS-Skill": st.column_config.NumberColumn(format="percent"),
            "Vola-Skill": st.column_config.NumberColumn(format="percent"),
            "p-Wert": st.column_config.NumberColumn(format="%.3f"),
            "Backtest Sharpe": st.column_config.NumberColumn(format="%.2f"),
        },
    )


# ===========================================================================
# Hauptfunktion
# ===========================================================================

def render_dashboard() -> None:
    """Haupt-Render-Funktion des Dashboards."""
    config = load_config(DEFAULT_CONFIG_PATH)
    ui_cfg = config["ui"]
    st.set_page_config(page_title=ui_cfg["page_title"], page_icon=ui_cfg["page_icon"],
                       layout=ui_cfg["layout"], initial_sidebar_state="expanded")

    analyzer = _get_analyzer()
    symbol, interval, lookback, force = _render_sidebar(config, analyzer)
    ui.render_disclaimer(ui_cfg["disclaimer_short"])

    if not symbol:
        st.info("Wähle links einen Coin aus.")
        return

    result = _run_analysis(symbol, interval, lookback, force, ui_cfg["analysis_ttl_seconds"])
    if result.error:
        st.error(f"**Analyse von {symbol} fehlgeschlagen**\n\n{result.error}")
        st.info("💡 Symbol ohne USDT eingeben (BTC, nicht BTCUSDT) · Internetverbindung prüfen · "
                "bei Rate-Limits kurz warten.")
        return

    name = result.market_data.get("name") or result.symbol
    st.title(f"{name} ({result.symbol})")
    tf = Timeframe(result.interval)
    ui.render_market_header(result.market_data, result.ohlcv["close"], tf.bars_per_day)

    notes = []
    if result.data_freshness_minutes is not None:
        notes.append(f"Kursdaten vor {result.data_freshness_minutes:.0f} Min. aktualisiert")
    if result.market_data.get("stale_minutes"):
        notes.append(f"Marktdaten von vor {result.market_data['stale_minutes']:.0f} Min. (CoinGecko-Limit)")
    if result.model_info:
        notes.append("Modell aus Cache" if result.model_info.get("from_cache")
                     else f"Modell neu trainiert in {result.training_time_seconds:.1f}s")
    st.caption(" · ".join(notes))
    if result.ml_error:
        st.warning(f"**KI-Analyse nicht möglich:** {result.ml_error}")
    if result.warnings:
        with st.expander(f"ℹ️ {len(result.warnings)} Datenhinweise"):
            for w in result.warnings:
                st.caption(f"• {w}")

    tabs = st.tabs(["📈 Chart & Signal", "🧮 Indikatoren", "🤖 KI-Modell", "💰 Backtest",
                    "💭 Sentiment", "⚖️ Vergleich", "🔭 Scanner"])
    with tabs[0]:
        _tab_chart(result, config)
    with tabs[1]:
        _tab_indicators(result, config)
    with tabs[2]:
        _tab_model(result)
    with tabs[3]:
        _tab_backtest(result, config)
    with tabs[4]:
        _tab_sentiment(result)
    with tabs[5]:
        _tab_compare(result, config, interval)
    with tabs[6]:
        _tab_scanner(config, interval)

    st.divider()
    st.info(ui_cfg["disclaimer_long"], icon="⚠️")
