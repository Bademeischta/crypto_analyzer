"""Streamlit-Hauptdashboard v2.0 – Tab-basiertes Layout.

Tabs:
  1. 📊 Analyse   – Chart + KI-Signal + Indikatoren
  2. 🔄 Backtest  – Portfolio-Simulation auf Walk-Forward-Signalen
  3. 🔍 Screener  – Mehrere Coins auf einen Blick
  4. 🤖 Modell    – Feature Importance + Eval-Metriken
  5. 💭 Sentiment – Fear & Greed + Reddit

Alle teuren Operationen werden in st.session_state gecacht.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.analyzer import CryptoAnalyzer, AnalysisResult
from src.ui.components import (
    render_backtest_chart,
    render_backtest_metrics,
    render_disclaimer,
    render_fear_greed_gauge,
    render_feature_importance_chart,
    render_macd_chart,
    render_market_data_row,
    render_model_performance_badge,
    render_price_chart,
    render_probability_bars,
    render_rsi_chart,
    render_screener_table,
    render_signal_card,
    render_volatility_badge,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@st.cache_resource(show_spinner=False)
def _get_analyzer() -> CryptoAnalyzer:
    return CryptoAnalyzer(_CONFIG_PATH)


def _run_analysis(
    analyzer: CryptoAnalyzer,
    symbol: str,
    interval: str,
    lookback_days: int,
    force_retrain: bool,
) -> AnalysisResult:
    cache_key = f"analysis_{symbol}_{interval}_{lookback_days}"
    force_key = f"force_retrain_{symbol}"

    if (
        cache_key not in st.session_state
        or force_retrain
        or st.session_state.get(force_key)
    ):
        with st.spinner(
            f"Analysiere {symbol}... (erster Lauf trainiert KI-Modell, ca. 20–60s)"
        ):
            result = analyzer.analyze(symbol, interval, lookback_days, force_retrain)
        st.session_state[cache_key] = result
        st.session_state[force_key] = False

    return st.session_state[cache_key]


def _get_or_run_analysis(
    analyzer: CryptoAnalyzer,
    symbol: str,
    interval: str,
    lookback_days: int,
    force_retrain: bool,
    analyze_btn: bool,
) -> AnalysisResult | None:
    cache_key = f"analysis_{symbol}_{interval}_{lookback_days}"
    if analyze_btn or force_retrain or cache_key not in st.session_state:
        return _run_analysis(analyzer, symbol, interval, lookback_days, force_retrain)
    return st.session_state.get(cache_key)


# ─────────────────────────────────────────────────────────────────────────────
# SCREENER
# ─────────────────────────────────────────────────────────────────────────────

def _run_screener(
    analyzer: CryptoAnalyzer,
    symbols: list[str],
    interval: str,
    lookback_days: int,
) -> list[dict[str, Any]]:
    """Führt einen schnellen technischen Screener für mehrere Coins durch."""
    rows = []
    progress = st.progress(0, text="Scanne Coins...")

    for i, sym in enumerate(symbols):
        progress.progress((i + 1) / len(symbols), text=f"Scanne {sym}...")
        try:
            df = analyzer._fetcher.get_ohlcv(sym, interval, lookback_days)
            if df.empty or len(df) < 30:
                continue

            from src.features.technical import TechnicalIndicators
            df = TechnicalIndicators(config=_load_config()).add_all(df)

            last = df.iloc[-1]

            # Richtungs-Signal aus Technik (schnell, kein ML)
            rsi = last.get("rsi_14", 50)
            macd_hist = last.get("macd_diff", 0)
            bb_pct = last.get("bb_pct", 0.5)
            ema9 = last.get("ema_9", None)
            ema21 = last.get("ema_21", None)
            close = last["close"]

            # Einfaches technisches Signal
            bullish_factors = 0
            bearish_factors = 0
            if rsi < 40:
                bullish_factors += 1
            elif rsi > 65:
                bearish_factors += 1

            if macd_hist > 0:
                bullish_factors += 1
            elif macd_hist < 0:
                bearish_factors += 1

            if ema9 is not None and ema21 is not None:
                if ema9 > ema21:
                    bullish_factors += 1
                elif ema9 < ema21:
                    bearish_factors += 1

            if bb_pct < 0.25:
                bullish_factors += 1
            elif bb_pct > 0.80:
                bearish_factors += 1

            if bullish_factors >= 3:
                signal = "BULLISH"
            elif bearish_factors >= 3:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"

            # 7-Tage-Preisänderung
            price_7d = None
            if len(df) >= 8:
                price_7d = (df["close"].iloc[-1] / df["close"].iloc[-8] - 1) * 100

            rows.append({
                "symbol": sym,
                "rsi_14": round(float(rsi), 1) if pd.notna(rsi) else None,
                "macd_diff": round(float(macd_hist), 4) if pd.notna(macd_hist) else None,
                "bb_pct": round(float(bb_pct), 3) if pd.notna(bb_pct) else None,
                "adx": round(float(last.get("adx", 0)), 1) if pd.notna(last.get("adx")) else None,
                "price_change_7d": round(price_7d, 2) if price_7d is not None else None,
                "signal": signal,
                "confidence": None,
                "close": round(float(close), 6),
            })
        except Exception as exc:
            logger.warning(f"Screener: {sym} fehlgeschlagen – {exc}")

    progress.empty()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# HAUPT-DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard() -> None:
    config = _load_config()
    ui_cfg = config["ui"]

    st.set_page_config(
        page_title=f"{ui_cfg['page_title']} v2",
        page_icon=ui_cfg["page_icon"],
        layout=ui_cfg["layout"],
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS für schöneres Layout ──────────────────────────────────
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 18px;
            border-radius: 6px;
            font-weight: 500;
        }
        div[data-testid="metric-container"] {
            background-color: rgba(128,128,128,0.05);
            border: 1px solid rgba(128,128,128,0.15);
            padding: 12px 16px;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Disclaimer ────────────────────────────────────────────────────────
    render_disclaimer(ui_cfg["disclaimer_short"])

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title(f"{ui_cfg['page_icon']} Crypto Analyzer")
        st.caption(f"v{config['app']['version']} · Bildungszwecke")
        st.divider()

        symbol_input = st.text_input(
            "Coin-Symbol",
            value=st.session_state.get("selected_symbol", ui_cfg["default_symbol"]),
            max_chars=10,
            help="Binance-Symbol ohne USDT (z.B. BTC, ETH, DOGE)",
        ).strip().upper()

        st.caption("Schnellwahl:")
        popular = ui_cfg["popular_symbols"]
        cols = st.columns(3)
        for i, sym in enumerate(popular):
            with cols[i % 3]:
                if st.button(sym, key=f"quick_{sym}", use_container_width=True):
                    st.session_state["selected_symbol"] = sym
                    st.rerun()

        if "selected_symbol" in st.session_state:
            symbol_input = st.session_state["selected_symbol"]

        st.divider()

        interval = st.selectbox(
            "Intervall",
            ui_cfg["available_intervals"],
            index=ui_cfg["available_intervals"].index(ui_cfg["default_interval"]),
            help="1d = Täglich (empfohlen)",
        )

        lookback_days = st.slider(
            "Historische Tage",
            min_value=90,
            max_value=730,
            value=ui_cfg["default_lookback_days"],
            step=30,
        )

        st.divider()

        analyze_btn = st.button(
            "🔍 Analysieren",
            type="primary",
            use_container_width=True,
        )
        force_retrain = st.button(
            "🔄 Modell neu trainieren",
            use_container_width=True,
        )

        # Trending
        st.divider()
        st.subheader("🔥 Trending")
        try:
            trending = _get_analyzer().get_trending_coins()
            for coin in trending[:6]:
                sym = coin.get("symbol", "")
                name = coin.get("name", sym)
                rank = coin.get("rank")
                rank_str = f"#{rank} " if rank else ""
                if st.button(
                    f"{rank_str}**{sym}** – {name}",
                    key=f"trending_{sym}",
                    use_container_width=True,
                ):
                    st.session_state["selected_symbol"] = sym
                    st.rerun()
        except Exception:
            st.caption("Trending nicht verfügbar.")

    # ── Keine Eingabe ────────────────────────────────────────────────────
    if not symbol_input:
        _render_welcome()
        return

    # ── Analyse ──────────────────────────────────────────────────────────
    result = _get_or_run_analysis(
        _get_analyzer(), symbol_input, interval, lookback_days, force_retrain, analyze_btn
    )

    if result is None:
        st.error("Analyse fehlgeschlagen. Bitte erneut versuchen.")
        return

    if result.error:
        st.error(f"**Fehler bei {symbol_input}**\n\n{result.error}")
        st.info(
            "💡 Tipps:\n"
            "- Prüfe das Symbol (BTC, ETH, DOGE – ohne USDT)\n"
            "- Internetverbindung prüfen\n"
            "- Kurz warten und erneut versuchen (API Rate Limit)"
        )
        return

    # ── Datenwarnungen ───────────────────────────────────────────────────
    if result.warnings:
        with st.expander(f"⚠️ {len(result.warnings)} Datenwarnungen"):
            for w in result.warnings:
                st.warning(w)

    # ── Header ──────────────────────────────────────────────────────────
    coin_name = result.market_data.get("name", result.symbol)
    freshness = (
        f"Daten: vor {result.data_freshness_minutes:.0f} Min."
        if result.data_freshness_minutes > 0
        else "Daten: frisch"
    )
    train_info = (
        f" · KI trainiert in {result.training_time_seconds:.1f}s"
        if result.training_time_seconds > 1
        else ""
    )

    st.title(f"📊 {coin_name} ({result.symbol})")
    st.caption(f"{freshness}{train_info}")

    render_market_data_row(result.market_data)
    st.divider()

    # ── Tabs ────────────────────────────────────────────────────────────
    tab_analyse, tab_backtest, tab_screener, tab_model, tab_sentiment = st.tabs([
        "📊 Analyse",
        "🔄 Backtest",
        "🔍 Screener",
        "🤖 Modell",
        "💭 Sentiment",
    ])

    # ──────────────────────────────────────────────────────────────────
    # TAB 1: ANALYSE
    # ──────────────────────────────────────────────────────────────────
    with tab_analyse:
        chart_col, signal_col = st.columns([2, 1], gap="large")

        with chart_col:
            if not result.ohlcv.empty:
                fig = render_price_chart(result.ohlcv, result.symbol)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Keine Chart-Daten verfügbar.")

        with signal_col:
            st.subheader("KI-Signal (5-Tage-Horizont)")
            if result.prediction:
                render_signal_card(result.prediction)
                st.markdown("---")
                render_volatility_badge(result.prediction)
                st.markdown("---")
                with st.expander("Wahrscheinlichkeiten", expanded=True):
                    render_probability_bars(result.prediction)
                st.caption(
                    f"Datenbasis bis: **{result.prediction.data_end_date}**"
                )
            else:
                st.info("Kein Signal verfügbar.")

        # Technische Indikatoren
        st.subheader("Technische Indikatoren")
        ind_col1, ind_col2 = st.columns(2)

        with ind_col1:
            st.caption("**RSI** – Über 70: Überkauft ⚠️  |  Unter 30: Überverkauft ⚠️")
            rsi_fig = render_rsi_chart(result.ohlcv)
            if rsi_fig:
                st.plotly_chart(rsi_fig, use_container_width=True)
            else:
                st.info("RSI-Daten nicht verfügbar.")

        with ind_col2:
            st.caption("**MACD** – Histogramm positiv: bullish  |  negativ: bearish")
            macd_fig = render_macd_chart(result.ohlcv)
            if macd_fig:
                st.plotly_chart(macd_fig, use_container_width=True)
            else:
                st.info("MACD-Daten nicht verfügbar.")

        # Indikator-Tabelle
        if not result.ohlcv.empty:
            st.caption("**Aktuelle Indikator-Werte (letzter Kerzenschluss)**")
            last = result.ohlcv.iloc[-1]
            indicator_pairs = [
                ("rsi_14", "RSI(14)"),
                ("rsi_7", "RSI(7)"),
                ("macd_diff", "MACD-Histogramm"),
                ("adx", "ADX (Trendstärke)"),
                ("bb_pct", "Bollinger %B"),
                ("hist_vol", "Hist. Volatilität"),
                ("volume_sma_ratio", "Vol./Ø-Volumen"),
                ("atr", "ATR"),
            ]
            ind_data = {}
            for col, label in indicator_pairs:
                if col in result.ohlcv.columns:
                    val = last[col]
                    ind_data[label] = f"{val:.4f}" if pd.notna(val) else "–"

            if ind_data:
                # 2-spaltige Tabelle
                items = list(ind_data.items())
                half = len(items) // 2 + len(items) % 2
                col_a, col_b = st.columns(2)
                with col_a:
                    df_a = pd.DataFrame(items[:half], columns=["Indikator", "Wert"])
                    st.dataframe(df_a, hide_index=True, use_container_width=True)
                with col_b:
                    df_b = pd.DataFrame(items[half:], columns=["Indikator", "Wert"])
                    st.dataframe(df_b, hide_index=True, use_container_width=True)

        # Coin-Vergleich
        with st.expander("⚖️ Coin-Vergleich"):
            st.caption("Vergleiche bis zu 5 Coins nach relativer Performance.")
            compare_input = st.text_input(
                "Symbole (komma-getrennt)",
                value=f"{result.symbol}, BTC",
                key="compare_input",
            )
            if st.button("Vergleich starten", key="compare_btn"):
                _render_coin_comparison(
                    compare_input, interval, min(lookback_days, 180)
                )

    # ──────────────────────────────────────────────────────────────────
    # TAB 2: BACKTEST
    # ──────────────────────────────────────────────────────────────────
    with tab_backtest:
        st.subheader("Walk-Forward Backtest")
        st.caption(
            "Simulation einer KI-gesteuerten Long-Only-Strategie auf historischen Daten. "
            "Signale stammen aus Walk-Forward-Folds (kein Lookahead-Bias). "
            "Vergangenheit ≠ Zukunft."
        )

        bt = result.backtest
        if bt is None:
            if result.eval_metrics and result.eval_metrics.n_folds == 0:
                st.info(
                    "Backtest-Daten sind nur beim ersten Training verfügbar. "
                    "Klicke '🔄 Modell neu trainieren' in der Seitenleiste."
                )
            else:
                st.warning(
                    "Kein Backtest möglich. Zu wenig Walk-Forward-Testdaten "
                    "(mindestens 5 Handelstage nötig)."
                )
        else:
            render_backtest_metrics(bt)
            st.markdown("---")

            bt_fig = render_backtest_chart(bt, result.symbol)
            st.plotly_chart(bt_fig, use_container_width=True)

            # Drawdown-Chart
            port = bt.series["portfolio"]
            peak = port.cummax()
            dd = ((port - peak) / peak * 100)

            dd_fig = go.Figure()
            dd_fig.add_trace(go.Scatter(
                x=dd.index,
                y=dd.values,
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="#c82538", width=1),
                fillcolor="rgba(200,37,56,0.15)",
            ))
            dd_fig.update_layout(
                title="Drawdown (%)",
                height=200,
                yaxis_title="%",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
                yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
                showlegend=False,
            )
            st.plotly_chart(dd_fig, use_container_width=True)

            st.info(config.get("backtest", {}).get("disclaimer", ""), icon="⚠️")

    # ──────────────────────────────────────────────────────────────────
    # TAB 3: SCREENER
    # ──────────────────────────────────────────────────────────────────
    with tab_screener:
        st.subheader("Technischer Multi-Coin Screener")
        st.caption(
            "Scannt mehrere Coins gleichzeitig und zeigt technische Signale. "
            "Basiert auf RSI, MACD, EMA-Kreuzung und Bollinger Bands."
        )

        screener_cfg = config.get("screener", {})
        default_syms = screener_cfg.get("default_symbols", ui_cfg["popular_symbols"])

        screener_symbols_raw = st.text_input(
            "Symbole (komma-getrennt)",
            value=", ".join(default_syms[:8]),
            key="screener_symbols",
        )
        screener_lookback = st.slider(
            "Lookback-Tage (Screener)",
            min_value=90,
            max_value=365,
            value=screener_cfg.get("lookback_days", 180),
            step=30,
            key="screener_lookback",
        )

        if st.button("🔍 Scan starten", type="primary", key="scan_btn"):
            scan_syms = [
                s.strip().upper()
                for s in screener_symbols_raw.split(",")
                if s.strip()
            ][:12]

            if scan_syms:
                rows = _run_screener(
                    _get_analyzer(), scan_syms, interval, screener_lookback
                )
                st.session_state["screener_rows"] = rows
                st.session_state["screener_timestamp"] = pd.Timestamp.now()

        if "screener_rows" in st.session_state:
            ts = st.session_state.get("screener_timestamp")
            if ts:
                st.caption(f"Letzter Scan: {ts.strftime('%H:%M:%S')}")
            render_screener_table(st.session_state["screener_rows"])

            # Verteilungs-Chart
            rows = st.session_state["screener_rows"]
            if rows:
                sigs = [r["signal"] for r in rows]
                sig_counts = pd.Series(sigs).value_counts().reset_index()
                sig_counts.columns = ["Signal", "Anzahl"]
                color_map = {
                    "BULLISH": "#1a7f37",
                    "NEUTRAL": "#b08800",
                    "BEARISH": "#c82538",
                }
                fig_pie = px.pie(
                    sig_counts,
                    values="Anzahl",
                    names="Signal",
                    color="Signal",
                    color_discrete_map=color_map,
                    title="Signal-Verteilung",
                    hole=0.4,
                )
                fig_pie.update_layout(
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                col_pie, col_info = st.columns([1, 2])
                with col_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col_info:
                    bullish_coins = [r["symbol"] for r in rows if r["signal"] == "BULLISH"]
                    bearish_coins = [r["symbol"] for r in rows if r["signal"] == "BEARISH"]
                    if bullish_coins:
                        st.success(f"Bullische Signale: **{', '.join(bullish_coins)}**")
                    if bearish_coins:
                        st.error(f"Bärische Signale: **{', '.join(bearish_coins)}**")
                    st.caption(
                        "⚠️ Technische Signale allein sind keine Handelsempfehlung. "
                        "Immer eigene Recherche durchführen."
                    )
        else:
            st.info("Klicke 'Scan starten' um den Screener zu starten.")

    # ──────────────────────────────────────────────────────────────────
    # TAB 4: MODELL
    # ──────────────────────────────────────────────────────────────────
    with tab_model:
        st.subheader("KI-Modell Transparenz")

        if result.eval_metrics and result.eval_metrics.n_folds > 0:
            st.caption(
                f"Walk-Forward-Validation über **{result.eval_metrics.n_folds} Folds**. "
                "Jeder Fold testet auf Daten, die beim Training strikt in der Zukunft lagen."
            )
            render_model_performance_badge(result.eval_metrics)
            st.info(result.eval_metrics.disclaimer, icon="ℹ️")

            # Fold-Accuracy-Verlauf
            if result.eval_metrics.fold_accuracies:
                accs = result.eval_metrics.fold_accuracies
                fold_df = pd.DataFrame({
                    "Fold": list(range(1, len(accs) + 1)),
                    "Accuracy": accs,
                    "Baseline (immer NEUTRAL)": [result.eval_metrics.baseline_accuracy] * len(accs),
                })
                fig_acc = px.line(
                    fold_df,
                    x="Fold",
                    y=["Accuracy", "Baseline (immer NEUTRAL)"],
                    title="Accuracy pro Walk-Forward-Fold",
                    color_discrete_map={
                        "Accuracy": "#2196F3",
                        "Baseline (immer NEUTRAL)": "#FF9800",
                    },
                    markers=True,
                )
                fig_acc.update_layout(
                    height=280,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    xaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
                    yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
                )
                st.plotly_chart(fig_acc, use_container_width=True)

            # Klassen-Precision
            if result.eval_metrics.avg_precision_per_class:
                prec = result.eval_metrics.avg_precision_per_class
                prec_df = pd.DataFrame(
                    [{"Klasse": k, "Precision": v} for k, v in prec.items()]
                )
                fig_prec = px.bar(
                    prec_df,
                    x="Klasse",
                    y="Precision",
                    title="Durchschnittliche Precision pro Klasse",
                    color="Klasse",
                    color_discrete_map={
                        "BULLISH": "#1a7f37",
                        "NEUTRAL": "#b08800",
                        "BEARISH": "#c82538",
                    },
                    text="Precision",
                )
                fig_prec.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig_prec.update_layout(
                    height=250,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[0, 1]),
                )
                st.plotly_chart(fig_prec, use_container_width=True)

        else:
            st.info(
                "Modell aus Cache geladen – keine aktuellen Eval-Metriken. "
                "Klicke '🔄 Modell neu trainieren' für aktuelle Metriken."
            )

        st.divider()
        if result.feature_importance:
            st.subheader("Feature Importance (Top 15)")
            st.caption(
                "Hohe Importance = mehr Einfluss auf KI-Vorhersage. "
                "Durchschnitt über alle Walk-Forward-Folds."
            )
            fi_fig = render_feature_importance_chart(result.feature_importance)
            st.plotly_chart(fi_fig, use_container_width=True)

            with st.expander("Vollständige Feature-Liste"):
                fi_df = pd.DataFrame(
                    sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True),
                    columns=["Feature", "Importance (%)"],
                )
                st.dataframe(fi_df, hide_index=True, use_container_width=True)
        else:
            st.info("Feature-Importance nicht verfügbar (gecachtes Modell).")

    # ──────────────────────────────────────────────────────────────────
    # TAB 5: SENTIMENT
    # ──────────────────────────────────────────────────────────────────
    with tab_sentiment:
        sent = result.sentiment
        st.subheader("Markt-Sentiment")

        sent_col1, sent_col2 = st.columns(2)

        with sent_col1:
            st.subheader("Fear & Greed Index")
            fg = sent.get("fear_greed", {})
            if fg.get("current_value") is not None:
                fig_gauge = render_fear_greed_gauge(
                    fg["current_value"], fg.get("current_label", "")
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                change = fg.get("change_3d")
                if change is not None:
                    direction = "gestiegener" if change > 0 else "gesunkener"
                    st.caption(
                        f"3-Tages-Veränderung: **{change:+.0f} Punkte** "
                        f"({direction} Optimismus)"
                    )
                st.caption(
                    "Contrarian: Extremer Fear → mögliche Kaufgelegenheit. "
                    "Extremer Greed → mögliche Übertreibung."
                )
            else:
                st.info("Fear & Greed Index nicht verfügbar.")

        with sent_col2:
            st.subheader(f"Reddit-Sentiment · {result.symbol}")
            reddit = sent.get("reddit", {})
            if reddit.get("post_count", 0) > 0:
                post_count = reddit["post_count"]
                bull_pct = reddit["bullish_score"] * 100
                bear_pct = reddit["bearish_score"] * 100
                neut_pct = reddit["neutral_score"] * 100
                avg_ups = reddit["avg_upvotes"]

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Relevante Posts", post_count)
                with col_b:
                    st.metric("Ø Upvotes", f"{avg_ups:.0f}")

                sent_df = pd.DataFrame({
                    "Sentiment": ["Bullish", "Neutral", "Bearish"],
                    "Anteil (%)": [bull_pct, neut_pct, bear_pct],
                })
                fig_sent = px.bar(
                    sent_df,
                    x="Anteil (%)",
                    y="Sentiment",
                    orientation="h",
                    color="Sentiment",
                    color_discrete_map={
                        "Bullish": "#1a7f37",
                        "Neutral": "#607D8B",
                        "Bearish": "#c82538",
                    },
                )
                fig_sent.update_layout(
                    height=200,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_sent, use_container_width=True)

                subs = reddit.get("subreddits_checked", [])
                if subs:
                    st.caption(f"Subreddits: {', '.join(f'r/{s}' for s in subs)}")

                top_titles = reddit.get("top_titles", [])
                if top_titles:
                    st.markdown("**Top Reddit-Posts:**")
                    for title in top_titles[:5]:
                        st.caption(f"• {title}")
            else:
                st.info(
                    f"Keine Reddit-Posts mit '{result.symbol}' gefunden. "
                    "Weniger bekannte Coins haben oft wenig Reddit-Aktivität."
                )

    # ── Finaler Disclaimer ──────────────────────────────────────────────
    st.divider()
    st.info(ui_cfg["disclaimer_long"], icon="⚠️")


# ─────────────────────────────────────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────────────────────────────────────

def _render_welcome() -> None:
    """Willkommensseite wenn kein Symbol eingegeben."""
    st.markdown(
        """
        ## Willkommen beim Crypto Analyzer v2

        Dieses Tool analysiert Kryptowährungen mit:

        | Feature | Beschreibung |
        |---|---|
        | 📊 **Technische Analyse** | RSI, MACD, Bollinger Bands, EMA, ADX, ATR |
        | 🤖 **KI-Vorhersage** | LightGBM mit Walk-Forward-Validation |
        | 🔄 **Backtesting** | Portfolio-Simulation auf historischen Signalen |
        | 🔍 **Screener** | Mehrere Coins gleichzeitig scannen |
        | 💭 **Sentiment** | Fear & Greed Index + Reddit-Analyse |

        **Coin eingeben** (links in der Seitenleiste) und **Analysieren** klicken.

        ---
        ⚠️ *Alle Analysen dienen ausschließlich Bildungszwecken. Kein Anlageprodukt.*
        """
    )


def _render_coin_comparison(
    compare_input: str, interval: str, lookback_days: int
) -> None:
    """Rendert den Coin-Vergleichs-Chart."""
    compare_symbols = [
        s.strip().upper() for s in compare_input.split(",") if s.strip()
    ][:5]

    compare_dfs: dict[str, pd.DataFrame] = {}
    analyzer_inst = _get_analyzer()

    with st.spinner("Lade Vergleichs-Daten..."):
        for sym in compare_symbols:
            try:
                df_cmp = analyzer_inst._fetcher.get_ohlcv(sym, interval, lookback_days)
                compare_dfs[sym] = df_cmp
            except Exception as e:
                st.warning(f"{sym}: Nicht ladbar ({e})")

    if len(compare_dfs) >= 2:
        fig_cmp = go.Figure()
        for sym, df_cmp in compare_dfs.items():
            norm = df_cmp["close"] / df_cmp["close"].iloc[0] * 100
            fig_cmp.add_trace(go.Scatter(
                x=df_cmp.index,
                y=norm,
                name=sym,
                mode="lines",
            ))
        fig_cmp.update_layout(
            title="Relative Performance (Index = 100)",
            height=400,
            yaxis_title="Index (Start = 100)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    elif compare_dfs:
        st.info("Mindestens 2 gültige Symbole für Vergleich nötig.")
