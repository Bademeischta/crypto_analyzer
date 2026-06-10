"""Streamlit-Hauptdashboard.

Layout (5 Sektionen):
  1. Disclaimer (immer sichtbar, oben)
  2. Coin-Auswahl + Controls
  3. Preischart (links) + KI-Signal-Karte (rechts)
  4. Technische Indikatoren (RSI, MACD) – zugeklappt
  5. Modell-Transparenz (Feature Importance, Performance) – zugeklappt
  6. Sentiment (Fear & Greed, Reddit) – zugeklappt

Alle teuren Operationen (Daten fetchen, Modell trainieren) werden durch
st.session_state gecacht damit Streamlit nicht bei jedem Widget-Klick neu rechnet.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# Projekt-Root zum Pythonpfad hinzufügen damit Imports funktionieren
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.analyzer import CryptoAnalyzer, AnalysisResult
from src.ui.components import (
    render_disclaimer,
    render_signal_card,
    render_volatility_badge,
    render_probability_bars,
    render_price_chart,
    render_rsi_chart,
    render_macd_chart,
    render_feature_importance_chart,
    render_fear_greed_gauge,
    render_model_performance_badge,
    render_market_data_row,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@st.cache_resource(show_spinner=False)
def _get_analyzer() -> CryptoAnalyzer:
    """Erstellt den Analyzer einmalig (wird von Streamlit gecacht)."""
    return CryptoAnalyzer(_CONFIG_PATH)


def _run_analysis(
    analyzer: CryptoAnalyzer,
    symbol: str,
    interval: str,
    lookback_days: int,
    force_retrain: bool,
) -> AnalysisResult:
    """Führt Analyse durch und speichert Ergebnis in session_state."""
    cache_key = f"analysis_{symbol}_{interval}_{lookback_days}"
    force_key = f"force_retrain_{symbol}"

    # Ergebnis nur neu berechnen wenn nötig
    if (
        cache_key not in st.session_state
        or force_retrain
        or st.session_state.get(force_key)
    ):
        with st.spinner(f"Analysiere {symbol}... (erster Lauf trainiert ML-Modell, ca. 10-30s)"):
            result = analyzer.analyze(symbol, interval, lookback_days, force_retrain)
        st.session_state[cache_key] = result
        st.session_state[force_key] = False

    return st.session_state[cache_key]


def render_dashboard() -> None:
    """Haupt-Render-Funktion des Dashboards."""
    config = _load_config()
    ui_cfg = config["ui"]

    # ── Seiten-Konfiguration ──────────────────────────────────────────────
    st.set_page_config(
        page_title=ui_cfg["page_title"],
        page_icon=ui_cfg["page_icon"],
        layout=ui_cfg["layout"],
        initial_sidebar_state="expanded",
    )

    # ── Disclaimer (immer sichtbar) ───────────────────────────────────────
    render_disclaimer(ui_cfg["disclaimer_short"])

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title(f"{ui_cfg['page_icon']} Crypto Analyzer")
        st.caption(f"Version {config['app']['version']}")
        st.divider()

        # Coin-Eingabe
        symbol_input = st.text_input(
            "Coin-Symbol eingeben",
            value=ui_cfg["default_symbol"],
            max_chars=10,
            help="Binance-Symbol ohne USDT (z.B. BTC, ETH, DOGE, SHIB, PEPE)",
        ).strip().upper()

        # Oder aus Vorschlagsliste
        st.caption("oder schnell wählen:")
        popular = ui_cfg["popular_symbols"]
        cols = st.columns(3)
        for i, sym in enumerate(popular):
            with cols[i % 3]:
                if st.button(sym, key=f"quick_{sym}", use_container_width=True):
                    symbol_input = sym
                    st.session_state["selected_symbol"] = sym

        # session_state Priorität
        if "selected_symbol" in st.session_state:
            symbol_input = st.session_state["selected_symbol"]

        st.divider()

        # Zeitraum
        interval = st.selectbox(
            "Kerzen-Intervall",
            ui_cfg["available_intervals"],
            index=ui_cfg["available_intervals"].index(ui_cfg["default_interval"]),
            help="1d = Täglich (empfohlen), 4h = 4-Stunden, 1h = Stündlich",
        )

        lookback_days = st.slider(
            "Historische Daten (Tage)",
            min_value=90,
            max_value=730,
            value=ui_cfg["default_lookback_days"],
            step=30,
            help="Mehr Daten = stabileres Modell, aber längere Ladezeit",
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
            help="Erzwingt ein Neutraining auch wenn das Modell noch frisch ist",
        )

        # Trending Coins
        st.divider()
        st.subheader("🔥 Trending")
        try:
            analyzer_for_trending = _get_analyzer()
            trending = analyzer_for_trending.get_trending_coins()
            for coin in trending[:7]:
                sym = coin.get("symbol", "")
                name = coin.get("name", sym)
                rank = coin.get("rank")
                rank_str = f"#{rank}" if rank else ""
                st.caption(f"{rank_str} **{sym}** – {name}")
        except Exception:
            st.caption("Trending-Daten nicht verfügbar.")

    # ── Kein Symbol → Willkommensseite ───────────────────────────────────
    if not symbol_input:
        st.info("Gib links ein Coin-Symbol ein und klicke 'Analysieren'.")
        return

    # Analyse starten wenn Button geklickt oder Symbol geändert
    if analyze_btn or force_retrain or f"analysis_{symbol_input}_{interval}_{lookback_days}" not in st.session_state:
        result = _run_analysis(
            _get_analyzer(),
            symbol_input,
            interval,
            lookback_days,
            force_retrain,
        )
    else:
        result = st.session_state.get(f"analysis_{symbol_input}_{interval}_{lookback_days}")
        if result is None:
            result = _run_analysis(_get_analyzer(), symbol_input, interval, lookback_days, False)

    # ── Fehlerfall ────────────────────────────────────────────────────────
    if result is None:
        st.error("Analyse fehlgeschlagen. Bitte versuche es erneut.")
        return

    if result.error:
        st.error(f"**Fehler bei der Analyse von {symbol_input}**\n\n{result.error}")
        st.info(
            "💡 Tipps:\n"
            "- Prüfe das Symbol (z.B. BTC, ETH, DOGE – kein USDT anhängen)\n"
            "- Stelle sicher dass du eine Internetverbindung hast\n"
            "- Warte kurz und versuche es erneut (API Rate Limit)"
        )
        return

    # ── Warnungen anzeigen ────────────────────────────────────────────────
    if result.warnings:
        with st.expander(f"⚠️ {len(result.warnings)} Datenwarnungen", expanded=False):
            for w in result.warnings:
                st.warning(w)

    # ── Seitentitel ───────────────────────────────────────────────────────
    coin_name = result.market_data.get("name", result.symbol)
    st.title(f"📊 {coin_name} ({result.symbol})")

    freshness_str = (
        f"Daten: vor {result.data_freshness_minutes:.0f} Min. aktualisiert"
        if result.data_freshness_minutes > 0
        else "Daten: frisch geladen"
    )
    train_str = (
        f" · Modell trainiert in {result.training_time_seconds:.1f}s"
        if result.training_time_seconds > 1
        else ""
    )
    st.caption(f"{freshness_str}{train_str}")

    # ── Marktdaten-Zeile ─────────────────────────────────────────────────
    render_market_data_row(result.market_data)
    st.divider()

    # ── Hauptbereich: Chart (links) + Signal (rechts) ─────────────────────
    chart_col, signal_col = st.columns([2, 1], gap="large")

    with chart_col:
        if not result.ohlcv.empty:
            fig = render_price_chart(result.ohlcv, result.symbol)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Keine Chart-Daten verfügbar.")

    with signal_col:
        st.subheader("KI-Signal")
        if result.prediction:
            render_signal_card(result.prediction)
            st.markdown("---")
            render_volatility_badge(result.prediction)
            st.markdown("---")
            with st.expander("Wahrscheinlichkeiten", expanded=True):
                render_probability_bars(result.prediction)
            st.caption(
                f"Modell trainiert auf Daten bis: **{result.prediction.data_end_date}**"
            )
        else:
            st.info("Kein Signal verfügbar.")

    # ── Technische Indikatoren ────────────────────────────────────────────
    with st.expander("📈 Technische Indikatoren", expanded=False):
        if not result.ohlcv.empty:
            ind_col1, ind_col2 = st.columns(2)
            with ind_col1:
                st.markdown("**RSI** — Relative Strength Index")
                st.caption("Über 70: Überkauft ⚠️ | Unter 30: Überverkauft ⚠️ | Dazwischen: Normal")
                rsi_fig = render_rsi_chart(result.ohlcv)
                if rsi_fig:
                    st.plotly_chart(rsi_fig, use_container_width=True)
                else:
                    st.info("RSI-Daten nicht verfügbar.")
            with ind_col2:
                st.markdown("**MACD** — Moving Average Convergence Divergence")
                st.caption("MACD > Signal-Linie: Bullisches Signal | MACD < Signal-Linie: Bärisches Signal")
                macd_fig = render_macd_chart(result.ohlcv)
                if macd_fig:
                    st.plotly_chart(macd_fig, use_container_width=True)
                else:
                    st.info("MACD-Daten nicht verfügbar.")

            # Letzte Indikator-Werte als Tabelle
            st.markdown("**Aktuelle Werte (letzter Kerzenschluss)**")
            last = result.ohlcv.iloc[-1]
            indicator_summary = {}
            for col, label in [
                ("rsi_14", "RSI(14)"),
                ("rsi_7", "RSI(7)"),
                ("macd_diff", "MACD-Hist."),
                ("adx", "ADX"),
                ("bb_pct", "Bollinger %B"),
                ("hist_vol", "Hist. Volatilität"),
                ("volume_sma_ratio", "Vol./Avg.Volumen"),
            ]:
                if col in result.ohlcv.columns:
                    val = last[col]
                    indicator_summary[label] = f"{val:.3f}" if pd.notna(val) else "–"

            if indicator_summary:
                ind_df = pd.DataFrame(
                    list(indicator_summary.items()),
                    columns=["Indikator", "Wert"],
                )
                st.dataframe(ind_df, use_container_width=True, hide_index=True)
        else:
            st.info("Keine Indikator-Daten verfügbar.")

    # ── Modell-Transparenz ────────────────────────────────────────────────
    with st.expander("🤖 Modell-Transparenz", expanded=False):
        if result.eval_metrics and result.eval_metrics.n_folds > 0:
            st.subheader("Historische Modell-Performance")
            st.caption(
                "Diese Metriken zeigen wie gut das Modell in vergangenen "
                "Testperioden war. **Vergangenheit ≠ Zukunft.**"
            )
            render_model_performance_badge(result.eval_metrics)

            # Fold-Accuracies als Liniendiagramm
            if result.eval_metrics.fold_accuracies:
                import plotly.express as px
                accs = result.eval_metrics.fold_accuracies
                fold_df = pd.DataFrame({
                    "Fold": list(range(1, len(accs) + 1)),
                    "Accuracy": accs,
                    "Baseline": [result.eval_metrics.baseline_accuracy] * len(accs),
                })
                fig_acc = px.line(
                    fold_df,
                    x="Fold",
                    y=["Accuracy", "Baseline"],
                    title="Accuracy pro Walk-Forward-Fold",
                    color_discrete_map={"Accuracy": "#2196F3", "Baseline": "#FF9800"},
                    markers=True,
                )
                fig_acc.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info(
                "Modell wurde aus dem Cache geladen. "
                "Klicke 'Modell neu trainieren' um aktuelle Performance-Metriken zu sehen."
            )

        st.divider()
        if result.feature_importance:
            st.subheader("Feature Importance")
            st.caption(
                "Welche Indikatoren haben das Modell am stärksten beeinflusst? "
                "Hohe Importance = mehr Einfluss auf das Signal."
            )
            fi_fig = render_feature_importance_chart(result.feature_importance)
            st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.info("Feature-Importance nicht verfügbar (gecachtes Modell).")

    # ── Sentiment-Panel ────────────────────────────────────────────────────
    with st.expander("💭 Sentiment-Analyse", expanded=False):
        sent = result.sentiment
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
                    "**Contrarian-Interpretation:** Extremer Fear kann eine "
                    "Kaufgelegenheit signalisieren – extremer Greed deutet auf "
                    "mögliche Korrektur hin. Keine Handlungsempfehlung."
                )
            else:
                st.info("Fear & Greed Index aktuell nicht verfügbar.")

        with sent_col2:
            st.subheader(f"Reddit-Sentiment für {result.symbol}")
            reddit = sent.get("reddit", {})
            if reddit.get("post_count", 0) > 0:
                post_count = reddit["post_count"]
                bull_pct = reddit["bullish_score"] * 100
                bear_pct = reddit["bearish_score"] * 100
                neut_pct = reddit["neutral_score"] * 100
                avg_ups = reddit["avg_upvotes"]

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Relevante Posts", post_count, help="Posts die das Symbol erwähnen")
                with col_b:
                    st.metric("Ø Upvotes", f"{avg_ups:.0f}")

                # Sentiment-Balken
                import plotly.express as px
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
                    height=180,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_sent, use_container_width=True)

                subs = reddit.get("subreddits_checked", [])
                st.caption(f"Durchsucht: {', '.join(f'r/{s}' for s in subs)}")

                # Top-Posts
                top_titles = reddit.get("top_titles", [])
                if top_titles:
                    st.markdown("**Top Reddit-Posts:**")
                    for title in top_titles:
                        st.caption(f"• {title}")

                st.caption(
                    "⚠️ Reddit-Sentiment basiert auf Keyword-Analyse und ist "
                    "kein zuverlässiger Handelsindikator."
                )
            else:
                st.info(
                    f"Keine Reddit-Posts mit '{result.symbol}' Erwähnung gefunden. "
                    f"Weniger bekannte Coins haben oft wenig Reddit-Aktivität."
                )

    # ── Coin-Vergleich ────────────────────────────────────────────────────
    with st.expander("⚖️ Coin-Vergleich (mehrere Symbole)", expanded=False):
        st.caption(
            "Vergleiche bis zu 5 Coins anhand ihrer relativen Performance."
        )
        compare_input = st.text_input(
            "Symbole komma-getrennt eingeben",
            value=f"{result.symbol}, BTC",
            key="compare_input",
            help="z.B.: DOGE, SHIB, PEPE",
        )

        if st.button("Vergleich starten", key="compare_btn"):
            compare_symbols = [
                s.strip().upper()
                for s in compare_input.split(",")
                if s.strip()
            ][:5]

            compare_dfs: dict[str, pd.DataFrame] = {}
            analyzer_inst = _get_analyzer()

            with st.spinner("Lade Vergleichs-Daten..."):
                for sym in compare_symbols:
                    try:
                        df_cmp = analyzer_inst._fetcher.get_ohlcv(
                            sym, interval, min(lookback_days, 180)
                        )
                        compare_dfs[sym] = df_cmp
                    except Exception as e:
                        st.warning(f"{sym}: Nicht ladbar ({e})")

            if len(compare_dfs) >= 2:
                import plotly.graph_objects as go

                fig_cmp = go.Figure()
                for sym, df_cmp in compare_dfs.items():
                    # Normalisiert auf ersten Preis = 100 (Index-Performance)
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
                )
                st.plotly_chart(fig_cmp, use_container_width=True)
            elif compare_dfs:
                st.info("Mindestens 2 gültige Symbole für Vergleich nötig.")

    # ── Finaler Disclaimer ────────────────────────────────────────────────
    st.divider()
    st.info(ui_cfg["disclaimer_long"], icon="⚠️")
