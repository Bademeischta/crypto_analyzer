"""Wiederverwendbare Streamlit-UI-Komponenten.

Alle Komponenten sind zustandslos (keine st.session_state Schreibzugriffe hier).
Sie nehmen Daten entgegen und rendern HTML/Streamlit-Elemente.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from src.backtest.engine import BacktestResult
from src.models.predictor import PredictionResult
from src.models.evaluator import AggregatedMetrics


def render_disclaimer(disclaimer_text: str) -> None:
    """Rendert den permanenten Pflicht-Disclaimer.

    Dieser Disclaimer ist nicht wegklickbar und erscheint immer am Seitenanfang.

    Args:
        disclaimer_text: Disclaimer-Text aus der Konfiguration.
    """
    st.error(
        f"⚠️ **WICHTIGER HINWEIS**\n\n{disclaimer_text}",
        icon="⚠️",
    )


def render_signal_card(prediction: PredictionResult) -> None:
    """Rendert die KI-Signal-Karte mit Ampel-System.

    Args:
        prediction: PredictionResult aus dem Predictor.
    """
    if not prediction.show_signal:
        st.warning(
            f"**Kein klares Signal**\n\n{prediction.no_signal_reason}",
            icon="🟡",
        )
        return

    signal = prediction.direction_label
    emoji = prediction.direction_emoji
    conf = prediction.confidence
    conf_pct = f"{conf:.0%}" if conf is not None else "–"
    horizon = prediction.horizon_days

    color_map = {"BULLISH": "#1a7f37", "NEUTRAL": "#b08800", "BEARISH": "#c82538"}
    color = color_map.get(signal, "#888888")

    st.markdown(
        f"""
        <div style="
            background: {color}18;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        ">
            <div style="font-size: 3rem; line-height: 1;">{emoji}</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {color}; margin: 8px 0;">
                {signal}
            </div>
            <div style="font-size: 1rem; color: #666;">
                Modell-Konfidenz: <strong>{conf_pct}</strong>
            </div>
            <div style="font-size: 0.85rem; color: #888; margin-top: 4px;">
                Horizont: ~{horizon} Tage
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_volatility_badge(prediction: PredictionResult) -> None:
    """Rendert das Volatilitäts-Risiko-Badge.

    Args:
        prediction: PredictionResult.
    """
    vola = prediction.volatility_label
    color_map = {"NIEDRIG": "#1a7f37", "MITTEL": "#b08800", "HOCH": "#c82538"}
    color = color_map.get(vola, "#888888")
    bar_fill = {"NIEDRIG": "33%", "MITTEL": "66%", "HOCH": "100%"}.get(vola, "50%")

    st.markdown(
        f"""
        <div style="padding: 12px 0;">
            <div style="font-size: 0.85rem; color: #888; margin-bottom: 4px;">
                Erwartetes Volatilitäts-Regime
            </div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {color};">
                {vola}
            </div>
            <div style="
                background: #e0e0e0;
                border-radius: 4px;
                height: 8px;
                margin-top: 6px;
            ">
                <div style="
                    background: {color};
                    width: {bar_fill};
                    height: 8px;
                    border-radius: 4px;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(prediction: PredictionResult) -> None:
    """Rendert die Wahrscheinlichkeiten aller drei Klassen.

    Args:
        prediction: PredictionResult mit probabilities-Dict.
    """
    probs = prediction.probabilities
    labels = ["BULLISH", "NEUTRAL", "BEARISH"]
    colors = ["#1a7f37", "#b08800", "#c82538"]

    for label, color in zip(labels, colors):
        p = probs.get(label, 0.0)
        st.markdown(
            f"""
            <div style="margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between;
                            font-size: 0.8rem; color: #555; margin-bottom: 2px;">
                    <span>{label}</span>
                    <span><strong>{p:.0%}</strong></span>
                </div>
                <div style="background: #e8e8e8; border-radius: 4px; height: 10px;">
                    <div style="background: {color}; width: {p * 100:.1f}%;
                                height: 10px; border-radius: 4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Erstellt einen interaktiven Preischart mit EMA-Overlays und Volumen.

    Args:
        df: OHLCV-DataFrame. Benötigt open, high, low, close, volume.
            Optional: ema_fast_series, ema_mid_series für Overlays (werden ignoriert
            wenn nicht vorhanden – Chart zeigt nur OHLCV).
        symbol: Coin-Symbol für Titel.

    Returns:
        Plotly Figure.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{symbol} – Preis (OHLCV)", "Volumen"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLCV",
            increasing_line_color="#1a7f37",
            decreasing_line_color="#c82538",
        ),
        row=1,
        col=1,
    )

    # EMA-Overlays (wenn vorhanden)
    for col_name, color, label in [
        ("ema_fast_9", "#2196F3", "EMA 9"),
        ("ema_mid_21", "#FF9800", "EMA 21"),
        ("ema_long_50", "#9C27B0", "EMA 50"),
    ]:
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    name=label,
                    line=dict(color=color, width=1.2),
                    opacity=0.8,
                ),
                row=1,
                col=1,
            )

    # Bollinger Bands (wenn vorhanden)
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_upper"],
                name="BB Oben",
                line=dict(color="#607D8B", width=1, dash="dot"),
                opacity=0.5,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_lower"],
                name="BB Unten",
                fill="tonexty",
                fillcolor="rgba(96,125,139,0.08)",
                line=dict(color="#607D8B", width=1, dash="dot"),
                opacity=0.5,
            ),
            row=1, col=1,
        )

    # Volumen-Bars
    colors_vol = [
        "#1a7f37" if c >= o else "#c82538"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volumen",
            marker_color=colors_vol,
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=550,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    return fig


def render_rsi_chart(df: pd.DataFrame) -> go.Figure | None:
    """Rendert RSI-Chart mit Overbought/Oversold-Linien.

    Args:
        df: DataFrame mit rsi_14 (und optional rsi_7) Spalte.

    Returns:
        Plotly Figure oder None wenn RSI nicht vorhanden.
    """
    if "rsi_14" not in df.columns:
        return None

    fig = go.Figure()
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Überkauft (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Überverkauft (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi_14"],
        name="RSI(14)", line=dict(color="#2196F3", width=2),
    ))
    if "rsi_7" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi_7"],
            name="RSI(7)", line=dict(color="#FF9800", width=1.5, dash="dot"),
        ))

    fig.update_layout(
        height=200,
        showlegend=True,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    return fig


def render_macd_chart(df: pd.DataFrame) -> go.Figure | None:
    """Rendert MACD-Chart.

    Args:
        df: DataFrame mit macd, macd_signal, macd_diff Spalten.

    Returns:
        Plotly Figure oder None.
    """
    if "macd" not in df.columns:
        return None

    colors_hist = ["#1a7f37" if v >= 0 else "#c82538" for v in df["macd_diff"].fillna(0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=df["macd_diff"],
        name="Histogramm", marker_color=colors_hist, opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd"],
        name="MACD", line=dict(color="#2196F3", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd_signal"],
        name="Signal", line=dict(color="#FF9800", width=1.5),
    ))
    fig.add_hline(y=0, line_color="gray", opacity=0.5)

    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    return fig


def render_feature_importance_chart(importance: dict[str, float]) -> go.Figure:
    """Rendert Feature-Importance als horizontales Balkendiagramm.

    Args:
        importance: Dict feature_name -> Wichtigkeit (in %, 0-100).

    Returns:
        Plotly Figure.
    """
    if not importance:
        fig = go.Figure()
        fig.update_layout(height=200, title="Keine Feature-Importance-Daten")
        return fig

    # Top 15 Features
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = [item[0] for item in reversed(top)]
    values = [item[1] for item in reversed(top)]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color="#2196F3",
        opacity=0.8,
    ))
    fig.update_layout(
        height=max(300, len(labels) * 22),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Wichtigkeit (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=False),
    )
    return fig


def render_fear_greed_gauge(current_value: int, label: str) -> go.Figure:
    """Rendert den Fear & Greed Index als Tachometer-Chart.

    Args:
        current_value: Aktueller Wert (0-100).
        label: Textlabel (z.B. "Extreme Fear").

    Returns:
        Plotly Figure.
    """
    color = (
        "#c82538" if current_value < 25
        else "#FF9800" if current_value < 45
        else "#607D8B" if current_value < 55
        else "#4CAF50" if current_value < 75
        else "#1a7f37"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_value,
        title={"text": f"Fear & Greed Index<br><span style='font-size:0.8em'>{label}</span>"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "#ffebee"},
                {"range": [25, 45], "color": "#fff3e0"},
                {"range": [45, 55], "color": "#f5f5f5"},
                {"range": [55, 75], "color": "#e8f5e9"},
                {"range": [75, 100], "color": "#c8e6c9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": current_value,
            },
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_model_performance_badge(metrics: AggregatedMetrics) -> None:
    """Rendert den Modell-Performance-Badge mit ehrlicher Accuracy-Anzeige.

    Args:
        metrics: AggregatedMetrics aus dem Evaluator.
    """
    acc_pct = f"{metrics.avg_accuracy:.0%}"
    baseline_pct = f"{metrics.baseline_accuracy:.0%}"
    beats_pct = f"{metrics.beats_baseline_pct:.0%}"
    mcc = f"{metrics.avg_mcc:+.2f}"

    badge_color = "#1a7f37" if metrics.model_is_useful else "#b08800"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Historische Trefferquote",
            acc_pct,
            help="Wie oft lag das Modell in historischen Testperioden richtig?",
        )
    with col2:
        st.metric(
            "Baseline (immer NEUTRAL)",
            baseline_pct,
            help="Naive Strategie: Immer 'NEUTRAL' vorherzusagen. Das Modell sollte besser sein.",
        )
    with col3:
        delta_vs_base = metrics.avg_accuracy - metrics.baseline_accuracy
        st.metric(
            "Schlägt Baseline in",
            beats_pct,
            delta=f"{delta_vs_base:+.1%} vs. Baseline",
            help="Anteil der Testperioden wo das Modell besser als die Baseline war.",
        )
    with col4:
        st.metric(
            "MCC (Qualitäts-Score)",
            mcc,
            help="Matthews Correlation Coefficient: 0 = zufällig, +1 = perfekt, -1 = immer falsch.",
        )

    st.caption(metrics.disclaimer)


def render_market_data_row(market_data: dict[str, Any]) -> None:
    """Rendert die Marktdaten-Kennzahlen in einer Zeile.

    Args:
        market_data: Dict aus CoinGeckoFetcher.get_market_data().
    """
    def fmt_large(val: float | None, suffix: str = "") -> str:
        if val is None:
            return "–"
        if val >= 1e12:
            return f"${val/1e12:.2f}T{suffix}"
        if val >= 1e9:
            return f"${val/1e9:.2f}B{suffix}"
        if val >= 1e6:
            return f"${val/1e6:.2f}M{suffix}"
        return f"${val:,.0f}{suffix}"

    cols = st.columns(5)
    data_points = [
        ("Market Cap", fmt_large(market_data.get("market_cap_usd")), "Gesamtmarktwert aller im Umlauf befindlichen Coins"),
        ("24h Volumen", fmt_large(market_data.get("volume_24h_usd")), "Gesamthandelsvol. der letzten 24 Stunden"),
        ("24h Änderung", (
            f"{market_data['price_change_24h_pct']:.2f}%"
            if market_data.get("price_change_24h_pct") is not None
            else "–"
        ), "Preisveränderung in den letzten 24 Stunden"),
        ("7d Änderung", (
            f"{market_data['price_change_7d_pct']:.2f}%"
            if market_data.get("price_change_7d_pct") is not None
            else "–"
        ), "Preisveränderung der letzten 7 Tage"),
        ("CMC Rank", (
            f"#{market_data['rank']}"
            if market_data.get("rank") is not None
            else "–"
        ), "Position im CoinMarketCap-Ranking nach Market Cap"),
    ]

    for col, (label, value, help_text) in zip(cols, data_points):
        with col:
            delta = None
            if "Änderung" in label:
                try:
                    num = float(value.replace("%", ""))
                    delta = f"{num:+.2f}%"
                    value = f"{num:.2f}%"
                except (ValueError, AttributeError):
                    pass
            st.metric(label, value, help=help_text)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST KOMPONENTEN
# ─────────────────────────────────────────────────────────────────────────────

def render_backtest_metrics(bt: BacktestResult) -> None:
    """Zeigt die wichtigsten Backtest-Kennzahlen als Metriken-Grid."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta_color = "normal" if bt.total_return_pct >= 0 else "inverse"
        st.metric(
            "Strategie-Rendite",
            f"{bt.total_return_pct:+.1f}%",
            help="Gesamtrendite der KI-Strategie im Backtest-Zeitraum",
        )
    with c2:
        st.metric(
            "Buy & Hold",
            f"{bt.bnh_return_pct:+.1f}%",
            help="Passive Buy-and-Hold-Strategie als Vergleich",
        )
    with c3:
        alpha_sign = "+" if bt.alpha_pct >= 0 else ""
        st.metric(
            "Alpha vs. B&H",
            f"{alpha_sign}{bt.alpha_pct:.1f}pp",
            help="Überrendite der KI-Strategie gegenüber Buy & Hold",
        )
    with c4:
        st.metric(
            "Max. Drawdown",
            f"{bt.max_drawdown_pct:.1f}%",
            help="Maximaler Peak-to-Trough-Verlust",
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric(
            "Sharpe Ratio",
            f"{bt.sharpe_ratio:.2f}",
            help="Annualisiertes Rendite/Risiko-Verhältnis (vereinfacht)",
        )
    with c6:
        st.metric(
            "Markt-Exposure",
            f"{bt.exposure_pct:.0f}%",
            help="Anteil der Tage mit aktiver Long-Position",
        )
    with c7:
        st.metric(
            "Anzahl Trades",
            str(bt.n_trades),
            help="Positionswechsel im Backtest-Zeitraum",
        )
    with c8:
        st.metric(
            "Win Rate",
            f"{bt.win_rate_pct:.0f}%",
            help="Anteil profitabler Long-Tage",
        )


def render_backtest_chart(bt: BacktestResult, symbol: str) -> go.Figure:
    """Erstellt den Backtest-Performance-Chart (Strategie vs. Buy & Hold)."""
    df = bt.series

    fig = go.Figure()

    # Strategie-Linie
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["portfolio"],
        name="KI-Strategie",
        line=dict(color="#2196F3", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: %{y:.1f}<extra></extra>",
    ))

    # Buy & Hold Linie
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["benchmark"],
        name="Buy & Hold",
        line=dict(color="#FF9800", width=2, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>B&H: %{y:.1f}<extra></extra>",
    ))

    # Long-Perioden als grüne Füllung
    long_mask = df["position"].shift(1).fillna(0) == 1
    if long_mask.any():
        long_df = df[long_mask]
        fig.add_trace(go.Scatter(
            x=long_df.index,
            y=long_df["portfolio"],
            name="Long-Position",
            mode="markers",
            marker=dict(color="#1a7f37", size=4, opacity=0.4),
            hoverinfo="skip",
        ))

    # Baseline bei 100
    fig.add_hline(y=100, line_dash="dot", line_color="gray", line_width=1, opacity=0.5)

    final_port = df["portfolio"].iloc[-1]
    final_bnh = df["benchmark"].iloc[-1]
    alpha_label = f"Alpha: {bt.alpha_pct:+.1f}pp"
    color = "#1a7f37" if bt.alpha_pct >= 0 else "#c82538"

    fig.update_layout(
        title=dict(
            text=f"{symbol} Backtest ({bt.period_label})  ·  {alpha_label}",
            font=dict(size=14),
        ),
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Portfolio-Wert (Start = 100)",
        xaxis_title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )
    return fig


def render_screener_table(screener_rows: list[dict[str, Any]]) -> None:
    """Rendert die Screener-Tabelle mit farbigen Signalen."""
    if not screener_rows:
        st.info("Keine Screener-Daten verfügbar.")
        return

    df = pd.DataFrame(screener_rows)

    # Signal-Spalte mit Emoji
    def format_signal(row):
        sig = row.get("signal", "–")
        emoji_map = {"BULLISH": "🟢 BULLISH", "BEARISH": "🔴 BEARISH", "NEUTRAL": "🟡 NEUTRAL"}
        return emoji_map.get(sig, "⚪ –")

    df["Signal"] = df.apply(format_signal, axis=1)

    display_cols = {
        "symbol": "Symbol",
        "rsi_14": "RSI(14)",
        "price_change_7d": "7T Änderung",
        "Signal": "KI-Signal",
        "confidence": "Konfidenz",
        "adx": "ADX",
    }

    display_df = pd.DataFrame()
    for src, label in display_cols.items():
        if src in df.columns:
            display_df[label] = df[src]

    # Formatierung
    if "RSI(14)" in display_df.columns:
        display_df["RSI(14)"] = pd.to_numeric(display_df["RSI(14)"], errors="coerce").round(1)
    if "7T Änderung" in display_df.columns:
        display_df["7T Änderung"] = pd.to_numeric(display_df["7T Änderung"], errors="coerce")
        display_df["7T Änderung"] = display_df["7T Änderung"].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "–"
        )
    if "ADX" in display_df.columns:
        display_df["ADX"] = pd.to_numeric(display_df["ADX"], errors="coerce").round(1)
    if "Konfidenz" in display_df.columns:
        display_df["Konfidenz"] = display_df["Konfidenz"].apply(
            lambda x: f"{x:.0%}" if pd.notna(x) and x is not None else "–"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(400, 40 + len(display_df) * 35),
    )
