"""Wiederverwendbare UI-Bausteine: Plotly-Figuren und kleine Streamlit-Renderer.

Alle Komponenten sind zustandslos (keine st.session_state-Schreibzugriffe).
Figuren setzen bewusst keine Hintergrundfarben – so passen sie sich über
Streamlits Plotly-Theme automatisch an Light- und Dark-Mode an.
"""

from __future__ import annotations

import html
import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.features.technical import describe_feature
from src.models.backtest import BacktestResult
from src.models.evaluator import AggregatedMetrics
from src.models.predictor import PredictionResult

# Farbpalette (funktioniert auf hellem und dunklem Hintergrund)
UP = "#16a34a"
DOWN = "#dc2626"
NEUTRAL = "#ca8a04"
BLUE = "#2563eb"
ORANGE = "#f59e0b"
PURPLE = "#8b5cf6"
TEAL = "#0d9488"
GRAY = "#64748b"
PINK = "#db2777"
SERIES = [BLUE, ORANGE, PURPLE, TEAL, UP, DOWN, PINK, GRAY]

SIGNAL_COLORS = {"BULLISH": UP, "NEUTRAL": NEUTRAL, "BEARISH": DOWN}
VOLA_COLORS = {"NIEDRIG": UP, "MITTEL": NEUTRAL, "HOCH": DOWN}
VERDICT_STYLE = {
    "signifikant": ("✅", UP),
    "schwach": ("ℹ️", NEUTRAL),
    "keine Vorhersagekraft": ("⚠️", DOWN),
}


def _layout(fig: go.Figure, height: int, **overrides: Any) -> go.Figure:
    settings: dict[str, Any] = {
        "height": height,
        "margin": dict(l=8, r=8, t=36, b=8),
        "legend": dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        "hovermode": "x unified",
    }
    fig.update_layout(**(settings | overrides))
    return fig


# ===========================================================================
# Formatierung
# ===========================================================================

def fmt_usd(value: float | None) -> str:
    """Kompakte USD-Darstellung ($1.23T, $4.5B, $0.00001234)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    v = float(value)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(v) >= limit:
            return f"${v / limit:,.2f}{suffix}"
    if abs(v) >= 1000:
        return f"${v:,.0f}"
    if abs(v) >= 1:
        return f"${v:,.2f}"
    if v == 0:
        return "$0"
    # Memecoins: signifikante Stellen statt 0.00
    digits = max(2, -int(math.floor(math.log10(abs(v)))) + 3)
    return f"${v:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1, signed: bool = True, ratio: bool = True) -> str:
    """Prozent-Darstellung. ``ratio=True``: 0.05 → '+5.0%'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    v = float(value) * (100 if ratio else 1)
    return f"{v:+.{digits}f}%" if signed else f"{v:.{digits}f}%"


def escape_md(text: str) -> str:
    """Neutralisiert Markdown/LaTeX in Fremdtexten (z.B. Reddit-Titel mit '$')."""
    out = html.escape(str(text))
    for ch in "\\`*_{}[]()#+!|$~>":
        out = out.replace(ch, "\\" + ch)
    return out


# ===========================================================================
# Karten & Badges (HTML)
# ===========================================================================

def render_disclaimer(text: str) -> None:
    """Kompakter, permanent sichtbarer Pflicht-Hinweis."""
    st.warning(text, icon="⚠️")


def render_signal_card(prediction: PredictionResult) -> None:
    """KI-Signal-Karte mit Horizont, Schwellen und historischer Einordnung."""
    if not prediction.show_signal:
        st.markdown(
            f"""
            <div style="border:2px dashed {GRAY};border-radius:14px;padding:18px;text-align:center;">
              <div style="font-size:2.4rem;line-height:1;">⚪</div>
              <div style="font-size:1.4rem;font-weight:700;margin:6px 0;">Kein klares Signal</div>
              <div style="font-size:.85rem;opacity:.8;">{html.escape(prediction.no_signal_reason)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    color = SIGNAL_COLORS.get(prediction.direction_label, GRAY)
    hist = prediction.historical or {}
    hist_line = ""
    if hist.get("n_signals"):
        hit = hist.get("direction_hit_rate", hist.get("hit_rate"))
        hit_txt = fmt_pct(hit, 0, signed=False) if hit is not None and not math.isnan(hit) else "–"
        hist_line = (
            f"<div style='font-size:.8rem;opacity:.85;margin-top:8px;'>"
            f"Historisch ({hist['n_signals']} vergleichbare Signale): Kurs lief in {hit_txt} der Fälle "
            f"in Signalrichtung, Ø {fmt_pct(hist.get('avg_return'))} in {html.escape(prediction.horizon_text)}</div>"
        )
    st.markdown(
        f"""
        <div style="background:{color}14;border:2px solid {color};border-radius:14px;padding:18px;text-align:center;">
          <div style="font-size:2.6rem;line-height:1;">{prediction.direction_emoji}</div>
          <div style="font-size:1.8rem;font-weight:800;color:{color};margin:6px 0;">{prediction.direction_label}</div>
          <div style="font-size:.95rem;">Wahrscheinlichkeit: <b>{prediction.confidence:.0%}</b>
            · Horizont: <b>{html.escape(prediction.horizon_text)}</b></div>
          {hist_line}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(probabilities: dict[str, float], colors: dict[str, str], order: list[str]) -> None:
    """Horizontale Wahrscheinlichkeitsbalken."""
    rows = []
    for label in order:
        p = float(probabilities.get(label, 0.0))
        c = colors.get(label, GRAY)
        rows.append(
            f"<div style='margin-bottom:6px;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:.8rem;'>"
            f"<span>{html.escape(label)}</span><b>{p:.0%}</b></div>"
            f"<div style='background:{GRAY}33;border-radius:4px;height:9px;'>"
            f"<div style='background:{c};width:{p * 100:.1f}%;height:9px;border-radius:4px;'></div></div></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_verdict_badge(verdict: str, label: str) -> None:
    """Kleines farbiges Badge für das Modell-Urteil."""
    icon, color = VERDICT_STYLE.get(verdict, ("❔", GRAY))
    st.markdown(
        f"<span style='background:{color}1f;color:{color};border:1px solid {color};"
        f"border-radius:999px;padding:2px 10px;font-size:.8rem;font-weight:600;'>"
        f"{icon} {html.escape(label)}: {html.escape(verdict)}</span>",
        unsafe_allow_html=True,
    )


def render_market_header(market: dict[str, Any], closes: pd.Series, bars_per_day: float) -> None:
    """Kennzahlen-Zeile: Preis, Änderungen, Market Cap, Volumen, Rang, ATH-Abstand."""
    price = market.get("price") or (float(closes.iloc[-1]) if len(closes) else None)
    change_7d = market.get("price_change_7d_pct")
    if change_7d is None and len(closes) > 7 * bars_per_day:
        change_7d = (closes.iloc[-1] / closes.iloc[-1 - int(7 * bars_per_day)] - 1) * 100
    spark = closes.iloc[-int(30 * bars_per_day):].round(10).tolist() if len(closes) else None

    cols = st.columns([1.35, 1, 1, 1, 0.8, 1])
    cols[0].metric("Preis", fmt_usd(price), fmt_pct(market.get("price_change_24h_pct"), 2, ratio=False),
                   chart_data=spark, chart_type="area", border=True, help="Live-Preis (Binance) · Δ 24h")
    cols[1].metric("7 Tage", fmt_pct(change_7d, 2, ratio=False), border=True)
    cols[2].metric("Market Cap", fmt_usd(market.get("market_cap_usd")), border=True,
                   help="CoinGecko · FDV: " + fmt_usd(market.get("fdv_usd")))
    cols[3].metric("Volumen 24h", fmt_usd(market.get("volume_24h_usd")), border=True)
    cols[4].metric("Rang", f"#{market['rank']}" if market.get("rank") else "–", border=True,
                   help="Market-Cap-Rang laut CoinGecko")
    ath = market.get("ath_change_pct")
    cols[5].metric("Abstand ATH", fmt_pct(ath, 1, ratio=False) if ath is not None else "–", border=True,
                   help=f"Allzeithoch: {fmt_usd(market.get('ath_usd'))}")


# ===========================================================================
# Kurs- und Indikator-Charts
# ===========================================================================

def historical_signals(oof: pd.DataFrame | None, threshold: float) -> pd.DataFrame:
    """Out-of-Sample-Signale über der Konfidenzschwelle (für Chart-Marker).

    Returns:
        DataFrame mit signal ("BULLISH"/"BEARISH"), confidence und fwd_ret je Zeitpunkt.
    """
    if oof is None or oof.empty:
        return pd.DataFrame(columns=["signal", "confidence", "fwd_ret"])
    proba = oof[["p_down", "p_neutral", "p_up"]].to_numpy()
    best = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    mask = (conf >= threshold) & (best != 1)
    return pd.DataFrame(
        {
            "signal": np.array(["BEARISH", "NEUTRAL", "BULLISH"])[best][mask],
            "confidence": conf[mask],
            "fwd_ret": oof["fwd_ret"].to_numpy()[mask],
        },
        index=oof.index[mask],
    )


def price_chart(
    df: pd.DataFrame,
    symbol: str,
    features_cfg: dict[str, Any],
    signals: pd.DataFrame | None = None,
    log_scale: bool = False,
) -> go.Figure:
    """Candlestick + EMAs + Bollinger + Volumen, optional mit historischen Modellsignalen."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=symbol,
        increasing_line_color=UP, decreasing_line_color=DOWN, showlegend=False,
    ), row=1, col=1)

    if {"bb_upper", "bb_lower"} <= set(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="Bollinger", line=dict(color=GRAY, width=1, dash="dot"),
                                 legendgroup="bb"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="Bollinger", fill="tonexty",
                                 fillcolor="rgba(100,116,139,0.08)", line=dict(color=GRAY, width=1, dash="dot"),
                                 legendgroup="bb", showlegend=False), row=1, col=1)

    ema_specs = [
        (f"ema_fast_{features_cfg['ema_fast']}", BLUE, f"EMA {features_cfg['ema_fast']}"),
        (f"ema_mid_{features_cfg['ema_mid']}", ORANGE, f"EMA {features_cfg['ema_mid']}"),
        (f"ema_long_{features_cfg['ema_long_short']}", PURPLE, f"EMA {features_cfg['ema_long_short']}"),
        (f"ema_long_{features_cfg['ema_long_long']}", TEAL, f"EMA {features_cfg['ema_long_long']}"),
    ]
    for col, color, label in ema_specs:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=label, line=dict(color=color, width=1.3)), row=1, col=1)

    if signals is not None and not signals.empty:
        visible = signals[signals.index >= df.index[0]]
        for label, marker, color, y_col in (("BULLISH", "triangle-up", UP, "low"), ("BEARISH", "triangle-down", DOWN, "high")):
            pts = visible[visible["signal"] == label]
            if pts.empty:
                continue
            y = df[y_col].reindex(pts.index) * (0.97 if label == "BULLISH" else 1.03)
            fig.add_trace(go.Scatter(
                x=pts.index, y=y, mode="markers", name=f"Signal {label.lower()}",
                marker=dict(symbol=marker, size=9, color=color, line=dict(width=1, color="white")),
                customdata=np.stack([pts["confidence"] * 100, pts["fwd_ret"] * 100], axis=1),
                hovertemplate="%{x}<br>P=%{customdata[0]:.0f}% · danach %{customdata[1]:+.1f}%<extra></extra>",
            ), row=1, col=1)

    vol_colors = np.where(df["close"] >= df["open"], UP, DOWN)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volumen", marker_color=vol_colors, opacity=0.6,
                         showlegend=False), row=2, col=1)

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(type="log" if log_scale else "linear", row=1, col=1)
    fig.update_yaxes(title_text="Vol.", row=2, col=1)
    return _layout(fig, 600)


def indicator_panel(df: pd.DataFrame, features_cfg: dict[str, Any]) -> go.Figure:
    """RSI, MACD, ADX/DI und Stochastic in einem gemeinsamen Panel."""
    rsi_s, rsi_l = f"rsi_{features_cfg['rsi_short_window']}", f"rsi_{features_cfg['rsi_long_window']}"
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=("RSI", "MACD", "ADX / DI", "Stochastic"),
    )
    if rsi_l in df:
        fig.add_trace(go.Scatter(x=df.index, y=df[rsi_l], name=f"RSI {features_cfg['rsi_long_window']}",
                                 line=dict(color=BLUE, width=1.8)), row=1, col=1)
    if rsi_s in df:
        fig.add_trace(go.Scatter(x=df.index, y=df[rsi_s], name=f"RSI {features_cfg['rsi_short_window']}",
                                 line=dict(color=ORANGE, width=1, dash="dot")), row=1, col=1)
    for level, color in ((70, DOWN), (30, UP)):
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.5, row=1, col=1)

    if "macd" in df:
        hist = df["macd_diff"]
        fig.add_trace(go.Bar(x=df.index, y=hist, name="MACD-Hist.", marker_color=np.where(hist >= 0, UP, DOWN),
                             opacity=0.6), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color=BLUE, width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color=ORANGE, width=1.2)),
                      row=2, col=1)

    if "adx" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["adx"], name="ADX", line=dict(color=PURPLE, width=1.8)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["adx_pos"], name="+DI", line=dict(color=UP, width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["adx_neg"], name="−DI", line=dict(color=DOWN, width=1)), row=3, col=1)
        fig.add_hline(y=25, line_dash="dot", line_color=GRAY, opacity=0.6, row=3, col=1)

    if "stoch_k" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["stoch_k"], name="%K", line=dict(color=TEAL, width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["stoch_d"], name="%D", line=dict(color=ORANGE, width=1)), row=4, col=1)
        for level in (80, 20):
            fig.add_hline(y=level, line_dash="dash", line_color=GRAY, opacity=0.5, row=4, col=1)

    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    fig.update_layout(showlegend=False)
    return _layout(fig, 720)


def indicator_summary(df: pd.DataFrame, features_cfg: dict[str, Any]) -> pd.DataFrame:
    """Aktuelle Indikatorwerte mit kurzer, regelbasierter Einordnung."""
    last = df.iloc[-1]
    rsi_col = f"rsi_{features_cfg['rsi_long_window']}"

    def val(col: str) -> float:
        return float(last[col]) if col in last and pd.notna(last[col]) else float("nan")

    rsi = val(rsi_col)
    adx = val("adx")
    di = val("di_diff")
    bb = val("bb_pct")
    macd_h = val("macd_hist_pct")
    vol_ratio = val("volume_ratio")
    ema_long = val("close_vs_ema_long")
    taker = val("taker_buy_ratio")
    rows = [
        ("RSI (14)", f"{rsi:.1f}", "überkauft" if rsi > 70 else "überverkauft" if rsi < 30 else "neutral"),
        ("MACD-Histogramm", f"{macd_h * 100:+.3f}% vom Kurs", "bullisches Momentum" if macd_h > 0 else "bärisches Momentum"),
        ("ADX", f"{adx:.1f}", ("starker Trend" if adx > 25 else "kein klarer Trend") + (" ↑" if di > 0 else " ↓")),
        ("Bollinger %B", f"{bb:.2f}", "am oberen Band" if bb > 0.95 else "am unteren Band" if bb < 0.05 else "innerhalb der Bänder"),
        (f"Kurs vs. EMA {features_cfg['ema_long_short']}", fmt_pct(ema_long), "darüber" if ema_long > 0 else "darunter"),
        ("Volumen vs. Ø", f"{vol_ratio:.2f}×", "erhöht" if vol_ratio > 1.5 else "niedrig" if vol_ratio < 0.6 else "normal"),
        ("Hist. Volatilität (ann.)", fmt_pct(val("hist_vol"), 0, signed=False), ""),
        ("Taker-Buy-Anteil", fmt_pct(taker + 0.5 if not math.isnan(taker) else None, 1, signed=False),
         "Käufer aggressiver" if taker > 0.02 else "Verkäufer aggressiver" if taker < -0.02 else "ausgeglichen"),
    ]
    return pd.DataFrame(rows, columns=["Indikator", "Wert", "Einordnung"])


# ===========================================================================
# Modell-Transparenz
# ===========================================================================

def feature_importance_chart(importance: dict[str, float], top_n: int = 15) -> go.Figure:
    """Permutation-Importance: stärkste und schädlichste Features."""
    items = list(importance.items())
    top = items[:top_n]
    worst = [kv for kv in items[-5:] if kv[1] < 0 and kv not in top]
    data = list(reversed(top + worst))
    fig = go.Figure(go.Bar(
        x=[v for _, v in data],
        y=[describe_feature(k) for k, _ in data],
        orientation="h",
        marker_color=[BLUE if v >= 0 else DOWN for _, v in data],
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(title_text="Anteil an der Out-of-Sample-Wichtigkeit (%)")
    return _layout(fig, max(320, len(data) * 24), hovermode="closest")


def fold_chart(metrics: AggregatedMetrics) -> go.Figure:
    """Log-Loss je Walk-Forward-Fold: Modell vs. Prior-Baseline (niedriger = besser)."""
    ft = pd.DataFrame(metrics.fold_table)
    labels = [pd.Timestamp(s).strftime("%Y-%m-%d") for s in ft["start"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=ft["log_loss_prior"], name="Baseline (Klassenhäufigkeit)", marker_color=GRAY, opacity=0.5))
    fig.add_trace(go.Bar(x=labels, y=ft["log_loss"], name="Modell", marker_color=BLUE))
    lo = float(min(ft["log_loss"].min(), ft["log_loss_prior"].min()))
    hi = float(max(ft["log_loss"].max(), ft["log_loss_prior"].max()))
    pad = max(0.01, (hi - lo) * 0.15)
    fig.update_yaxes(title_text="Log-Loss", range=[lo - pad, hi + pad])
    fig.update_xaxes(title_text="Beginn der Testperiode")
    return _layout(fig, 320, barmode="group")


def calibration_chart(calibration: list[dict[str, float]]) -> go.Figure:
    """Zuverlässigkeitsdiagramm: vorhergesagte Konfidenz vs. tatsächliche Trefferquote."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1 / 3, 1], y=[1 / 3, 1], mode="lines", name="perfekt kalibriert",
                             line=dict(color=GRAY, dash="dash")))
    if calibration:
        cal = pd.DataFrame(calibration)
        fig.add_trace(go.Scatter(
            x=cal["predicted"], y=cal["observed"], mode="lines+markers", name="Modell",
            marker=dict(size=np.clip(np.sqrt(cal["count"]) * 1.5, 6, 22), color=BLUE),
            customdata=cal["count"], hovertemplate="Konfidenz %{x:.0%} → Treffer %{y:.0%} (n=%{customdata})<extra></extra>",
        ))
    fig.update_xaxes(title_text="Vorhergesagte Wahrscheinlichkeit", tickformat=".0%", range=[0.3, 1])
    fig.update_yaxes(title_text="Tatsächliche Trefferquote", tickformat=".0%", range=[0, 1])
    return _layout(fig, 320, hovermode="closest")


def confusion_chart(confusion: list[list[int]], labels: list[str]) -> go.Figure:
    """Konfusionsmatrix als Heatmap (Zeilen = tatsächlich, Spalten = vorhergesagt)."""
    z = np.array(confusion, dtype=float)
    row_pct = z / np.maximum(z.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(
        z=row_pct, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1, showscale=False,
        text=[[f"{int(c)}<br>{p:.0%}" for c, p in zip(row, prow)] for row, prow in zip(z, row_pct)],
        texttemplate="%{text}", hovertemplate="tatsächlich %{y} → vorhergesagt %{x}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Vorhergesagt")
    fig.update_yaxes(title_text="Tatsächlich", autorange="reversed")
    return _layout(fig, 320, hovermode="closest")


def signal_stats_table(stats: dict[str, dict[str, float]], unconditional: float) -> pd.DataFrame:
    """Tabelle: Was passierte nach Signalen oberhalb der Konfidenzschwelle?"""
    rows = []
    for name in ("BULLISH", "BEARISH", "NEUTRAL"):
        s = stats.get(name, {})
        if not s.get("n_signals"):
            continue
        rows.append({
            "Signal": name,
            "Anzahl": str(int(s["n_signals"])),
            "Anteil Zeit": fmt_pct(s.get("coverage"), 1, signed=False),
            "Label korrekt": fmt_pct(s.get("hit_rate"), 0, signed=False),
            "Kurs in Richtung": fmt_pct(s.get("direction_hit_rate"), 0, signed=False),
            "Ø Return danach": fmt_pct(s.get("avg_return"), 2),
        })
    rows.append({"Signal": "alle Zeitpunkte", "Anzahl": "–", "Anteil Zeit": "100%", "Label korrekt": "–",
                 "Kurs in Richtung": "–", "Ø Return danach": fmt_pct(unconditional, 2)})
    return pd.DataFrame(rows)


# ===========================================================================
# Backtest
# ===========================================================================

def equity_chart(bt: BacktestResult, log_scale: bool = False) -> go.Figure:
    """Equity-Kurven Strategie vs. Buy & Hold plus Drawdown."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    for col, color in (("Strategie", BLUE), ("Buy & Hold", GRAY)):
        fig.add_trace(go.Scatter(x=bt.equity.index, y=bt.equity[col], name=col, line=dict(color=color, width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.drawdown.index, y=bt.drawdown[col], name=f"Drawdown {col}", showlegend=False,
                                 fill="tozeroy", line=dict(color=color, width=1)), row=2, col=1)
    exposure = bt.position.replace(0, np.nan)
    fig.add_trace(go.Scatter(
        x=exposure.index, y=np.where(exposure.notna(), bt.equity["Strategie"], np.nan),
        mode="markers", marker=dict(size=3, color=np.where(exposure > 0, UP, DOWN)), name="investiert",
    ), row=1, col=1)
    fig.update_yaxes(title_text="Wert (Start = 1)", type="log" if log_scale else "linear", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    return _layout(fig, 520)


def backtest_table(bt: BacktestResult) -> pd.DataFrame:
    """Kennzahlen Strategie vs. Buy & Hold nebeneinander."""
    spec = [
        ("Gesamtrendite", "total_return", "pct"),
        ("CAGR", "cagr", "pct"),
        ("Volatilität (ann.)", "ann_volatility", "pct"),
        ("Sharpe Ratio", "sharpe", "num"),
        ("Sortino Ratio", "sortino", "num"),
        ("Max. Drawdown", "max_drawdown", "dd"),
        ("Calmar Ratio", "calmar", "num"),
    ]

    def fmt(v: float | None, kind: str) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "–"
        if kind == "dd":
            return fmt_pct(v, signed=False)
        return fmt_pct(v) if kind == "pct" else f"{v:.2f}"

    rows = [{"Kennzahl": label, "Strategie": fmt(bt.strategy.get(key), kind), "Buy & Hold": fmt(bt.buy_hold.get(key), kind)}
            for label, key, kind in spec]
    s = bt.strategy
    rows += [
        {"Kennzahl": "Zeit investiert", "Strategie": fmt_pct(s.get("exposure"), 0, signed=False), "Buy & Hold": "100%"},
        {"Kennzahl": "Anzahl Trades", "Strategie": str(s.get("n_trades", 0)), "Buy & Hold": "1"},
        {"Kennzahl": "Gewinn-Trades", "Strategie": fmt(s.get("win_rate"), "pct").lstrip("+"), "Buy & Hold": "–"},
        {"Kennzahl": "Ø Trade", "Strategie": fmt(s.get("avg_trade"), "pct"), "Buy & Hold": "–"},
        {"Kennzahl": "Kosten gesamt", "Strategie": fmt_pct(s.get("costs_paid"), 2, signed=False), "Buy & Hold": "–"},
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# Sentiment
# ===========================================================================

def fear_greed_gauge(value: int, label: str) -> go.Figure:
    """Fear & Greed Index als Tachometer."""
    color = DOWN if value < 25 else ORANGE if value < 46 else GRAY if value < 55 else "#65a30d" if value < 76 else UP
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": html.escape(label), "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "rgba(220,38,38,.18)"},
                {"range": [25, 46], "color": "rgba(245,158,11,.15)"},
                {"range": [46, 55], "color": "rgba(100,116,139,.12)"},
                {"range": [55, 76], "color": "rgba(101,163,13,.15)"},
                {"range": [76, 100], "color": "rgba(22,163,74,.2)"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=24, r=24, t=48, b=8))
    return fig


def fear_greed_history_chart(history: list[dict[str, Any]]) -> go.Figure:
    """Verlauf des Fear & Greed Index."""
    h = pd.DataFrame(history)
    fig = go.Figure(go.Scatter(x=pd.to_datetime(h["date"]), y=h["value"], mode="lines", fill="tozeroy",
                               line=dict(color=BLUE, width=1.5), name="Fear & Greed"))
    for level, color in ((25, DOWN), (75, UP)):
        fig.add_hline(y=level, line_dash="dot", line_color=color, opacity=0.6)
    fig.update_yaxes(range=[0, 100])
    return _layout(fig, 260)


# ===========================================================================
# Vergleich
# ===========================================================================

def comparison_chart(normalized: pd.DataFrame) -> go.Figure:
    """Relative Performance (Start = 100)."""
    fig = go.Figure()
    for i, sym in enumerate(normalized.columns):
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[sym], name=sym, mode="lines",
                                 line=dict(color=SERIES[i % len(SERIES)], width=2)))
    fig.add_hline(y=100, line_dash="dot", line_color=GRAY)
    fig.update_yaxes(title_text="Index (Start = 100)")
    return _layout(fig, 420)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Korrelationsmatrix der Returns."""
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        text=np.round(corr.values, 2), texttemplate="%{text}", showscale=False,
    ))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 60 + 50 * len(corr), hovermode="closest")
