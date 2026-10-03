"""Technische Indikatoren – reine pandas/numpy-Implementierung.

Keine externe Bibliothek (kein ta, kein pandas-ta, kein ta-lib).

Zwei Arten von Spalten:
  - **Chart-Spalten** (absolute Preisniveaus wie EMA, Bollinger-Bänder, MACD in
    USD): für die Visualisierung, NICHT als ML-Feature geeignet, da nicht
    stationär – ein Modell würde sonst "BTC kostet 60k" lernen statt Muster.
  - **Feature-Spalten**: normalisierte, stationäre Größen (Verhältnisse,
    Oszillatoren, Returns), die über Zeit und Coins vergleichbar sind.

Alle Indikatoren verwenden ausschließlich Daten bis einschließlich der
aktuellen Kerze (kein Lookahead). Glättungen nutzen ``min_periods``, damit die
Einschwingphase als NaN markiert wird statt verfälschte Werte zu liefern.

Implementiert:
  Momentum:    RSI (Wilder), Stochastic %K/%D
  Trend:       EMA, MACD, ADX/DI (Wilder), Kaufman Efficiency Ratio
  Volatilität: ATR (Wilder), Bollinger Bands, historische & Garman-Klass-Volatilität
  Volumen:     OBV-Steigung, Volumen-Ratio/Z-Score, VWAP-Abstand, Taker-Buy-Ratio
  Struktur:    Log-Returns, Return-Z-Score, Abstand zu Hoch/Tief
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import Timeframe

logger = logging.getLogger(__name__)

# Rohdaten-Spalten (nie Features)
BASE_COLUMNS: frozenset[str] = frozenset({
    "open", "high", "low", "close", "volume", "quote_volume", "num_trades", "taker_buy_base",
})

# Lesbare Beschreibungen für die Modell-Transparenz (Präfix-Match)
FEATURE_LABELS: dict[str, str] = {
    "rsi_": "RSI",
    "stoch_k": "Stochastic %K",
    "stoch_d": "Stochastic %D",
    "macd_pct": "MACD (relativ zum Kurs)",
    "macd_hist_pct": "MACD-Histogramm (relativ)",
    "ema_cross_fast_mid": "EMA-Cross schnell/mittel",
    "ema_cross_long": "EMA-Cross 50/200",
    "close_vs_ema_mid": "Abstand Kurs ↔ EMA mittel",
    "close_vs_ema_long": "Abstand Kurs ↔ EMA 50",
    "adx": "ADX (Trendstärke)",
    "di_diff": "+DI − −DI (Trendrichtung)",
    "efficiency_ratio": "Efficiency Ratio (Trendsauberkeit)",
    "atr_pct": "ATR in % vom Kurs",
    "bb_width": "Bollinger-Bandbreite",
    "bb_pct": "Bollinger %B",
    "hist_vol": "Historische Volatilität (ann.)",
    "gk_vol": "Garman-Klass-Volatilität (ann.)",
    "vol_ratio": "Volatilitäts-Ratio kurz/lang",
    "volume_ratio": "Volumen / Ø-Volumen",
    "volume_z": "Volumen-Z-Score",
    "obv_slope": "OBV-Steigung (Kaufdruck)",
    "vwap_dist": "Abstand zum VWAP",
    "taker_buy_ratio": "Taker-Buy-Anteil (Orderflow)",
    "ret_z": "Return-Z-Score",
    "ret_": "Log-Return",
    "dist_high": "Abstand zum N-Kerzen-Hoch",
    "dist_low": "Abstand zum N-Kerzen-Tief",
    "corr_": "Korrelation",
    "beta_": "Beta",
    "rel_strength_": "Relative Stärke",
    "fng_": "Fear & Greed",
}


def describe_feature(name: str) -> str:
    """Menschenlesbare Beschreibung eines Feature-Namens."""
    for prefix, label in FEATURE_LABELS.items():
        if name.startswith(prefix):
            suffix = name[len(prefix):].strip("_")
            if not suffix or not prefix.endswith("_"):
                return label
            if prefix in ("corr_", "beta_", "rel_strength_"):
                suffix = suffix.upper()
            elif prefix == "fng_":
                suffix = {"value": "Wert", "change_7d": "Δ 7 Tage"}.get(suffix, suffix)
            elif prefix == "ret_":
                suffix = f"{suffix} Kerzen"
            return f"{label} ({suffix})"
    return name


class TechnicalIndicators:
    """Berechnet Chart-Indikatoren und stationäre ML-Features.

    Args:
        config: Vollständiges config-Dict (nutzt Sektion ``features``).
        interval: Kerzen-Intervall (für korrekte Annualisierung).
    """

    def __init__(self, config: dict[str, Any], interval: str = "1d") -> None:
        self._cfg = config["features"]
        self._tf = Timeframe(interval)
        self._chart_columns: set[str] = set()

    @property
    def chart_columns(self) -> frozenset[str]:
        """Spalten, die nur für Charts gedacht sind (nach ``add_all`` gefüllt)."""
        return frozenset(self._chart_columns)

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt alle Indikatoren hinzu. Das Original wird nicht verändert."""
        out = df.copy()
        new: dict[str, pd.Series] = {}
        new.update(self._momentum(out))
        new.update(self._trend(out))
        new.update(self._volatility(out))
        new.update(self._volume(out))
        new.update(self._structure(out))
        # Einmal zusammenfügen statt spaltenweise (vermeidet Fragmentierung)
        return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)

    def feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Alle ML-tauglichen Spalten eines von ``add_all`` erzeugten DataFrames."""
        return [c for c in df.columns if c not in BASE_COLUMNS and c not in self._chart_columns]

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    def _momentum(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        c = self._cfg
        close, high, low = df["close"], df["high"], df["low"]
        k, d = stochastic(high, low, close, c["stoch_window"], c["stoch_smooth_window"])
        return {
            f"rsi_{c['rsi_short_window']}": rsi(close, c["rsi_short_window"]),
            f"rsi_{c['rsi_long_window']}": rsi(close, c["rsi_long_window"]),
            "stoch_k": k,
            "stoch_d": d,
        }

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------

    def _trend(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        c = self._cfg
        close = df["close"]

        ema_fast = ema(close, c["ema_fast"])
        ema_mid = ema(close, c["ema_mid"])
        ema_ls = ema(close, c["ema_long_short"])
        ema_ll = ema(close, c["ema_long_long"])
        macd_line, signal_line, hist = macd(close, c["macd_fast"], c["macd_slow"], c["macd_signal"])
        adx_s, plus_di, minus_di = adx(df["high"], df["low"], close, c["adx_window"])

        chart = {
            f"ema_fast_{c['ema_fast']}": ema_fast,
            f"ema_mid_{c['ema_mid']}": ema_mid,
            f"ema_long_{c['ema_long_short']}": ema_ls,
            f"ema_long_{c['ema_long_long']}": ema_ll,
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_diff": hist,
            "adx_pos": plus_di,
            "adx_neg": minus_di,
        }
        self._chart_columns.update(chart)

        features = {
            "macd_pct": macd_line / close,
            "macd_hist_pct": hist / close,
            "ema_cross_fast_mid": ema_fast / ema_mid - 1.0,
            "ema_cross_long": ema_ls / ema_ll - 1.0,
            "close_vs_ema_mid": close / ema_mid - 1.0,
            "close_vs_ema_long": close / ema_ls - 1.0,
            "adx": adx_s,
            "di_diff": plus_di - minus_di,
            "efficiency_ratio": efficiency_ratio(close, c["efficiency_window"]),
        }
        return {**chart, **features}

    # ------------------------------------------------------------------
    # Volatilität
    # ------------------------------------------------------------------

    def _volatility(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        c = self._cfg
        close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]
        ann = self._tf.annualization

        atr_s = atr(high, low, close, c["atr_window"])
        upper, mid, lower = bollinger_bands(close, c["bollinger_window"], c["bollinger_std"])
        log_ret = np.log(close / close.shift(1))

        chart = {"atr": atr_s, "bb_upper": upper, "bb_mid": mid, "bb_lower": lower}
        self._chart_columns.update(chart)

        hv_w = c["historical_volatility_window"]
        short_w, long_w = c["vol_ratio_short_window"], c["vol_ratio_long_window"]
        band = (upper - lower).replace(0, np.nan)
        features = {
            "atr_pct": atr_s / close,
            "bb_width": (upper - lower) / mid.replace(0, np.nan),
            "bb_pct": (close - lower) / band,
            "hist_vol": log_ret.rolling(hv_w, min_periods=hv_w).std() * ann,
            "gk_vol": garman_klass_volatility(open_, high, low, close, hv_w) * ann,
            "vol_ratio": (
                log_ret.rolling(short_w, min_periods=short_w).std()
                / log_ret.rolling(long_w, min_periods=long_w).std().replace(0, np.nan)
            ),
        }
        return {**chart, **features}

    # ------------------------------------------------------------------
    # Volumen
    # ------------------------------------------------------------------

    def _volume(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        c = self._cfg
        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
        vol_w, vwap_w, flow_w = c["volume_sma_window"], c["vwap_window"], c["orderflow_window"]

        obv_s = obv(close, volume)
        self._chart_columns.add("obv")

        vol_sum = volume.rolling(vwap_w, min_periods=vwap_w).sum().replace(0, np.nan)
        typical = (high + low + close) / 3.0
        vwap = (typical * volume).rolling(vwap_w, min_periods=vwap_w).sum() / vol_sum

        log_vol = np.log1p(volume)
        vol_mean = log_vol.rolling(vol_w, min_periods=vol_w).mean()
        vol_std = log_vol.rolling(vol_w, min_periods=vol_w).std().replace(0, np.nan)
        flow_vol = volume.rolling(flow_w, min_periods=flow_w).sum().replace(0, np.nan)

        features: dict[str, pd.Series] = {
            "obv": obv_s,
            "volume_ratio": volume / volume.rolling(vol_w, min_periods=vol_w).mean().replace(0, np.nan),
            "volume_z": (log_vol - vol_mean) / vol_std,
            # Netto-Volumenfluss der letzten N Kerzen relativ zum Gesamtvolumen: [-1, 1]
            "obv_slope": (obv_s - obv_s.shift(flow_w)) / flow_vol,
            "vwap_dist": close / vwap - 1.0,
        }
        if "taker_buy_base" in df.columns:
            # Anteil aggressiver Käufer am Volumen (0.5 = ausgeglichen) → zentriert auf 0
            buy = df["taker_buy_base"].rolling(flow_w, min_periods=flow_w).sum()
            features["taker_buy_ratio"] = buy / flow_vol - 0.5
        return features

    # ------------------------------------------------------------------
    # Preisstruktur / Returns
    # ------------------------------------------------------------------

    def _structure(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        c = self._cfg
        close, high, low = df["close"], df["high"], df["low"]
        log_ret = np.log(close / close.shift(1))
        z_w, range_w = c["return_z_window"], c["range_window"]

        features = {f"ret_{p}": np.log(close / close.shift(p)) for p in c["return_periods"]}
        features["ret_z"] = log_ret / log_ret.rolling(z_w, min_periods=z_w).std().replace(0, np.nan)
        features["dist_high"] = close / high.rolling(range_w, min_periods=range_w).max() - 1.0
        features["dist_low"] = close / low.rolling(range_w, min_periods=range_w).min() - 1.0
        return features


# ===========================================================================
# Freistehende Berechnungsfunktionen (pure pandas/numpy)
# ===========================================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponentieller gleitender Durchschnitt (alpha = 2 / (span + 1)).

    ``min_periods=span``: Die ersten Werte sind NaN statt stark vom
    Startwert verzerrt (wichtig z.B. für EMA 200).
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder-Glättung (RMA): EMA mit alpha = 1 / window."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index nach Wilder (0–100).

    Randfälle: nur Gewinne → 100, nur Verluste → 0, keine Bewegung → 50.
    """
    delta = close.diff()
    avg_gain = wilder_smooth(delta.clip(lower=0), window)
    avg_loss = wilder_smooth(-delta.clip(upper=0), window)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out.where(avg_gain.notna() & avg_loss.notna())


def macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD-Linie, Signal-Linie und Histogramm (in Preiseinheiten)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int, smooth: int
) -> tuple[pd.Series, pd.Series]:
    """Stochastischer Oszillator %K und %D (0–100).

    Hinweis: Williams %R ist exakt %K − 100 und wird daher nicht zusätzlich
    als Feature geführt (perfekt korreliert = keine neue Information).
    """
    low_n = low.rolling(window, min_periods=window).min()
    high_n = high.rolling(window, min_periods=window).max()
    k = 100.0 * (close - low_n) / (high_n - low_n).replace(0, np.nan)
    return k, k.rolling(smooth, min_periods=smooth).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(H−L, |H−C_prev|, |L−C_prev|)."""
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=False).fillna(high - low)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average True Range nach Wilder (RMA der True Range)."""
    return wilder_smooth(true_range(high, low, close), window)


def bollinger_bands(close: pd.Series, window: int, n_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (oben, Mitte, unten) mit Populations-Std (ddof=0, wie Bollinger)."""
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return mid + n_std * std, mid, mid - n_std * std


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index mit +DI und −DI nach Wilder (je 0–100)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)

    atr_s = wilder_smooth(true_range(high, low, close), window).replace(0, np.nan)
    plus_di = 100.0 * wilder_smooth(plus_dm, window) / atr_s
    minus_di = 100.0 * wilder_smooth(minus_dm, window) / atr_s

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder_smooth(dx, window), plus_di, minus_di


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (kumulativ)."""
    return (volume * np.sign(close.diff()).fillna(0.0)).cumsum()


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Kaufman Efficiency Ratio: |Netto-Bewegung| / Summe der Einzelbewegungen (0–1).

    1 = perfekt sauberer Trend, ~0 = Seitwärts-Rauschen.
    """
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return net / path.replace(0, np.nan)


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Garman-Klass-Volatilität pro Kerze (nutzt OHLC → effizienter als Close-to-Close)."""
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    var = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    return np.sqrt(var.rolling(window, min_periods=window).mean().clip(lower=0))
