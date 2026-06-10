"""Technische Indikatoren – reine pandas/numpy Implementierung.

Keine externe Bibliothek (kein ta, kein pandas-ta, kein ta-lib).
Alle Formeln sind standardisiert und verifiziert gegen bekannte Referenzen.

Implementierte Indikatoren:
  Momentum:   RSI(n), Stochastic %K/%D, Williams %R
  Trend:      EMA(n), MACD, ADX (+DI / -DI)
  Volatilität: ATR, Bollinger Bands, Historische Volatilität
  Volumen:    OBV, Volumen-SMA-Ratio, VWAP-Abstand
  Struktur:   Log-Returns, Volatilitätsregime
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Berechnet alle technischen Indikatoren aus dem Masterplan.

    Args:
        config: Features-Sektion der config.yaml (vollständiges config-Dict).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config["features"]

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt alle Indikatoren-Spalten zum DataFrame hinzu.

        Args:
            df: OHLCV-DataFrame mit Spalten open/high/low/close/volume.

        Returns:
            Erweiterter DataFrame. Original wird nicht modifiziert.
        """
        df = df.copy()
        df = self._add_momentum(df)
        df = self._add_trend(df)
        df = self._add_volatility(df)
        df = self._add_volume_indicators(df)
        df = self._add_price_structure(df)
        return df

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI (kurz + lang), Stochastic %K/%D, Williams %R."""
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        rsi_short_w  = self._cfg["rsi_short_window"]
        rsi_long_w   = self._cfg["rsi_long_window"]
        stoch_w      = self._cfg["stoch_window"]
        stoch_smooth = self._cfg["stoch_smooth_window"]
        wr_w         = self._cfg["williams_r_window"]

        df[f"rsi_{rsi_short_w}"] = _rsi(close, rsi_short_w)
        df[f"rsi_{rsi_long_w}"]  = _rsi(close, rsi_long_w)

        k, d = _stochastic(high, low, close, stoch_w, stoch_smooth)
        df["stoch_k"] = k
        df["stoch_d"] = d

        df["williams_r"] = _williams_r(high, low, close, wr_w)

        return df

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD, EMA-Crosses (normalisiert), ADX."""
        close = df["close"]

        macd_fast = self._cfg["macd_fast"]
        macd_slow = self._cfg["macd_slow"]
        macd_sig  = self._cfg["macd_signal"]
        ema_fast  = self._cfg["ema_fast"]
        ema_mid   = self._cfg["ema_mid"]
        ema_ls    = self._cfg["ema_long_short"]
        ema_ll    = self._cfg["ema_long_long"]
        adx_w     = self._cfg["adx_window"]

        macd_line, signal_line, hist = _macd(close, macd_fast, macd_slow, macd_sig)
        df["macd"]        = macd_line
        df["macd_signal"] = signal_line
        df["macd_diff"]   = hist

        ema_fast_s = _ema(close, ema_fast)
        ema_mid_s  = _ema(close, ema_mid)
        ema_ls_s   = _ema(close, ema_ls)
        ema_ll_s   = _ema(close, ema_ll)

        # Absolute EMA-Serien für Chart-Overlays (aus ML-Features ausgeschlossen)
        df[f"ema_fast_{ema_fast}"] = ema_fast_s
        df[f"ema_mid_{ema_mid}"]   = ema_mid_s
        df[f"ema_long_{ema_ls}"]   = ema_ls_s

        # Normalisierte EMA-Cross-Differenzen als ML-Feature (stationär)
        df["ema_cross_fast_mid"] = (ema_fast_s - ema_mid_s) / ema_mid_s.replace(0, np.nan)
        df["ema_cross_ls_ll"]    = (ema_ls_s - ema_ll_s)   / ema_ll_s.replace(0, np.nan)

        adx, adx_pos, adx_neg = _adx(df["high"], df["low"], close, adx_w)
        df["adx"]     = adx
        df["adx_pos"] = adx_pos
        df["adx_neg"] = adx_neg

        return df

    # ------------------------------------------------------------------
    # Volatilität
    # ------------------------------------------------------------------

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR, Bollinger Bands (Width + %B), historische Volatilität."""
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        atr_w  = self._cfg["atr_window"]
        bb_w   = self._cfg["bollinger_window"]
        bb_std = self._cfg["bollinger_std"]
        hv_w   = self._cfg["historical_volatility_window"]

        df["atr"] = _atr(high, low, close, atr_w)

        bb_upper, bb_mid, bb_lower = _bollinger_bands(close, bb_w, bb_std)
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower
        df["bb_mid"]   = bb_mid
        # Normalisierte Bandbreite (entfernt Preisniveau-Abhängigkeit)
        df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
        # Preis-Position innerhalb der Bänder [0, 1]
        band_range = (bb_upper - bb_lower).replace(0, np.nan)
        df["bb_pct"]   = (close - bb_lower) / band_range

        # Historische Volatilität: annualisierte Std. der log-Returns
        log_ret = np.log(close / close.shift(1))
        df["hist_vol"] = log_ret.rolling(hv_w).std() * np.sqrt(252)

        return df

    # ------------------------------------------------------------------
    # Volumen
    # ------------------------------------------------------------------

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV, Volumen-SMA-Ratio, VWAP-Abstand."""
        close  = df["close"]
        volume = df["volume"]
        high   = df["high"]
        low    = df["low"]

        vol_sma_w = self._cfg["volume_sma_window"]
        vwap_w    = self._cfg["vwap_window"]

        df["obv"]       = _obv(close, volume)
        obv_sma         = df["obv"].rolling(vol_sma_w).mean()
        df["obv_trend"] = df["obv"] - obv_sma

        vol_sma = volume.rolling(vol_sma_w).mean()
        df["volume_sma_ratio"] = volume / vol_sma.replace(0, np.nan)

        # Rollender VWAP-Abstand
        typical_price = (high + low + close) / 3
        tp_vol         = typical_price * volume
        rolling_vwap   = tp_vol.rolling(vwap_w).sum() / volume.rolling(vwap_w).sum()
        df["vwap_dist"] = (close - rolling_vwap) / rolling_vwap.replace(0, np.nan)

        return df

    # ------------------------------------------------------------------
    # Preis-Struktur / Returns
    # ------------------------------------------------------------------

    def _add_price_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-Returns für 1d/3d/7d, Volatilitätsregime."""
        close          = df["close"]
        return_periods: list[int] = self._cfg["return_periods"]
        vola_w         = self._cfg["volatility_regime_window"]

        for p in return_periods:
            df[f"log_return_{p}d"] = np.log(close / close.shift(p))

        daily_log_ret         = np.log(close / close.shift(1))
        df["volatility_regime"] = daily_log_ret.rolling(vola_w).std()

        return df


# ===========================================================================
# Freistehende Berechnungsfunktionen (pure pandas/numpy, keine externen Deps)
# ===========================================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponentieller gleitender Durchschnitt (Wilder-Smoothing via ewm).

    Args:
        series: Preisreihe.
        span: Fensterlänge.

    Returns:
        EMA-Series.
    """
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index nach Wilder.

    Formel: RSI = 100 - 100 / (1 + RS), RS = avg_up / avg_down
    Verwendet EWMA (com = window - 1) wie die originale Wilder-Methode.

    Args:
        close: Schlusskurse.
        window: RSI-Periode (typisch 14).

    Returns:
        RSI-Series (0–100).
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))

    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd(
    close: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD, Signal-Linie und Histogramm.

    Args:
        close: Schlusskurse.
        fast: Schnelle EMA-Periode (typisch 12).
        slow: Langsame EMA-Periode (typisch 26).
        signal: Signal-EMA-Periode (typisch 9).

    Returns:
        Tupel (macd, signal_line, histogram).
    """
    ema_fast    = _ema(close, fast)
    ema_slow    = _ema(close, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    smooth: int,
) -> tuple[pd.Series, pd.Series]:
    """Stochastischer Oszillator %K und %D.

    Args:
        high: Tageshochs.
        low: Tagestiefs.
        close: Schlusskurse.
        window: Lookback-Periode (typisch 14).
        smooth: Glättungsperiode für %D (typisch 3).

    Returns:
        Tupel (%K, %D) jeweils als Series (0–100).
    """
    low_n  = low.rolling(window).min()
    high_n = high.rolling(window).max()
    denom  = (high_n - low_n).replace(0, np.nan)
    k = 100.0 * (close - low_n) / denom
    d = k.rolling(smooth).mean()
    return k, d


def _williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """Williams %R.

    Args:
        high: Tageshochs.
        low: Tagestiefs.
        close: Schlusskurse.
        window: Lookback-Periode (typisch 14).

    Returns:
        Williams %R Series (-100–0).
    """
    high_n = high.rolling(window).max()
    low_n  = low.rolling(window).min()
    denom  = (high_n - low_n).replace(0, np.nan)
    return -100.0 * (high_n - close) / denom


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """Average True Range (ATR) nach Wilder.

    True Range = max(H-L, |H-C_prev|, |L-C_prev|)

    Args:
        high: Tageshochs.
        low: Tagestiefs.
        close: Schlusskurse.
        window: Glättungsperiode (typisch 14).

    Returns:
        ATR-Series.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()


def _bollinger_bands(
    close: pd.Series,
    window: int,
    n_std: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Args:
        close: Schlusskurse.
        window: SMA-Periode (typisch 20).
        n_std: Standardabweichungs-Multiplikator (typisch 2.0).

    Returns:
        Tupel (upper, middle, lower).
    """
    mid   = close.rolling(window).mean()
    std   = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index (ADX) mit +DI und -DI.

    Implementiert nach Wilder's original Methode.

    Args:
        high: Tageshochs.
        low: Tagestiefs.
        close: Schlusskurse.
        window: Glättungsperiode (typisch 14).

    Returns:
        Tupel (ADX, +DI, -DI) jeweils als Series (0–100).
    """
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    # Directional Movement
    up_move   = high - prev_high
    down_move = prev_low - low

    # +DM: Aufwärtsbewegung > Abwärtsbewegung und positiv
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=close.index)
    minus_dm_s = pd.Series(minus_dm, index=close.index)

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder-Smoothing (EWMA mit com = window - 1)
    atr_s       = tr.ewm(com=window - 1, adjust=False).mean()
    smooth_pdm  = plus_dm_s.ewm(com=window - 1, adjust=False).mean()
    smooth_mdm  = minus_dm_s.ewm(com=window - 1, adjust=False).mean()

    # Directional Indicators
    plus_di  = 100.0 * smooth_pdm  / atr_s.replace(0, np.nan)
    minus_di = 100.0 * smooth_mdm  / atr_s.replace(0, np.nan)

    # DX und ADX
    di_sum  = (plus_di + minus_di).replace(0, np.nan)
    dx      = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx     = dx.ewm(com=window - 1, adjust=False).mean()

    return adx, plus_di, minus_di


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (OBV).

    OBV steigt wenn close > prev_close, fällt wenn close < prev_close.

    Args:
        close: Schlusskurse.
        volume: Handelsvolumen.

    Returns:
        OBV-Series (kumulativ).
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()
