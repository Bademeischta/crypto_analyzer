import numpy as np
import pandas as pd
import pytest

from src.features.technical import (
    BASE_COLUMNS,
    TechnicalIndicators,
    atr,
    bollinger_bands,
    describe_feature,
    ema,
    rsi,
    stochastic,
    true_range,
)


def test_rsi_bounds_and_edge_cases(ohlcv):
    values = rsi(ohlcv["close"], 14).dropna()
    assert values.between(0, 100).all()

    rising = pd.Series(np.arange(1.0, 60.0))
    assert rsi(rising, 14).dropna().eq(100.0).all()
    falling = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert rsi(falling, 14).dropna().eq(0.0).all()
    flat = pd.Series(np.full(40, 5.0))
    assert rsi(flat, 14).dropna().eq(50.0).all()


def test_atr_matches_wilder_recursion(ohlcv):
    window = 14
    tr = true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"]).to_numpy()
    expected = np.full(len(tr), np.nan)
    acc = tr[0]
    for i in range(len(tr)):
        acc = tr[0] if i == 0 else acc + (tr[i] - acc) / window
        if i >= window - 1:
            expected[i] = acc
    result = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], window).to_numpy()
    np.testing.assert_allclose(result[window:], expected[window:], rtol=1e-10)


def test_ema_has_warmup_nans():
    s = pd.Series(np.arange(300.0))
    out = ema(s, 200)
    assert out.iloc[:199].isna().all()
    assert out.iloc[199:].notna().all()


def test_bollinger_mid_is_sma(ohlcv):
    upper, mid, lower = bollinger_bands(ohlcv["close"], 20, 2.0)
    np.testing.assert_allclose(mid.dropna(), ohlcv["close"].rolling(20).mean().dropna())
    assert (upper.dropna() >= lower.dropna()).all()


def test_stochastic_bounds(ohlcv):
    k, d = stochastic(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14, 3)
    assert k.dropna().between(0, 100).all()
    assert d.dropna().between(0, 100).all()


def test_feature_columns_exclude_raw_and_chart_columns(config, ohlcv):
    tech = TechnicalIndicators(config, "1d")
    frame = tech.add_all(ohlcv)
    features = tech.feature_columns(frame)
    assert features
    assert not set(features) & BASE_COLUMNS
    assert not set(features) & tech.chart_columns
    for chart_col in ("bb_upper", "macd", "obv", "atr"):
        assert chart_col in frame.columns and chart_col not in features


def test_features_are_price_scale_invariant(config, ohlcv):
    """Stationarität: Ein 1000× höheres Preisniveau darf kein Feature verändern."""
    tech = TechnicalIndicators(config, "1d")
    scaled = ohlcv.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] *= 1000
    a = tech.add_all(ohlcv)
    b = TechnicalIndicators(config, "1d").add_all(scaled)
    for col in tech.feature_columns(a):
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(), rtol=1e-7, atol=1e-9, err_msg=col)


def test_annualization_depends_on_interval(config, ohlcv):
    daily = TechnicalIndicators(config, "1d").add_all(ohlcv)["hist_vol"]
    hourly = TechnicalIndicators(config, "1h").add_all(ohlcv)["hist_vol"]
    np.testing.assert_allclose((hourly / daily).dropna(), np.sqrt(24))


@pytest.mark.parametrize(
    "name, expected",
    [("rsi_14", "RSI (14)"), ("corr_btc", "Korrelation (BTC)"), ("adx", "ADX (Trendstärke)"), ("unknown", "unknown")],
)
def test_describe_feature(name, expected):
    assert describe_feature(name) == expected
