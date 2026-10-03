"""Die wichtigsten Tests des Projekts: kein Lookahead-Bias in Features und Labels."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_fear_greed, make_ohlcv
from src.features.pipeline import FeaturePipeline


@pytest.fixture
def references():
    return {"BTC": make_ohlcv(seed=7, price=30_000), "ETH": make_ohlcv(seed=8, price=2_000)}


def _assert_frames_equal_where_known(a: pd.DataFrame, b: pd.DataFrame) -> None:
    for col in a.columns:
        np.testing.assert_allclose(
            a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float), rtol=1e-9, atol=1e-12, equal_nan=True,
            err_msg=f"Feature '{col}' ändert sich durch zukünftige Daten (Lookahead!)",
        )


@pytest.mark.parametrize("cut", [400, 650])
def test_features_do_not_use_future_data(config, ohlcv, references, cut):
    """Anhängen zukünftiger Kerzen darf keinen einzigen vergangenen Feature-Wert verändern."""
    fng = make_fear_greed()
    pipe = FeaturePipeline(config, "1d")
    full = pipe.build(ohlcv, references, fng)
    past = FeaturePipeline(config, "1d").build(ohlcv.iloc[:cut], references, fng)

    cols = full.feature_names
    assert set(cols) == set(past.feature_names)
    _assert_frames_equal_where_known(past.frame[cols], full.frame[cols].iloc[:cut])


def test_reference_features_do_not_use_future_reference_data(config, ohlcv, references):
    cut = 500
    truncated_refs = {k: v.iloc[:cut] for k, v in references.items()}
    pipe = FeaturePipeline(config, "1d")
    a = pipe.build(ohlcv.iloc[:cut], truncated_refs).frame
    b = FeaturePipeline(config, "1d").build(ohlcv, references).frame.iloc[:cut]
    ref_cols = [c for c in a.columns if c.startswith(("corr_", "beta_", "rel_strength_"))]
    assert ref_cols
    _assert_frames_equal_where_known(a[ref_cols], b[ref_cols])


def test_labels_are_known_only_after_horizon(config, ohlcv):
    fm = FeaturePipeline(config, "1d").build(ohlcv)
    h = fm.horizon_bars
    # Die letzten h Kerzen haben kein Label und dürfen nicht im Training sein
    assert fm.X.index[-1] <= ohlcv.index[-1 - h]
    assert fm.last_timestamp == ohlcv.index[-1]
    assert len(fm.X) == len(fm.y_direction) == len(fm.y_volatility) == len(fm.forward_return)


def test_direction_label_matches_definition(config, ohlcv):
    cfg = config
    cfg["ml"]["direction"]["label_mode"] = "fixed"
    fm = FeaturePipeline(cfg, "1d").build(ohlcv)
    h = fm.horizon_bars
    close = ohlcv["close"]
    fwd = close.shift(-h) / close - 1
    up, down = cfg["ml"]["direction"]["up_threshold"], cfg["ml"]["direction"]["down_threshold"]
    expected = np.where(fwd > up, 2, np.where(fwd < down, 0, 1))
    expected = pd.Series(expected, index=close.index).loc[fm.X.index]
    pd.testing.assert_series_equal(fm.y_direction, expected.astype(int), check_names=False)
    np.testing.assert_allclose(fm.forward_return, fwd.loc[fm.X.index])


def test_labels_do_not_depend_on_data_beyond_horizon(config, ohlcv):
    """Label bei t darf nur Kurse bis t+h nutzen – Klassengrenzen inklusive."""
    cut = 600
    full = FeaturePipeline(config, "1d").build(ohlcv)
    part = FeaturePipeline(config, "1d").build(ohlcv.iloc[:cut])
    common = part.X.index
    pd.testing.assert_series_equal(part.y_direction, full.y_direction.loc[common])
    pd.testing.assert_series_equal(part.y_volatility, full.y_volatility.loc[common])


def test_volatility_adjusted_labels_are_roughly_balanced(config):
    fm = FeaturePipeline(config, "1d").build(make_ohlcv(n=1500, seed=3))
    dist = fm.label_distribution
    assert all(0.15 < share < 0.6 for share in dist.values()), dist


def test_fear_greed_is_lagged_by_one_day(config, ohlcv):
    fng = make_fear_greed()
    fm = FeaturePipeline(config, "1d").build(ohlcv, fear_greed=fng)
    ts = ohlcv.index[300]
    # Tageskerze ts schließt um ts+1d → verfügbar ist der Wert vom Tag ts (veröffentlicht, gilt ab ts+1d)
    assert fm.frame.loc[ts, "fng_value"] == fng.loc[ts]
    assert fm.frame.loc[ts, "fng_change_7d"] == fng.loc[ts] - fng.loc[ts - pd.Timedelta(days=7)]


def test_intraday_horizon_in_bars(config):
    df = make_ohlcv(n=3000, freq="1h")
    pipe = FeaturePipeline(config, "1h")
    assert pipe.horizon_bars == 5 * 24
    fm = pipe.build(df)
    assert fm.horizon_bars == 120


def test_last_row_is_latest_bar(config, ohlcv, references):
    fm = FeaturePipeline(config, "1d").build(ohlcv, references, make_fear_greed())
    pd.testing.assert_series_equal(fm.last_row, fm.frame[fm.feature_names].iloc[-1], check_names=False)
    assert not fm.last_row[fm.required_features].isna().any()
