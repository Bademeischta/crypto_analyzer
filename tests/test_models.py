"""Walk-Forward, Engine, Kalibrierung und Evaluierung."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from src.features.pipeline import FeatureMatrix, FeaturePipeline
from src.models.engine import (
    Calibration,
    ConstantProbaModel,
    fit_calibration,
    fit_classifier,
    lightgbm_available,
    predict_proba_aligned,
    resolve_engine,
)
from src.models.evaluator import ModelEvaluator, _block_ttest_p
from src.models.trainer import ModelTrainer, compute_walk_forward_folds

# ---------------------------------------------------------------------------
# Walk-Forward-Folds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rolling", [None, 250])
def test_folds_are_purged_contiguous_and_anchored_to_end(rolling):
    n, min_train, test, embargo = 1000, 200, 50, 7
    folds = compute_walk_forward_folds(n, min_train, test, embargo, max_folds=10, rolling_window=rolling)
    assert len(folds) == 10
    assert folds[-1].test_end == n
    for prev, cur in zip(folds, folds[1:]):
        assert prev.test_end == cur.test_start  # lückenlos, keine Überlappung
    for f in folds:
        assert f.test_start - f.train_end == embargo  # Embargo eingehalten
        assert f.train_end - f.train_start >= min_train
        assert set(f.train_idx).isdisjoint(f.test_idx)
        if rolling:
            assert f.train_end - f.train_start <= rolling


def test_folds_stop_when_training_too_small():
    folds = compute_walk_forward_folds(300, 200, 50, 5, max_folds=10)
    assert all(f.train_end - f.train_start >= 200 for f in folds)
    assert len(folds) == 1


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_resolve_engine():
    assert resolve_engine("hist_gb") == "hist_gb"
    assert resolve_engine("auto") == ("lightgbm" if lightgbm_available() else "hist_gb")


@pytest.mark.parametrize("engine", ["hist_gb"] + (["lightgbm"] if lightgbm_available() else []))
def test_predict_proba_aligned_with_missing_class(config, engine):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = np.where(X[:, 0] > 0, 2, 0)  # Klasse 1 fehlt komplett
    model = fit_classifier(config["ml"]["model"], engine, X, y)
    proba = predict_proba_aligned(model, X[:10])
    assert proba.shape == (10, 3)
    np.testing.assert_allclose(proba[:, 1], 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_single_class_falls_back_to_constant_model(config):
    model = fit_classifier(config["ml"]["model"], "hist_gb", np.zeros((50, 2)), np.full(50, 1))
    assert isinstance(model, ConstantProbaModel)
    np.testing.assert_allclose(predict_proba_aligned(model, np.zeros((3, 2))), [[0, 1, 0]] * 3)


# ---------------------------------------------------------------------------
# Kalibrierung
# ---------------------------------------------------------------------------


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_calibration_tames_overconfident_model():
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 0.5  # schwaches echtes Signal …
    proba = _softmax(logits * 4)  # … aber massiv überkonfident
    prior = np.full(3, 1 / 3)
    cal = fit_calibration(proba, y, prior)
    assert cal.temperature > 1.5
    calibrated = cal.apply(proba, prior)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)

    def ll(p):
        return -np.mean(np.log(p[np.arange(n), y]))

    assert ll(calibrated) < ll(proba)


def test_calibration_falls_back_to_prior_for_noise():
    rng = np.random.default_rng(2)
    n = 3000
    y = rng.choice(3, n, p=[0.2, 0.5, 0.3])
    proba = _softmax(rng.normal(0, 2, (n, 3)))  # reines Rauschen
    prior = np.array([0.2, 0.5, 0.3])
    cal = fit_calibration(proba, y, prior)
    assert cal.weight < 0.15
    assert Calibration().apply(proba, prior) is not None


# ---------------------------------------------------------------------------
# Training (End-to-End auf synthetischen Daten)
# ---------------------------------------------------------------------------


def _synthetic_matrix(n: int = 900, informative: bool = True, seed: int = 0) -> FeatureMatrix:
    """Feature-Matrix mit (optional) eingebautem, echtem Signal in Feature f0."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    X = pd.DataFrame(rng.normal(size=(n, 5)), index=idx, columns=[f"f{i}" for i in range(5)])
    if informative:
        y = np.digitize(X["f0"] + rng.normal(0, 0.6, n), [-0.45, 0.45])
    else:
        y = rng.integers(0, 3, n)
    y_vol = np.digitize(X["f1"] + rng.normal(0, 3, n), [-1, 1])
    fwd = pd.Series((y - 1) * 0.02 + rng.normal(0, 0.01, n), index=idx)
    return FeatureMatrix(
        X=X, y_direction=pd.Series(y, index=idx), y_volatility=pd.Series(y_vol, index=idx),
        forward_return=fwd, next_return=fwd / 5, feature_names=list(X.columns),
        required_features=list(X.columns), last_row=X.iloc[-1], last_timestamp=idx[-1], frame=X,
        horizon_bars=5, current_thresholds=(-0.02, 0.02),
    )


def test_training_detects_real_signal_and_rejects_noise(config, tmp_path):
    trainer = ModelTrainer(config, tmp_path)
    evaluator = ModelEvaluator(config)

    signal = trainer.train(_synthetic_matrix(informative=True), "SIG", "1d", "fp")
    m = evaluator.evaluate(signal.oof, signal.horizon_bars)
    assert m.verdict == "signifikant" and m.skill_score > 0.05 and m.p_value < 0.01
    top_feature = next(iter(sorted(signal.feature_importance.items(), key=lambda kv: -kv[1])))[0]
    assert top_feature == "f0"

    noise = trainer.train(_synthetic_matrix(informative=False, seed=5), "NOISE", "1d", "fp")
    m_noise = evaluator.evaluate(noise.oof, noise.horizon_bars)
    assert m_noise.verdict != "signifikant"
    assert m_noise.skill_score < 0.02


def test_oof_structure_and_persistence(config, tmp_path):
    fm = FeaturePipeline(config, "1d").build(make_ohlcv(n=900))
    trainer = ModelTrainer(config, tmp_path)
    result = trainer.train(fm, "BTC", "1d", "abc123")

    oof = result.oof
    assert oof.index.is_monotonic_increasing and oof.index.is_unique
    assert set(oof.index) <= set(fm.X.index)
    np.testing.assert_allclose(oof[["p_down", "p_neutral", "p_up"]].sum(axis=1), 1.0, rtol=1e-9)
    assert oof["fold"].nunique() == len(result.folds) >= config["ml"]["walk_forward"]["min_folds"]
    # Die OOF-Historie endet mit den neuesten gelabelten Daten
    assert oof.index[-1] == fm.X.index[-1]

    loaded = trainer.load("BTC", "1d", "abc123")
    assert loaded is not None and loaded.from_cache
    pd.testing.assert_frame_equal(loaded.oof, oof)
    assert trainer.load("BTC", "1d", "other") is None
    # Neues Bundle ersetzt das alte
    trainer.save("BTC", "1d", result.__class__(**{**result.__dict__, "fingerprint": "new"}))
    assert len(list(tmp_path.glob("BTC_1d_*.joblib"))) == 1


def test_training_rejects_too_little_history(config, tmp_path):
    fm = FeaturePipeline(config, "1d").build(make_ohlcv(n=320))
    with pytest.raises(ValueError, match="Zu wenig Historie"):
        ModelTrainer(config, tmp_path).train(fm, "NEW", "1d", "fp")


def test_parallel_and_sequential_training_are_identical(config, tmp_path):
    fm = _synthetic_matrix(n=700)
    config["ml"]["parallel_folds"] = 1
    seq = ModelTrainer(config, tmp_path / "a").train(fm, "X", "1d", "fp")
    config["ml"]["parallel_folds"] = 4
    par = ModelTrainer(config, tmp_path / "b").train(fm, "X", "1d", "fp")
    pd.testing.assert_frame_equal(seq.oof, par.oof)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _oof(y, proba, prior=(1 / 3, 1 / 3, 1 / 3), fwd=None):
    n = len(y)
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    proba = np.asarray(proba, dtype=float)
    return pd.DataFrame({
        "fold": np.repeat(np.arange(1, 5), int(np.ceil(n / 4)))[:n],
        "p_down": proba[:, 0], "p_neutral": proba[:, 1], "p_up": proba[:, 2],
        "y_true": y, "pred": proba.argmax(axis=1), "majority": 1,
        "prior_down": prior[0], "prior_neutral": prior[1], "prior_up": prior[2],
        "fwd_ret": fwd if fwd is not None else np.zeros(n), "next_ret": np.zeros(n),
    }, index=idx)


def test_evaluator_signal_statistics():
    y = np.array([2, 2, 0, 1] * 50)
    proba = np.tile([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.8, 0.1], [0.3, 0.4, 0.3]], (50, 1))
    fwd = np.tile([0.05, -0.01, 0.0, 0.0], 50)
    stats = ModelEvaluator.signal_statistics(_oof(y, proba, fwd=fwd), threshold=0.6)
    bull = stats["BULLISH"]
    assert bull["n_signals"] == 100
    assert bull["hit_rate"] == 1.0
    assert bull["direction_hit_rate"] == 0.5
    assert bull["avg_return"] == pytest.approx(0.02)
    assert stats["BEARISH"]["n_signals"] == 0


def test_evaluator_perfect_vs_prior(config):
    y = np.tile([0, 1, 2], 100)
    perfect = np.eye(3)[y] * 0.9 + 0.1 / 3
    m = ModelEvaluator(config).evaluate(_oof(y, perfect), horizon_bars=1)
    assert m.accuracy == 1.0 and m.skill_score > 0.5 and m.verdict == "signifikant"

    uniform = np.full((len(y), 3), 1 / 3)
    m0 = ModelEvaluator(config).evaluate(_oof(y, uniform), horizon_bars=1)
    assert m0.skill_score == pytest.approx(0.0, abs=1e-9)
    assert m0.verdict == "keine Vorhersagekraft"


def test_block_ttest_requires_enough_blocks():
    assert np.isnan(_block_ttest_p(np.ones(50), block=10))
    rng = np.random.default_rng(0)
    assert _block_ttest_p(rng.normal(0.5, 1, 1000), block=5) < 0.001
