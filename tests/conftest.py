"""Gemeinsame Test-Fixtures: synthetische Marktdaten und eine schnelle Test-Config.

Alle Tests laufen offline – kein Test darf echte APIs aufrufen.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


def make_ohlcv(
    n: int = 900,
    seed: int = 0,
    freq: str = "1D",
    start: str = "2021-01-01",
    vol: float = 0.03,
    drift: float = 0.0003,
    price: float = 100.0,
) -> pd.DataFrame:
    """Synthetische, konsistente OHLCV-Daten mit Volatilitäts-Clustering."""
    rng = np.random.default_rng(seed)
    regime = np.exp(pd.Series(rng.normal(0, 0.35, n)).rolling(20, min_periods=1).mean().to_numpy())
    rets = rng.normal(drift, vol, n) * regime
    close = price * np.exp(np.cumsum(rets))
    open_ = np.r_[price, close[:-1]] * (1 + rng.normal(0, vol / 5, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.lognormal(10, 0.5, n)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "num_trades": rng.integers(1_000, 10_000, n).astype(float),
            "taker_buy_base": volume * rng.uniform(0.35, 0.65, n),
        },
        index=idx,
    )


def make_fear_greed(start: str = "2020-06-01", days: int = 2000, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = np.clip(50 + np.cumsum(rng.normal(0, 4, days)), 5, 95).round()
    idx = pd.date_range(start, periods=days, freq="1D", tz="UTC")
    return pd.Series(values, index=idx, name="fear_greed")


def fast_config(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Default-Config mit kleinen Walk-Forward-Fenstern für schnelle Tests."""
    cfg = copy.deepcopy(base or load_config(ROOT / "config.yaml"))
    wf = cfg["ml"]["walk_forward"]
    wf.update(min_train_days=200, test_window_days=40, max_folds=5, min_folds=3, importance_folds=2,
              permutation_repeats=1)
    cfg["ml"]["model"].update(n_estimators=25)
    cfg["ml"]["parallel_folds"] = 2
    cfg["ml"]["volatility"]["regime_lookback_days"] = 120
    cfg["ml"]["training_lookback_days"] = 1000
    # CI testet beide Engines: CA_TEST_ENGINE=hist_gb erzwingt den scikit-learn-Fallback
    cfg["ml"]["engine"] = os.environ.get("CA_TEST_ENGINE", cfg["ml"]["engine"])
    return cfg


@pytest.fixture
def config() -> dict[str, Any]:
    return fast_config()


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def tmp_config_path(tmp_path: Path) -> Path:
    """Schreibt eine schnelle Test-Config mit Datenpfaden im tmp-Verzeichnis."""
    cfg = fast_config()
    cfg["paths"] = {
        "data_dir": "data",
        "cache_dir": "data/cache",
        "models_dir": "data/models",
        "logs_dir": "data/logs",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path
