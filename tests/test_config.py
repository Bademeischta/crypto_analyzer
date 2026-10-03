import copy

import pytest

from src.config import ConfigError, Timeframe, _validate, config_fingerprint, per_interval


def test_default_config_is_valid(config):
    _validate(config)


def test_timeframe_conversion():
    assert Timeframe("1d").bars(5) == 5
    assert Timeframe("4h").bars(5) == 30
    assert Timeframe("1h").bars(5) == 120
    assert Timeframe("1h").bars_per_day == 24
    assert Timeframe("1d").periods_per_year == 365
    assert Timeframe("1h").describe_bars(120) == "5 Tage"
    assert Timeframe("1h").describe_bars(12) == "12 Stunden"
    with pytest.raises(ValueError):
        Timeframe("7m")


def test_per_interval():
    value = {"1h": 90, "default": 365}
    assert per_interval(value, "1h") == 90
    assert per_interval(value, "4h") == 365
    assert per_interval(42, "1h") == 42


def test_fingerprint_changes_with_ml_config(config):
    other = copy.deepcopy(config)
    other["ml"]["model"]["learning_rate"] = 0.5
    assert config_fingerprint(config) != config_fingerprint(other)
    ui_only = copy.deepcopy(config)
    ui_only["ui"]["default_symbol"] = "ETH"
    assert config_fingerprint(config) == config_fingerprint(ui_only)


@pytest.mark.parametrize(
    "path, value",
    [
        (("ml", "engine"), "xgboost"),
        (("ml", "confidence_display_threshold"), 0.2),
        (("ml", "direction", "label_mode"), "magic"),
        (("ml", "calibration"), "isotonic"),
    ],
)
def test_invalid_config_raises(config, path, value):
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ConfigError):
        _validate(config)
