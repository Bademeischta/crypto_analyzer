import numpy as np
import pandas as pd
import pytest

from src.models.backtest import generate_positions, performance_stats, run_backtest


def _oof(p_up, next_ret):
    n = len(p_up)
    p_up = np.asarray(p_up, dtype=float)
    rest = (1 - p_up) / 2
    return pd.DataFrame(
        {"p_down": rest, "p_neutral": rest, "p_up": p_up, "next_ret": next_ret},
        index=pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC"),
    )


def test_always_long_without_costs_equals_buy_and_hold():
    rng = np.random.default_rng(0)
    ret = rng.normal(0.001, 0.03, 300)
    bt = run_backtest(_oof(np.full(300, 0.9), ret), 0.5, 365, fee_bps=0, slippage_bps=0)
    np.testing.assert_allclose(bt.equity["Strategie"], bt.equity["Buy & Hold"])
    assert bt.strategy["n_trades"] == 1
    assert bt.strategy["exposure"] == 1.0


def test_no_signal_stays_flat():
    bt = run_backtest(_oof(np.full(50, 0.4), np.full(50, 0.01)), 0.5, 365)
    assert (bt.equity["Strategie"] == 1.0).all()
    assert bt.strategy["n_trades"] == 0
    assert bt.strategy["total_return"] == 0.0


def test_costs_are_charged_on_entry_and_exit():
    p_up = [0.9, 0.9, 0.2, 0.2]
    ret = [0.0, 0.0, 0.0, 0.0]
    bt = run_backtest(_oof(p_up, ret), 0.5, 365, fee_bps=10, slippage_bps=0)
    assert bt.equity["Strategie"].iloc[-1] == pytest.approx((1 - 0.001) ** 2)
    assert bt.strategy["costs_paid"] == pytest.approx(0.002)
    assert bt.strategy["n_trades"] == 1
    assert bt.strategy["avg_trade"] == pytest.approx((1 - 0.001) ** 2 - 1)


def test_position_is_applied_to_next_bar_return():
    # Signal an Tag 0 → verdient den Return von Tag 0→1 (next_ret[0]), nicht früher
    bt = run_backtest(_oof([0.9, 0.2, 0.2], [0.10, 0.50, 0.50]), 0.5, 365, fee_bps=0, slippage_bps=0)
    assert bt.equity["Strategie"].iloc[-1] == pytest.approx(1.10)


def test_short_and_min_hold():
    oof = _oof([0.9, 0.2, 0.2, 0.2, 0.2], [0.0] * 5)
    oof.loc[:, "p_down"] = [0.05, 0.7, 0.7, 0.1, 0.1]
    oof.loc[:, "p_neutral"] = 1 - oof["p_up"] - oof["p_down"]
    pos = generate_positions(oof, 0.5, allow_short=True)
    assert pos.tolist() == [1, -1, -1, 0, 0]
    held = generate_positions(oof, 0.5, allow_short=False, min_hold_bars=3)
    assert held.tolist() == [1, 1, 1, 0, 0]


def test_performance_stats_known_values():
    r = pd.Series([0.01] * 365)
    s = performance_stats(r, 365)
    assert s["total_return"] == pytest.approx(1.01**365 - 1)
    assert s["cagr"] == pytest.approx(1.01**365 - 1)
    assert s["max_drawdown"] == 0.0
    dd = performance_stats(pd.Series([0.1, -0.5, 0.2]), 365)
    assert dd["max_drawdown"] == pytest.approx(-0.5)
