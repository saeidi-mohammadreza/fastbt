import numpy as np
import pytest

from fastbt import Costs, simulate, simulate_loop, synthetic_ohlc, ema


@pytest.fixture(scope="module")
def bars():
    return synthetic_ohlc(20_000, seed=7)


def crossover(b):
    return np.sign(ema(b["close"], 10) - ema(b["close"], 50))


def test_vectorised_equals_loop_bit_for_bit(bars):
    tgt = crossover(bars)
    a = simulate(bars["open"], bars["close"], tgt, Costs(2, 3))
    b = simulate_loop(bars["open"], bars["close"], tgt, Costs(2, 3))
    for k in ("position", "gross", "cost", "net", "equity"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k


def test_decision_is_filled_at_next_open():
    o = np.array([100.0, 110.0, 121.0])
    c = np.array([100.0, 110.0, 121.0])
    r = simulate(o, c, np.array([1.0, 1.0, 1.0]), Costs(0, 0))
    assert r.position.tolist() == [0.0, 1.0, 1.0]
    # entered at the open of bar 1 (110) and open == close on that bar: no gain yet
    assert r.gross[1] == 0.0
    # held from close 110 through open 121 -> +10%
    assert r.gross[2] == pytest.approx(0.10)


def test_costs_charged_on_turnover_only():
    o = c = np.full(5, 100.0)
    r = simulate(o, c, np.array([1, 1, -1, 0, 0], float), Costs(fee_bps=5, slippage_bps=5))
    # turnover per bar: 0, 1 (enter), 0, 2 (flip), 1 (exit)
    assert np.allclose(r.cost, np.array([0, 1, 0, 2, 1]) * 10e-4)
    assert r.n_trades == 3


def test_flat_strategy_has_flat_equity(bars):
    r = simulate(bars["open"], bars["close"], np.zeros(len(bars["close"])))
    assert np.all(r.equity == 1.0)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        simulate(np.ones(3), np.ones(4), np.ones(3))
    with pytest.raises(ValueError):
        simulate(np.array([1.0, np.nan]), np.ones(2), np.ones(2))
    with pytest.raises(ValueError):
        simulate(np.ones(2), np.ones(2), np.array([1.0, np.inf]))
