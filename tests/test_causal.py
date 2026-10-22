import numpy as np
import pytest

from fastbt import LookaheadError, audit_lookahead, ema, rolling_mean, synthetic_ohlc, zscore

BARS = synthetic_ohlc(3_000, seed=3)


def test_causal_signals_pass_the_audit():
    audit_lookahead(lambda b: np.sign(ema(b["close"], 12) - ema(b["close"], 48)), BARS)
    audit_lookahead(lambda b: np.sign(-zscore(b["close"], 50)), BARS)


def test_centred_window_is_caught():
    def leaky(b):
        x = b["close"]
        centred = np.convolve(x, np.ones(21) / 21, mode="same")  # uses 10 future bars
        return np.sign(x - centred)
    with pytest.raises(LookaheadError):
        audit_lookahead(leaky, BARS)


def test_full_sample_normalisation_is_caught():
    with pytest.raises(LookaheadError):
        audit_lookahead(lambda b: np.sign(b["close"] - b["close"].mean()), BARS)


def test_peeking_at_next_close_is_caught():
    with pytest.raises(LookaheadError):
        audit_lookahead(lambda b: np.sign(np.append(np.diff(b["close"]), 0.0)), BARS)


def test_rolling_mean_matches_naive():
    x = np.arange(10, dtype=float)
    m = rolling_mean(x, 3)
    assert np.isnan(m[:2]).all() and m[2:].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_ema_of_empty_input_is_empty():
    assert ema(np.array([]), 10).shape == (0,)


def test_audit_needs_enough_bars():
    with pytest.raises(ValueError):
        audit_lookahead(lambda b: b["close"], synthetic_ohlc(10, seed=0))
