"""Causal indicators and a lookahead audit.

A signal function maps bars -> one decision per bar. It is *causal* when the
decision at bar t does not change if any bar after t changes. ``audit_lookahead``
checks that property empirically for any signal function: it replaces the future
after random cut points with noise and requires the past decisions to be
bit-identical. It catches the classic leaks: centred windows, ``shift(-1)``,
full-sample normalisation, and fitting on the whole series.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["ema", "rolling_mean", "rolling_std", "zscore", "audit_lookahead", "LookaheadError"]


class LookaheadError(AssertionError):
    pass


def ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average seeded with the first value (causal)."""
    if span < 1:
        raise ValueError("span must be >= 1")
    a = 2.0 / (span + 1.0)
    out = np.empty(len(x))
    acc = float(x[0])
    for i, v in enumerate(np.asarray(x, dtype=np.float64)):
        acc = v if i == 0 else a * v + (1.0 - a) * acc
        out[i] = acc
    return out


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last ``w`` values; NaN until ``w`` values exist."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan)
    if w <= len(x):
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan)
    if w <= len(x):
        win = np.lib.stride_tricks.sliding_window_view(x, w)
        out[w - 1:] = win.std(axis=1, ddof=1)
    return out


def zscore(x: np.ndarray, w: int) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - rolling_mean(x, w)) / rolling_std(x, w)


def audit_lookahead(signal_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
                    bars: dict[str, np.ndarray], n_cuts: int = 20, seed: int = 0) -> None:
    """Raise ``LookaheadError`` if any decision depends on a later bar."""
    rng = np.random.default_rng(seed)
    base = np.asarray(signal_fn(bars))
    n = len(base)
    cuts = rng.integers(n // 10, n - 1, size=n_cuts)
    for t in cuts:
        perturbed = {}
        for k, v in bars.items():
            v2 = np.array(v, dtype=np.float64, copy=True)
            v2[t + 1:] = v2[t + 1:] * np.exp(rng.normal(0.0, 0.05, n - t - 1))
            perturbed[k] = v2
        s = np.asarray(signal_fn(perturbed))
        same = (s[: t + 1] == base[: t + 1]) | (np.isnan(s[: t + 1]) & np.isnan(base[: t + 1]))
        if not np.all(same):
            first = int(np.argmin(same))
            raise LookaheadError(
                f"decision at bar {first} changed when only bars > {t} were modified"
            )
