"""Deterministic synthetic OHLC data (no market data is shipped or needed)."""
from __future__ import annotations

import numpy as np

__all__ = ["synthetic_ohlc"]


def synthetic_ohlc(n: int, seed: int = 0, start: float = 100.0,
                   vol: float = 0.01, regime_len: int = 500) -> dict[str, np.ndarray]:
    """Geometric random walk with drifting regimes and an overnight gap.

    Returns ``open``, ``high``, ``low``, ``close`` arrays of length ``n``.
    """
    rng = np.random.default_rng(seed)
    n_reg = max(1, -(-n // regime_len))
    drift = np.repeat(rng.normal(0.0, vol / 50, n_reg), regime_len)[:n]
    ret = drift + rng.normal(0.0, vol, n)
    close = start * np.exp(np.cumsum(ret))
    gap = rng.normal(0.0, vol / 3, n)
    open_ = np.empty(n)
    open_[0] = start
    open_[1:] = close[:-1] * np.exp(gap[1:])
    span = np.abs(rng.normal(0.0, vol / 2, n))
    high = np.maximum(open_, close) * np.exp(span)
    low = np.minimum(open_, close) * np.exp(-span)
    return {"open": open_, "high": high, "low": low, "close": close}
