"""Bar-based simulation with an explicit fill and cost model.

Timing convention (the one most leaks come from, so it is fixed here, not left
to each strategy):

* A decision for bar ``t`` may use information up to and including the **close**
  of bar ``t``.
* It is executed at the **open of bar t+1**.
* PnL of bar ``t+1`` is therefore ``position[t] * (close[t+1] / open[t+1] - 1)``
  on the bar it was entered, and ``position * (close / prev_close - 1)`` on bars
  where the position is held through the open.

Two implementations are provided: a vectorised NumPy one (``simulate``) and a
plain loop (``simulate_loop``) that Numba compiles when it is installed. Tests
require them to agree exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # optional acceleration
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised when numba is absent
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

__all__ = ["Costs", "Result", "simulate", "simulate_loop", "HAVE_NUMBA"]


@dataclass(frozen=True)
class Costs:
    """Per-side costs in basis points of traded notional."""

    fee_bps: float = 1.0
    slippage_bps: float = 1.0

    @property
    def per_unit_turnover(self) -> float:
        return (self.fee_bps + self.slippage_bps) * 1e-4


@dataclass
class Result:
    position: np.ndarray   # position held during each bar (after the open fill)
    gross: np.ndarray      # gross return of each bar
    cost: np.ndarray       # cost charged at each bar's open
    net: np.ndarray        # net return of each bar
    equity: np.ndarray     # compounded equity, starts at 1.0

    @property
    def n_trades(self) -> int:
        return int(np.count_nonzero(np.diff(self.position, prepend=0.0)))


def _validate(open_: np.ndarray, close: np.ndarray, target: np.ndarray) -> None:
    if not (open_.shape == close.shape == target.shape) or open_.ndim != 1:
        raise ValueError("open, close and target must be 1-D arrays of equal length")
    if np.any(~np.isfinite(open_)) or np.any(~np.isfinite(close)):
        raise ValueError("prices must be finite")
    if np.any(open_ <= 0) or np.any(close <= 0):
        raise ValueError("prices must be positive")
    if np.any(~np.isfinite(target)):
        raise ValueError("target positions must be finite (use 0 for flat)")


def simulate(open_: np.ndarray, close: np.ndarray, target: np.ndarray,
             costs: Costs = Costs()) -> Result:
    """Vectorised simulation. ``target[t]`` is the position decided at the close of
    bar ``t`` (e.g. +1 long, -1 short, 0.5 half long); it is filled at ``open[t+1]``."""
    open_ = np.asarray(open_, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    _validate(open_, close, target)
    n = len(close)
    pos = np.zeros(n)
    pos[1:] = target[:-1]                       # filled at the next open
    prev = np.zeros(n)
    prev[1:] = pos[:-1]                         # held into this bar's open
    # held overnight part: prev position earns close[t-1] -> open[t]
    gap = np.zeros(n)
    gap[1:] = open_[1:] / close[:-1] - 1.0
    intrabar = close / open_ - 1.0
    # the overnight leg and the intrabar leg compound, exactly as in the loop
    gross = (1.0 + prev * gap) * (1.0 + pos * intrabar) - 1.0
    cost = np.abs(pos - prev) * costs.per_unit_turnover
    net = (1.0 + gross) * (1.0 - cost) - 1.0
    equity = np.cumprod(1.0 + net)
    return Result(pos, gross, cost, net, equity)


@njit(cache=False)
def _loop(open_, close, target, c):
    n = close.shape[0]
    pos = np.zeros(n)
    gross = np.zeros(n)
    cost = np.zeros(n)
    net = np.zeros(n)
    equity = np.zeros(n)
    held = 0.0
    eq = 1.0
    for t in range(n):
        # 1) the position carried from the previous bar earns the overnight gap
        g1 = 0.0
        if t > 0:
            g1 = held * (open_[t] / close[t - 1] - 1.0)
        # 2) at the open, trade to the target decided at the previous close
        new = target[t - 1] if t > 0 else 0.0
        cst = abs(new - held) * c
        # 3) the new position earns open -> close
        g2 = new * (close[t] / open_[t] - 1.0)
        gr = (1.0 + g1) * (1.0 + g2) - 1.0
        nt = (1.0 + gr) * (1.0 - cst) - 1.0
        eq *= 1.0 + nt
        pos[t] = new
        gross[t] = gr
        cost[t] = cst
        net[t] = nt
        equity[t] = eq
        held = new
    return pos, gross, cost, net, equity


def simulate_loop(open_: np.ndarray, close: np.ndarray, target: np.ndarray,
                  costs: Costs = Costs()) -> Result:
    """Reference loop implementation (Numba-compiled when available)."""
    open_ = np.ascontiguousarray(open_, dtype=np.float64)
    close = np.ascontiguousarray(close, dtype=np.float64)
    target = np.ascontiguousarray(target, dtype=np.float64)
    _validate(open_, close, target)
    return Result(*_loop(open_, close, target, costs.per_unit_turnover))
