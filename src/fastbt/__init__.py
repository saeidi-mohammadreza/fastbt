"""fastbt: a small, exact bar backtester with research hygiene built in."""
from .causal import LookaheadError, audit_lookahead, ema, rolling_mean, rolling_std, zscore
from .cv import Split, purged_walk_forward
from .data import synthetic_ohlc
from .engine import HAVE_NUMBA, Costs, Result, simulate, simulate_loop
from .metrics import max_drawdown, sharpe, summary
from .sweep import run_sweep

__all__ = [
    "Costs", "Result", "simulate", "simulate_loop", "HAVE_NUMBA",
    "ema", "rolling_mean", "rolling_std", "zscore", "audit_lookahead", "LookaheadError",
    "Split", "purged_walk_forward", "synthetic_ohlc",
    "sharpe", "max_drawdown", "summary", "run_sweep",
]
__version__ = "0.1.0"
