"""Performance metrics. The annualisation factor is always an explicit argument."""
from __future__ import annotations

import numpy as np

__all__ = ["sharpe", "max_drawdown", "summary"]


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    """Annualised Sharpe ratio (zero risk-free rate): mean/std * sqrt(periods_per_year).

    A per-bar Sharpe of 0.1 on daily bars is 0.1 * sqrt(252) = 1.59 annualised;
    always report which one you quote.
    """
    r = np.asarray(returns, dtype=np.float64)
    s = r.std(ddof=1)
    if not np.isfinite(s) or s == 0:
        return float("nan")
    return float(r.mean() / s * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(e)
    return float(np.min(e / peak - 1.0))


def summary(result, periods_per_year: float) -> dict:
    return {
        "total_return": float(result.equity[-1] - 1.0),
        "sharpe_annualised": sharpe(result.net, periods_per_year),
        "max_drawdown": max_drawdown(result.equity),
        "n_trades": result.n_trades,
        "cost_paid": float(result.cost.sum()),
    }
