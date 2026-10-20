"""A picklable example objective used by the README, the examples and the tests."""
from __future__ import annotations

import numpy as np

from .causal import ema
from .data import synthetic_ohlc
from .engine import Costs, simulate
from .metrics import summary


def evaluate(params: dict) -> dict:
    """Moving-average crossover on seeded synthetic data; deterministic in ``params``."""
    bars = synthetic_ohlc(params.get("n", 50_000), seed=params["seed"])
    c = bars["close"]
    tgt = np.sign(ema(c, params["fast"]) - ema(c, params["slow"]))
    res = simulate(bars["open"], c, tgt, Costs(params.get("fee_bps", 1.0), params.get("slip_bps", 1.0)))
    out = summary(res, periods_per_year=params.get("ppy", 252))
    return {**params, **out}
