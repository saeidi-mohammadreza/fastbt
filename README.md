# fastbt

[![CI](https://github.com/saeidi-mohammadreza/fastbt/actions/workflows/ci.yml/badge.svg)](https://github.com/saeidi-mohammadreza/fastbt/actions/workflows/ci.yml)

**A small, exact bar backtester where the research-hygiene rules are code, not conventions.**

Most backtests that look too good are wrong in one of four ways: the decision used a bar it could not have seen, the fill happened at a price that was never available, the validation leaked labels across the train/test boundary, or the "parallel speed-up" silently changed the numbers. `fastbt` makes each of those a tested property.

| guarantee | how it is enforced |
| --- | --- |
| **Decisions never see the future** | a decision made at bar *t*'s close is filled at bar *t+1*'s open, fixed in the engine; `audit_lookahead()` perturbs every bar after random cut points and requires earlier decisions to stay bit-identical |
| **Explicit fills and costs** | fees and slippage in basis points of turnover, charged at the fill; overnight gap and intrabar legs compounded separately |
| **Leak-free validation** | `purged_walk_forward()` drops training samples whose label window reaches into the test window, plus an embargo |
| **Parallel = serial** | `run_sweep()` returns exactly the serial result with any number of workers (tested) |
| **Two engines, one answer** | a vectorised NumPy engine and a loop engine (Numba-compiled when installed) must agree to the last bit |

```python
import numpy as np
from fastbt import synthetic_ohlc, ema, simulate, Costs, summary, audit_lookahead

bars = synthetic_ohlc(500_000, seed=1)
signal = lambda b: np.sign(ema(b["close"], 20) - ema(b["close"], 100))

audit_lookahead(signal, bars)            # raises LookaheadError if the signal peeks
res = simulate(bars["open"], bars["close"], signal(bars), Costs(fee_bps=1, slippage_bps=1))
summary(res, periods_per_year=252 * 390) # Sharpe annualisation is an argument, never implied
```

## The lookahead audit catches the usual leaks

From `tests/test_causal.py`; each of these is rejected, while EMA crossovers and trailing z-scores pass:

```python
np.convolve(x, np.ones(21) / 21, mode="same")   # centred window: 10 future bars
x - x.mean()                                    # full-sample normalisation
np.append(np.diff(x), 0)                        # tomorrow's change as today's signal
```

## Purged walk-forward CV

```python
from fastbt import purged_walk_forward

for split in purged_walk_forward(n=len(labels), n_splits=5, min_train=50_000,
                                 label_horizon=240, embargo=60):
    model.fit(X[split.train], y[split.train])
    evaluate(model, X[split.test], y[split.test])
```

With a 240-bar forward-return label, the last 240 training samples before each test window would have "seen" test prices; they are removed, and so is a 60-bar embargo.

## Speed

`python examples/benchmark.py` (2 M bars; 16-config sweep of 200k bars each), as measured in CI on GitHub-hosted runners:

| | NumPy only | with Numba |
| --- | ---: | ---: |
| vectorised simulator | 29–35 M bars/s | 44–47 M bars/s |
| loop engine (Numba) | | 94–119 M bars/s, bit-identical to NumPy |
| 16-config sweep, serial → 4 workers | 1.9 s → 1.2 s | 1.6 s → 0.8 s |

Every parallel run returns exactly the serial results (`identical=True` in each job's log).

## Run the tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Scope

Single instrument, bar data, target-position strategies (long/short/fractional), market fills at the next open. No order book, no partial fills, no borrow costs; those belong in a different tool. Synthetic data only: no market data is included.

MIT licence. Author: Mohammadreza Saeidi. A clean-room, generic re-implementation of ideas I use in production backtesting (structural no-lookahead keys, purged walk-forward validation, bit-identical verification of every speed-up); it contains no employer code.
