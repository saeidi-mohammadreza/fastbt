"""Throughput of the simulator and the speed-up of a parallel sweep.

    python examples/benchmark.py
"""
import os
import time

import numpy as np

from fastbt import HAVE_NUMBA, Costs, ema, run_sweep, simulate, simulate_loop, synthetic_ohlc
from fastbt.examples_support import evaluate


def best_of(fn, k=5):
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    n = 2_000_000
    b = synthetic_ohlc(n, seed=0)
    tgt = np.sign(ema(b["close"], 20) - ema(b["close"], 100))
    t_vec = best_of(lambda: simulate(b["open"], b["close"], tgt, Costs()))
    print(f"simulate (NumPy, vectorised): {n / t_vec / 1e6:6.1f} M bars/s")
    if HAVE_NUMBA:
        simulate_loop(b["open"][:10], b["close"][:10], tgt[:10])  # compile
        t_loop = best_of(lambda: simulate_loop(b["open"], b["close"], tgt, Costs()))
        print(f"simulate_loop (Numba):        {n / t_loop / 1e6:6.1f} M bars/s")
        a, c = simulate(b["open"], b["close"], tgt), simulate_loop(b["open"], b["close"], tgt)
        print("identical:", all(np.array_equal(getattr(a, k), getattr(c, k))
                                for k in ("position", "net", "equity")))

    grid = [{"fast": f, "slow": s, "seed": 1, "n": 200_000}
            for f in (5, 8, 13, 21) for s in (34, 55, 89, 144)]
    t0 = time.perf_counter(); serial = run_sweep(evaluate, grid, workers=1); t1 = time.perf_counter()
    w = min(4, os.cpu_count() or 1)
    par = run_sweep(evaluate, grid, workers=w, chunksize=2); t2 = time.perf_counter()
    print(f"sweep of {len(grid)} configs: serial {t1 - t0:.2f}s, {w} workers {t2 - t1:.2f}s, "
          f"identical={serial == par}")


if __name__ == "__main__":
    main()
