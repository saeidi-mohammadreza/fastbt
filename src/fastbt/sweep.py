"""Parameter sweeps that give the same answer in parallel as in serial."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Sequence

__all__ = ["run_sweep"]


def run_sweep(fn: Callable[[dict], dict], grid: Sequence[dict] | Iterable[dict],
              workers: int = 1, chunksize: int = 8) -> list[dict]:
    """Evaluate ``fn(params)`` for every params dict; results keep grid order.

    ``fn`` must be a module-level function (picklable) and deterministic: any
    randomness has to be seeded from ``params``. The tests check that a 4-worker
    run returns exactly the serial result.
    """
    grid = list(grid)
    if workers <= 1:
        return [fn(p) for p in grid]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, grid, chunksize=chunksize))
