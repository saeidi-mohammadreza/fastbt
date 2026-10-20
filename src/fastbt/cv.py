"""Purged walk-forward cross-validation with an embargo.

When a label at bar i is computed from bars i..i+h (a forward return over h bars),
a training sample close to the test window "sees" test-period prices through its
label. Purging drops training samples whose label window overlaps the test window;
the embargo additionally drops a margin of samples *after* each test window, which
matters once training data may follow the test window (k-fold) and is kept here so
the same splitter is safe either way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

__all__ = ["Split", "purged_walk_forward"]


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray


def purged_walk_forward(n: int, n_splits: int, *, min_train: int, label_horizon: int,
                        embargo: int = 0, expanding: bool = True) -> Iterator[Split]:
    """Yield chronological splits over ``n`` samples.

    The span after ``min_train`` is cut into ``n_splits`` consecutive test windows.
    Training uses samples before each window (all of them if ``expanding``, else a
    window of ``min_train``), minus the last ``label_horizon`` samples whose labels
    reach into the test window, minus ``embargo`` further samples.
    """
    if n_splits < 1 or min_train < 1 or label_horizon < 0 or embargo < 0:
        raise ValueError("invalid arguments")
    if min_train >= n:
        raise ValueError("min_train must be smaller than n")
    edges = np.linspace(min_train, n, n_splits + 1).astype(int)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        train_end = lo - label_horizon - embargo  # exclusive
        train_start = 0 if expanding else max(0, lo - min_train)
        if train_end <= train_start:
            continue
        yield Split(np.arange(train_start, train_end), np.arange(lo, hi))
