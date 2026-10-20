import numpy as np
import pytest

from fastbt import purged_walk_forward, run_sweep
from fastbt.examples_support import evaluate


@pytest.mark.parametrize("expanding", [True, False])
def test_purge_and_embargo(expanding):
    h, emb = 24, 5
    splits = list(purged_walk_forward(5_000, 5, min_train=1_000, label_horizon=h,
                                      embargo=emb, expanding=expanding))
    assert len(splits) == 5
    prev_hi = None
    for s in splits:
        lo = s.test[0]
        # every training label window [i, i+h] ends before the embargo gap
        assert s.train.max() + h < lo - emb
        assert np.all(np.diff(s.test) == 1)
        if prev_hi is not None:
            assert lo == prev_hi  # test windows tile the sample without overlap
        prev_hi = s.test[-1] + 1
    assert prev_hi == 5_000


def test_invalid_arguments():
    with pytest.raises(ValueError):
        list(purged_walk_forward(100, 3, min_train=100, label_horizon=1))


def test_parallel_sweep_is_identical_to_serial():
    grid = [{"fast": f, "slow": s, "seed": 11} for f in (5, 10, 20) for s in (50, 100)]
    serial = run_sweep(evaluate, grid, workers=1)
    parallel = run_sweep(evaluate, grid, workers=4, chunksize=1)
    assert serial == parallel
