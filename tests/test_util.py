import numpy as np
import pytest
from biotite.util import map_unique


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n_arrays", [1, 2, 3])
@pytest.mark.parametrize("n_results", [1, 2])
def test_map_unique(seed, n_arrays, n_results):
    """
    Check that :func:`map_unique()` gives the same result as applying the
    function to each element.
    """
    LENGTH = 1000
    N_UNIQUE = 5

    def function(*values):
        if n_results == 1:
            return "-".join(values)
        else:
            return tuple(f"{i}:" + "-".join(values) for i in range(n_results))

    rng = np.random.default_rng(seed)
    arrays = [
        rng.choice(np.array([f"{i}_{j}" for j in range(N_UNIQUE)]), LENGTH)
        for i in range(n_arrays)
    ]

    test_mapped = map_unique(function, *arrays)
    ref_mapped = np.array([function(*values) for values in zip(*arrays)])

    assert test_mapped.tolist() == ref_mapped.tolist()
