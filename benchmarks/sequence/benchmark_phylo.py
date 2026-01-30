import numpy as np
import pytest
import biotite.sequence.phylo as phylo

N = 20


@pytest.fixture(scope="module")
def distances():
    np.random.seed(0)
    rand = np.random.rand(N, N).astype(np.float32)
    distances = (rand + rand.T) / 2
    np.fill_diagonal(distances, 0)
    return distances


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "method",
    [phylo.upgma, phylo.neighbor_joining],
    ids=lambda x: x.__name__,
)
def benchmark_clustering(distances, method):
    """
    Perform hierarchical clustering from a distance matrix.
    """
    method(distances)
