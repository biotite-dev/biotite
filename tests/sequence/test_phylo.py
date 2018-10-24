# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite.sequence.phylo as phylo
from .util import data_dir

@pytest.fixture
def distances():
    # Distances are based on the example
    # "Dendrogram of the BLOSUM62 matrix"
    return np.loadtxt(join(data_dir, "distances.txt"), dtype=int)

def test_upgma(distances):
    pass