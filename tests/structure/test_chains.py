# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
from os.path import join
from ..util import data_dir
import pytest


@pytest.fixture
def array():
    return strucio.load_structure(join(data_dir("structure"), "1igy.mmtf"))

def test_get_chain_starts(array):
    _, ref_starts = np.unique(array.chain_id, return_index=True)
    test_starts = struc.get_chain_starts(array)
    # All first occurences of a chain id are automatically chain starts
    assert set(ref_starts).issubset(set(test_starts))

def test_get_chains(array):
    assert struc.get_chains(array).tolist() == ["A", "B", "C", "D", "E", "F"]

def test_get_chain_count(array):
    assert struc.get_chain_count(array) == 6

def test_chain_iter(array):
    n = 0
    for chain in struc.get_chains(array):
        n += 1
        assert isinstance(array, struc.AtomArray)
    assert n == 6