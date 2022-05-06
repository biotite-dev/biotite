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
    """
    Compare :func:`test_get_chain_starts()` with :func:`np.unique` in a
    case where chain ID unambiguously identify chains.
    """
    _, ref_starts = np.unique(array.chain_id, return_index=True)
    test_starts = struc.get_chain_starts(array)
    # All first occurences of a chain id are automatically chain starts
    assert set(ref_starts).issubset(set(test_starts))

def test_get_chain_starts_same_id(array):
    """
    Expect correct number of chains in a case where two successive
    chains have the same chain ID (as possible in an assembly).
    """
    # Concatenate two chains with same ID
    array = array[array.chain_id == "A"]
    merged = array + array
    assert struc.get_chain_starts(merged).tolist() == [0, array.array_length()]

def test_apply_chain_wise(array):
    data = struc.apply_chain_wise(array, np.ones(len(array)), np.sum)
    assert data.tolist() == [
        len(array[array.chain_id == chain_id])
        for chain_id in np.unique(array.chain_id)
    ]

def test_spread_chain_wise(array):
    input_data = np.unique(array.chain_id)
    output_data = struc.spread_chain_wise(array, input_data)
    assert output_data.tolist() == array.chain_id.tolist()

def test_get_chain_masks(array):
    SAMPLE_SIZE = 100
    np.random.seed(0)
    indices = np.random.randint(0, array.array_length(), SAMPLE_SIZE)
    test_masks = struc.get_chain_masks(array, indices)
    for index, test_mask in zip(indices, test_masks):
        ref_mask = array.chain_id == array.chain_id[index]
        assert test_mask.tolist() == ref_mask.tolist()

def test_get_chain_starts_for(array):
    SAMPLE_SIZE = 100
    np.random.seed(0)
    indices = np.random.randint(0, array.array_length(), SAMPLE_SIZE)
    ref_starts = np.array(
        [np.where(mask)[0][0] for mask
         in struc.get_chain_masks(array, indices)]
    )
    test_starts = struc.get_chain_starts_for(array, indices)
    assert test_starts.tolist() == ref_starts.tolist()

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