# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
from os.path import join
from ..util import data_dir
import itertools
from biotite.structure.basepairs import _get_proximate_basepair_candidates, \
                                        base_pairs
import numpy as np

def convert_indices_to_res_chain_id(atomarray, indices):
    """
    Convert a list of tuples, containing the first indices of the
    base residues, to a list of tuples containing the residue ids and
    chain ids of the bases.
    """
    res_chain_id = []
    for base1, base2 in indices:
        res_chain_id.append(
            ((atomarray[base1].chain_id, atomarray[base1].res_id),
            (atomarray[base2].chain_id, atomarray[base2].res_id))
            )
    return res_chain_id


def reversed_iterator(iter):
    """
    Returns a reversed list of the elements of an Iterator.
    """
    return reversed(list(iter))

@pytest.fixture
def nuc_sample_array():
    return strucio.load_structure(join(data_dir("structure"), "1qxb.cif"))

@pytest.fixture
def nuc_sample_array_no_hydrogens(nuc_sample_array):
    return nuc_sample_array[
        np.isin(nuc_sample_array.element, ["H"], invert=True)
    ]

@pytest.fixture
def basepairs_fw(nuc_sample_array):
    """
    Generate a test output for the base_pairs function.
    """
    residue_indices = struc.residues.get_residue_starts(nuc_sample_array)[0:24]
    basepairs = []
    for i in range(12):
        basepairs.append((residue_indices[i], residue_indices[-1*(i+1)]))
    basepairs = convert_indices_to_res_chain_id(nuc_sample_array, basepairs)
    return basepairs

@pytest.fixture
def basepairs_rv(basepairs_fw):
    """
    Generate a reversed test output for the base_pairs function.
    """
    reverser = []
    for base1, base2 in basepairs_fw:
        reverser.append((base2, base1))
    return reverser

def check_output(computed_basepairs, basepairs_fw, basepairs_rv):
    """
    Check the output of base_pairs.
    """

    # Check if basepairs are unique in computed_basepairs
    seen = set()
    assert (not any(
        (base1, base2) in seen) or (base2, base1 in seen)
        or seen.add((base1, base2)) for base1, base2 in computed_basepairs
        )
    # Check if the right number of basepairs is in computed_baspairs
    assert(len(computed_basepairs) == len(basepairs_fw))
    # Check if the right basepairs are in computed_basepairs
    for comp_basepair in computed_basepairs:
        assert ((comp_basepair in basepairs_fw) \
                or (comp_basepair in basepairs_rv))
"""
@pytest.mark.parametrize(
    "atom_array, test_unique",
    itertools.product([nuc_sample_array, nuc_sample_array_no_hydrogens],
                      [True, False])
)
"""

def test_base_pairs_forward(nuc_sample_array_no_hydrogens, basepairs_fw, basepairs_rv):
    """
    Test for the function base_pairs.
    """
    computed_basepairs = base_pairs(nuc_sample_array_no_hydrogens, unique=True)
    check_output(convert_indices_to_res_chain_id(
        nuc_sample_array_no_hydrogens, computed_basepairs), basepairs_fw, basepairs_rv
            )


def test_base_pairs_reverse(nuc_sample_array, basepairs_fw, basepairs_rv):
    """
    Reverse the order of residues in the atom_array and then test the
    function base_pairs.
    """
    
    # Reverse sequence of residues in nuc_sample_array
    reversed_nuc_sample_array = struc.AtomArray(0) 
    for residue in reversed_iterator(struc.residue_iter(nuc_sample_array)):
        reversed_nuc_sample_array = reversed_nuc_sample_array + residue
    
    computed_basepairs = base_pairs(reversed_nuc_sample_array)
    computed_basepairs = convert_indices_to_res_chain_id(
        reversed_nuc_sample_array, computed_basepairs
                                                    )
    check_output(computed_basepairs, basepairs_fw, basepairs_rv)

