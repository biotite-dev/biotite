import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
from os.path import join
from ..util import data_dir
from biotite.structure.basepairs import _get_proximate_basepair_candidates, \
                                        get_basepairs


def convert_indices_to_res_chain_id(atomarray, indices):
    res_chain_id = []
    for base1, base2 in indices:
        res_chain_id.append(
            ((atomarray[base1].chain_id, atomarray[base1].res_id),
            (atomarray[base2].chain_id, atomarray[base2].res_id))
            )
    return res_chain_id


def reversed_iterator(iter):
    return reversed(list(iter))


#TODO: Remove tests for private functions
def test_get_proximate_basepair_candidates():
    nuc_sample_array = strucio.load_structure(
        join(data_dir("structure"), "5ugo.cif")
    )
    
    assert ( len(_get_proximate_basepair_candidates(nuc_sample_array))
                == 128 )


@pytest.fixture
def nuc_sample_array():
    return strucio.load_structure(join(data_dir("structure"), "1qxb.cif"))


@pytest.fixture
def basepairs_fw(nuc_sample_array):
    residue_indices = struc.residues.get_residue_starts(nuc_sample_array)[0:24]
    basepairs = []
    for i in range(12):
        basepairs.append((residue_indices[i], residue_indices[-1*(i+1)]))
    basepairs = convert_indices_to_res_chain_id(nuc_sample_array, basepairs)
    return basepairs

@pytest.fixture
def basepairs_rv(basepairs_fw):
    reverser = []
    for base1, base2 in basepairs_fw:
        reverser.append((base2, base1))
    return reverser

def check_output(computed_basepairs, basepairs_fw, basepairs_rv):
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


def test_get_basepairs_forward(nuc_sample_array, basepairs_fw, basepairs_rv):
    computed_basepairs = get_basepairs(nuc_sample_array)
    check_output(convert_indices_to_res_chain_id(
        nuc_sample_array, computed_basepairs), basepairs_fw, basepairs_rv
            )


def test_get_basepairs_reverse(nuc_sample_array, basepairs_fw, basepairs_rv):
    # Reverse sequence of residues in nuc_sample_array
    reversed_nuc_sample_array = struc.AtomArray(0) 
    for residue in reversed_iterator(struc.residue_iter(nuc_sample_array)):
        reversed_nuc_sample_array = reversed_nuc_sample_array + residue
    
    computed_basepairs = get_basepairs(reversed_nuc_sample_array)
    computed_basepairs = convert_indices_to_res_chain_id(
        reversed_nuc_sample_array, computed_basepairs
                                                    )
    check_output(computed_basepairs, basepairs_fw, basepairs_rv)
