# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
from tests.util import data_dir


@pytest.fixture
def array():
    return strucio.load_structure(join(data_dir("structure"), "1l2y.bcif"))[0]


def test_apply_residue_wise(array):
    data = struc.apply_residue_wise(array, np.ones(len(array)), np.sum)
    assert data.tolist() == [len(array[array.res_id == i]) for i in range(1, 21)]


def test_spread_residue_wise(array):
    input_data = np.arange(1, 21)
    output_data = struc.spread_residue_wise(array, input_data)
    assert output_data.tolist() == array.res_id.tolist()


def test_get_residue_masks(array):
    SAMPLE_SIZE = 100
    np.random.seed(0)
    indices = np.random.randint(0, array.array_length(), SAMPLE_SIZE)
    masks = struc.get_residue_masks(array, indices)
    for index, mask in zip(indices, masks):
        ref_mask = array.res_id == array.res_id[index]
        assert mask.tolist() == ref_mask.tolist()


def test_get_residue_starts_for(array):
    SAMPLE_SIZE = 100
    np.random.seed(0)
    indices = np.random.randint(0, array.array_length(), SAMPLE_SIZE)
    ref_starts = np.array(
        [np.where(mask)[0][0] for mask in struc.get_residue_masks(array, indices)]
    )
    test_starts = struc.get_residue_starts_for(array, indices)
    assert test_starts.tolist() == ref_starts.tolist()


def test_get_residues(array):
    ids, names = struc.get_residues(array)
    assert ids.tolist() == list(range(1, 21))
    assert names.tolist() == [
        "ASN",
        "LEU",
        "TYR",
        "ILE",
        "GLN",
        "TRP",
        "LEU",
        "LYS",
        "ASP",
        "GLY",
        "GLY",
        "PRO",
        "SER",
        "SER",
        "GLY",
        "ARG",
        "PRO",
        "PRO",
        "PRO",
        "SER",
    ]
    assert len(ids) == struc.get_residue_count(array)


@pytest.mark.parametrize("seed", range(10))
def test_get_residue_positions(array, seed):
    """
    The positions returned by :func:`get_residue_positions()` with some ``indices``,
    should be the same as indexing the return value of
    :func:`get_all_residue_positions()` with the given indices.
    """
    SIZE = 100

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.array_length(), size=SIZE)

    all_positions = struc.get_all_residue_positions(array)
    ref_positions = all_positions[indices]

    test_positions = struc.get_residue_positions(array, indices)

    assert test_positions.tolist() == ref_positions.tolist()


def test_residue_iter(array):
    centroid = [struc.centroid(res).tolist() for res in struc.residue_iter(array)]
    ref_centroid = struc.apply_residue_wise(array, array.coord, np.average, axis=0)
    assert centroid == ref_centroid.tolist()


def test_get_atom_name_indices(array):
    """
    The indices returned by `get_atom_name_indices()` should only point to one atom per
    residue (for a peptide chain) and should always point to an atom with the
    requested atom name.
    """
    ATOM_NAMES = ("CA", "CB")

    for atom_name, indices in zip(
        ATOM_NAMES, struc.get_atom_name_indices(array, ATOM_NAMES).T, strict=True
    ):
        # There should be one index per residue
        # (including residues where the atom is not present)
        assert len(indices) == struc.get_residue_count(array)
        indices = indices[indices != -1]
        # The indices should capture all atoms in the structure with the requested name
        assert len(indices) == np.count_nonzero(array.atom_name == atom_name)
        # All indices should actually point to atoms with the requested name
        assert np.all(array.atom_name[indices] == atom_name)
        # All indices should point to unique residues
        assert len(set(array.res_id[indices])) == len(indices)
