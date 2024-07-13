# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import json
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.mark.parametrize("pdb_id", ["1l2y", "1gya"])
def test_sasa_consistency(pdb_id):
    """
    Check that SASA computation for a single model reproduces results from MDTraj.
    """
    # Load precomputed hydrogen bond triplets from MDTraj
    with open(join(data_dir("structure"), "misc", "sasa.json")) as file:
        ref_data = json.load(file)
    ref_sasa = np.array(ref_data[pdb_id])

    file = pdb.PDBFile.read(join(data_dir("structure"), pdb_id + ".pdb"))
    array = file.get_structure(model=1)
    test_sasa = struc.sasa(array, vdw_radii="Single", point_number=5000)

    # Assert that all atoms have less than 10% SASA difference
    assert np.all(np.isclose(test_sasa, ref_sasa, rtol=1e-1, atol=1e-1))
    # Assert that more than 98% of atoms have less than 1% SASA difference
    assert (
        np.count_nonzero(np.isclose(test_sasa, ref_sasa, rtol=1e-2, atol=1e-1))
        / len(test_sasa)
        > 0.98
    )


@pytest.mark.parametrize("pdb_id", ["1l2y", "1gya"])
def test_coarse_grained(pdb_id):
    # Multi atom SASA (ProtOr), compare with single atom SASA
    # on residue level
    file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), pdb_id + ".bcif"))
    array = pdbx.get_structure(file, model=1)
    array = array[struc.filter_amino_acids(array)]
    sasa = struc.apply_residue_wise(
        array, struc.sasa(array, vdw_radii="ProtOr"), np.nansum
    )
    sasa_exp = struc.apply_residue_wise(
        array, struc.sasa(array, vdw_radii="Single"), np.nansum
    )

    # Assert that more than 90% of atoms
    # have less than 10% SASA difference
    assert (
        np.count_nonzero(np.isclose(sasa, sasa_exp, rtol=1e-1, atol=1)) / len(sasa)
        > 0.9
    )
    # Assert that more than 98% of atoms
    # have less than 40% SASA difference
    assert (
        np.count_nonzero(np.isclose(sasa, sasa_exp, rtol=4e-1, atol=1)) / len(sasa)
        > 0.98
    )
