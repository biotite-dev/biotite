# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from ..util import cannot_import, data_dir


# Ignore warning about dummy unit cell vector
@pytest.mark.filterwarnings("ignore")
@pytest.mark.skipif(cannot_import("mdtraj"), reason="MDTraj is not installed")
@pytest.mark.parametrize("pdb_id", ["1l2y", "1gya"])
def test_single(pdb_id):
    file_name = join(data_dir("structure"), pdb_id + ".pdb")

    # Single atom SASA, compare with MDTraj
    file = pdb.PDBFile.read(file_name)
    array = file.get_structure(model=1)
    sasa = struc.sasa(array, vdw_radii="Single", point_number=5000)

    import mdtraj
    from biotite.structure.info.radii import _SINGLE_RADII as radii

    # Use the same atom radii
    radii = {element.capitalize(): radius / 10 for element, radius in radii.items()}
    traj = mdtraj.load(file_name)
    # Conversion from nm^2 to A^2
    sasa_exp = (
        mdtraj.shrake_rupley(traj, change_radii=radii, n_sphere_points=5000)[0] * 100
    )

    # Assert that more than 90% of atoms
    # have less than 10% SASA difference
    assert (
        np.count_nonzero(np.isclose(sasa, sasa_exp, rtol=1e-1, atol=1e-1)) / len(sasa)
        > 0.9
    )
    # Assert that more than 98% of atoms
    # have less than 1% SASA difference
    assert (
        np.count_nonzero(np.isclose(sasa, sasa_exp, rtol=1e-2, atol=1e-1)) / len(sasa)
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
