# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import warnings
import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.mmtf as mmtf
from .util import data_dir


@pytest.mark.xfail(raises=ImportError)
@pytest.mark.parametrize("pdb_id", ["1l2y", "1gya"])
def test_single(pdb_id):
    file_name = join(data_dir, pdb_id+".pdb")
    
    # Single atom SASA, compare with MDTraj
    file = pdb.PDBFile()
    file.read(file_name)
    array = file.get_structure(model=1)
    sasa = struc.sasa(array, vdw_radii="Single", point_number=5000)
    
    from biotite.structure.sasa import _single_radii as radii
    import mdtraj
    # Use the same atom radii
    radii = {element.capitalize() : radius / 10
             for element, radius in radii.items()}
    with warnings.catch_warnings():
        # Ignore warning about dummy unit cell vector
        warnings.simplefilter("ignore")
        traj = mdtraj.load(file_name)
    # Conversion from nm^2 to A^2
    sasa_exp = mdtraj.shrake_rupley(
        traj, change_radii=radii, n_sphere_points=5000
    )[0] * 100

    
    # Assert that more than 90% of atoms
    # have less than 10% SASA difference
    assert np.count_nonzero(
        np.isclose(sasa, sasa_exp, rtol=1e-1, atol=1e-1)
    ) / len(sasa) > 0.9
    # Assert that more than 98% of atoms
    # have less than 1% SASA difference
    assert np.count_nonzero(
        np.isclose(sasa, sasa_exp, rtol=1e-2, atol=1e-1)
    ) / len(sasa) > 0.98


@pytest.mark.parametrize("pdb_id", ["1l2y", "1gya"])
def test_coarse_grained(pdb_id):
    # Multi atom SASA (ProtOr), compare with single atom SASA
    # on residue level
    file = mmtf.MMTFFile()
    file.read(join(data_dir, pdb_id+".mmtf"))
    array = mmtf.get_structure(file, model=1)
    array = array[struc.filter_amino_acids(array)]
    sasa = struc.apply_residue_wise(
        array, struc.sasa(array, vdw_radii="ProtOr"), np.nansum
    )
    sasa_exp = struc.apply_residue_wise(
        array, struc.sasa(array, vdw_radii="Single"), np.nansum
    )

    # Assert that more than 90% of atoms
    # have less than 20% SASA difference
    assert np.count_nonzero(
        np.isclose(sasa, sasa_exp, rtol=2e-1, atol=1)
    ) / len(sasa) > 0.9
    # Assert that more than 98% of atoms
    # have less than 50% SASA difference
    assert np.count_nonzero(
        np.isclose(sasa, sasa_exp, rtol=5e-1, atol=1)
    ) / len(sasa) > 0.98