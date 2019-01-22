# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.info as strucinfo
from biotite.structure.io import load_structure, save_structure
from .util import data_dir


def test_mass():
    """
    Test whether the mass of a residue is approximately the same as the
    sum of the masses of its contained atoms.
    """
    array = load_structure(join(data_dir, "1l2y.mmtf"))[0]
    _, res_names = struc.get_residues(array)
    masses = [strucinfo.mass(res_name) for res_name in res_names]
    # C-terminus normally has additional oxygen atoms
    masses[-1] += strucinfo.mass("O")
    ref_masses = [strucinfo.mass(res) for res in struc.residue_iter(array)]
    #assert masses == pytest.approx(ref_masses)
    # Up to three additional/missing hydrogens are allowed
    # (protonation state)
    mass_diff = np.abs(np.array(
        [mass - ref_mass for mass, ref_mass in zip(masses, ref_masses)]
    ))
    assert (mass_diff // strucinfo.mass("H") <= 3).all()
    assert np.allclose((mass_diff % strucinfo.mass("H")), 0, atol=5e-3)
