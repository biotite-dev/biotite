# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import datetime
from tempfile import TemporaryFile
import glob
from os.path import join, split, splitext
import pytest
import numpy as np
import biotite.structure as struc
from biotite.structure import Atom
import biotite.structure.info as info
import biotite.structure.io.xyz as xyz
from ..util import data_dir


def draw_random_struct(N_atoms):

    return struc.array(
        [
            Atom(
                [
                    float(np.random.rand()) for _ in range(3)
                ],
                element="C"
            ) for _ in range(N_atoms)
        ]
    )


def test_header_conversion():
    """
    Write known example data to the header of a xyz file and expect
    to retrieve the same information when reading the file again.
    """
    ref_names = (
        "Testxyz", "JD", "Biotite",
        str(datetime.datetime.now().replace(second=0, microsecond=0)),
        "3D", "Lorem", "Ipsum", "123", "Lorem ipsum dolor sit amet"
    )
    
    # draw this many random Atoms for structure
    # necessary as header should only exist for a non empty XYZFile
    N_sample = 10
    
    for name in ref_names:
    
        xyz_file = xyz.XYZFile()
        xyz_file.set_structure(draw_random_struct(N_sample))
        xyz_file.set_header(name)        
        print(xyz_file)
        temp = TemporaryFile("w+")
        xyz_file.write(temp)

        temp.seek(0)
        xyz_file = xyz.XYZFile.read(temp)
        atom_numbers, mol_names = xyz_file.get_header()
        
        temp.close()

        assert mol_names[0] == name
        assert atom_numbers[0] == N_sample


@pytest.mark.parametrize(
    "path",
    [x for x in glob.glob(join(data_dir("structure"), "molecules", "*.xyz")) if "CO" not in x],
)
def test_structure_conversion(path):
    """
    After reading a xyz file, writing the structure back to a new file
    and reading it again should give the same structure.

    In this case an SDF file is used, but it is compatible with the
    xyz format.
    """
    xyz_file = xyz.XYZFile.read(path)
    
    
    
    ref_atoms = xyz.get_structure(xyz_file)
    

    xyz_file = xyz.XYZFile()
    xyz.set_structure(xyz_file, ref_atoms[0])
    temp = TemporaryFile("w+")
    xyz_file.write(temp)

    temp.seek(0)
    xyz_file = xyz.XYZFile.read(temp)
    test_atoms = xyz.get_structure(xyz_file)
    temp.close()
    
    print("test_atoms ::")
    print(test_atoms)    
    print("")
    print("")
    print("")
    
    print("ref_atoms ::")
    print(ref_atoms)
    print("")
    print("")
    print("")    

    assert test_atoms == ref_atoms


#@pytest.mark.parametrize(
#    "path", glob.glob(join(data_dir("structure"), "xyzecules", "*.sdf")),
#)
#def test_pdbx_consistency(path):
#    """
#    Check if the structure parsed from a xyz file is equal to the same
#    structure read from the *Chemical Component Dictionary* in PDBx
#    format.

#    In this case an SDF file is used, but it is compatible with the
#    xyz format.
#    """
#    xyz_name = split(splitext(path)[0])[1]
#    ref_atoms = info.residue(xyz_name)
#    # The CCD contains information about aromatic bond types,
#    # but the SDF test files do not
#    ref_atoms.bonds.remove_aromaticity()

#    xyz_file = xyz.XYZFile.read(path)
#    test_atoms = xyz_file.get_structure()

#    assert test_atoms.coord.shape == ref_atoms.coord.shape
#    assert test_atoms.coord.flatten().tolist() \
#        == ref_atoms.coord.flatten().tolist()
#    assert test_atoms.element.tolist() == ref_atoms.element.tolist()
#    assert test_atoms.charge.tolist() == ref_atoms.charge.tolist()
#    assert set(tuple(bond) for bond in test_atoms.bonds.as_array()) \
#        == set(tuple(bond) for bond in  ref_atoms.bonds.as_array())
