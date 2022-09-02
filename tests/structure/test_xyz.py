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
from biotite.structure import Atom, AtomArray, AtomArrayStack
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
    
    # draw this many random atoms for structure
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

        assert mol_names == name
        assert atom_numbers == N_sample


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir("structure"), "molecules", "*.xyz"))
#    [x for x in glob.glob(join(data_dir("structure"), "molecules", "*.xyz")) if "CO" not in x],
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
    xyz.set_structure(xyz_file, ref_atoms)
    temp = TemporaryFile("w+")
    xyz_file.write(temp)
    

    temp.seek(0)
    xyz_file = xyz.XYZFile.read(temp) 
    test_atoms = xyz.get_structure(xyz_file)
    temp.close()
    

    instance_cond = isinstance(ref_atoms, AtomArray)
    instance_cond = instance_cond | isinstance(ref_atoms, AtomArrayStack)
    assert instance_cond     
    assert test_atoms == ref_atoms

            

