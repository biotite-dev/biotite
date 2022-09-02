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
from biotite.structure import Atom, AtomArray, AtomArrayStack, BondList
import biotite.structure.info as info
import biotite.structure.io.mol2 as mol2
from ..util import data_dir

def draw_random_struct(N_atoms, charge=False):

    arr= struc.array(
        [
            Atom(
                [
                    float(np.random.rand()) for _ in range(3)
                ],
                element="C",
                charge=int(np.random.randint(-5,5))
            ) for _ in range(N_atoms)
        ]
    )
    
    arr.bonds=BondList(N_atoms)
    
    for i in range(1,N_atoms):
        arr.bonds.add_bond(i-1, i, 1)

    
    return arr
    
    


def test_header_conversion():
    """
    Write known example data to the header of a Mol2 file and expect
    to retrieve the same information when reading the file again.
    """
    ref_names = (
        "Testxyz", "JD", "Biotite",
        str(datetime.datetime.now().replace(second=0, microsecond=0)),
        "3D", "Lorem", "Ipsum", "123", "Lorem ipsum dolor sit amet"
    )
    
    ref_nums = np.random.randint(1, 10, len(ref_names))
    ref_mol_type = [
        mol2.MOL2File.supported_mol_types[x] for x in np.random.randint(
                0,5,len(ref_names)
            ).astype(int)
    ]

    ref_charge_type = [
        mol2.MOL2File.supported_charge_types[x] for x in np.random.randint(
            0,12,len(ref_names)
        ).astype(int)
    ]              
    
    
    # draw this many random Atoms for structure
    # necessary as header should only exist for a non empty XYZFile
    N_sample = 10
    
    for i, name in enumerate(ref_names):
        mol2_file = mol2.MOL2File()        
        mol2_file.set_header(
            name, 
            ref_nums[i], 
            ref_mol_type[i], 
            ref_charge_type[i]
       
        )               
        # need to set some structure for file writing
        rand_struct = draw_random_struct(
            ref_nums[i],
            charge = (ref_charge_type[i] != "NO_CHARGES"),
        )
        mol2_file.set_structure(rand_struct)  

                
        temp = TemporaryFile("w+")
        mol2_file.write(temp)

        temp.seek(0)
        mol2_file = mol2.MOL2File.read(temp)
        test_header = mol2_file.get_header()        
        temp.close()

        assert len(test_header) > 0
        assert test_header[0] == name        
        assert test_header[1] == ref_nums[i]
        assert test_header[2] == ref_mol_type[i]
        assert test_header[3] == ref_charge_type[i]


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir("structure"), "molecules", "*.mol2"))
)
def test_structure_conversion(path):
    """
    After reading a mol2 file, writing the structure back to a new file
    and reading it again should give the same structure.
    """
    mol2_file = mol2.MOL2File.read(path)
        
    ref_header = mol2_file.get_header()
    ref_atoms = mol2.get_structure(mol2_file)
    

    mol2_file = mol2.MOL2File()
    mol2_file.set_header(
        ref_header[0], ref_header[1], ref_header[2], ref_header[3]
    )
    mol2.set_structure(mol2_file, ref_atoms)
    temp = TemporaryFile("w+")
    mol2_file.write(temp)
    

    temp.seek(0)
    mol2_file = mol2.MOL2File.read(temp) 
    test_atoms = mol2.get_structure(mol2_file)
    temp.close()
    

    instance_cond = isinstance(ref_atoms, AtomArray)
    instance_cond = instance_cond | isinstance(ref_atoms, AtomArrayStack)
    assert instance_cond
    assert test_atoms == ref_atoms
    assert test_atoms.bonds == ref_atoms.bonds

            
@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir("structure"), "molecules", "*.mol2"))
)
def test_charge_rounding(path):
    """
    After reading a mol2 file, writing the structure back to a new file
    and reading it again should give us a file where the partial_charges are
    identical to the rounded charges as without changing them manually this
    is what will be written to the charge column. Also setting the charges 
    specifically before setting the structure will lead to the partial_charges
    being used in the set_structure function.
    """
    mol2_file = mol2.MOL2File.read(path)
        
    ref_header = mol2_file.get_header()
    ref_atoms = mol2.get_structure(mol2_file)
    
    if ref_header[3] == "NO_CHARGES":        
        assert mol2_file.get_charges() is None        
    else:
    
        ref_partial_charges = mol2.get_charges(mol2_file)
        ref_charges = ref_atoms.charge
        

        mol2_file = mol2.MOL2File()
        mol2_file.set_header(
            ref_header[0], ref_header[1], ref_header[2], ref_header[3]
        )
        mol2.set_structure(mol2_file, ref_atoms)       
        temp = TemporaryFile("w+")
        mol2_file.write(temp)
        

        temp.seek(0)
        mol2_file = mol2.MOL2File.read(temp) 
        test_atoms = mol2.get_structure(mol2_file)
        test_charges = mol2.get_charges(mol2_file)
        temp.close()
        
        mol2_file = mol2.MOL2File()
        mol2_file.set_header(
            ref_header[0], ref_header[1], ref_header[2], ref_header[3]
        )
        mol2.set_charges(mol2_file, ref_partial_charges)
        mol2.set_structure(mol2_file, ref_atoms)           
        temp = TemporaryFile("w+")    
        mol2_file.write(temp)    
        
        temp.seek(0)
        mol2_file = mol2.MOL2File.read(temp) 
        test2_atoms = mol2.get_structure(mol2_file)
        test2_charges = mol2.get_charges(mol2_file)
        temp.close()    
        

        instance_cond = isinstance(ref_atoms, AtomArray)
        instance_cond = instance_cond | isinstance(ref_atoms, AtomArrayStack)
        assert instance_cond

        assert test_atoms == ref_atoms
        assert test2_atoms == ref_atoms
                
        assert test_atoms.bonds == ref_atoms.bonds
        assert test2_atoms.bonds == ref_atoms.bonds        
        
        assert np.all(np.rint(ref_partial_charges) == test_charges)        
        assert np.all(ref_partial_charges == test2_charges)
  
