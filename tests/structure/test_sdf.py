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
import biotite.structure.info as info
import biotite.structure.io.sdf as sdf
from ..util import data_dir


#def test_header_conversion():
#    """
#    Write known example data to the header of a SD file and expect
#    to retrieve the same information when reading the file again.
#    """
#    ref_header = (
#        "TestSD", "JD", "Biotite",
#        datetime.datetime.now().replace(second=0, microsecond=0),
#        "3D", "Lorem", "Ipsum", "123", "Lorem ipsum dolor sit amet"
#    )

#    sdf_file = sdf.SDFile()
#    sdf_file.set_header(*ref_header)
#    print(sdf_file)
#    temp = TemporaryFile("w+")
#    sdf_file.write(temp)

#    temp.seek(0)
#    sdf_file = sdf.SDFile.read(temp)
#    test_header = sdf_file.get_header()
#    temp.close()

#    assert test_header == ref_header


@pytest.mark.parametrize(
    "path, omit_charge",
    itertools.product(    
        glob.glob(
            join(data_dir("structure"), "molecules", "*.mol")
        )
        +
        glob.glob(
            join(data_dir("structure"), "molecules", "*.sdf")
        ),
        [False, True]
    )    
#    itertools.product(    
#        glob.glob(join(data_dir("structure"), "molecules", "*.sdf")),
#        [False, True]
#    )
)
def test_structure_conversion(path, omit_charge):
    """
    After reading a SD file, writing the structure back to a new file
    and reading it again should give the same structure.

    In this case an SDF file is used, but it is compatible with the
    SD format.
    """
    sdf_file = sdf.SDFile.read(path)
#    ref_header = sdf.get_header(sdf_file)
    ref_header_line = sdf_file.lines[1]
    ref_atoms = sdf.get_structure(sdf_file)
    ref_meta_information = sdf.get_metainformation(sdf_file)
    if omit_charge:
        ref_atoms.del_annotation("charge")

    sdf_file = sdf.SDFile()
#    sdf.set_header(sdf_file, *ref_header)    
    sdf.set_structure(sdf_file, ref_atoms)
    sdf.set_metainformation(sdf_file, ref_meta_information)
    temp = TemporaryFile("w+")
    

    lines_to_be_written = sdf_file.lines
    
    
    sdf_file.write(temp)

    temp.seek(0)
    sdf_file = sdf.SDFile.read(temp)
    test_atoms = sdf.get_structure(sdf_file)
    test_meta_information = sdf.get_metainformation(sdf_file)
    test_header = sdf.get_header(sdf_file)

    if omit_charge:
        assert np.all(test_atoms.charge == 0)
        test_atoms.del_annotation("charge") 
    temp.close()
    

#    cond_header   = test_header == ref_header
#    print("############TEST ["+str(path) + "]#################################")
#    print("ref_header      :: " + str(ref_header))
#    print("ref_header_line :: " +str(ref_header_line))
#    print("test_header     :: " + str(test_header))
#    assert cond_header        

    cond_atoms  = test_atoms == ref_atoms
    assert cond_atoms
        
    cond_meta   = test_meta_information == ref_meta_information
    assert cond_meta    



#@pytest.mark.parametrize(
#    "path", glob.glob(join(data_dir("structure"), "SDecules", "*.SD")),
#)
#def test_pdbx_consistency(path):
#    """
#    Check if the structure parsed from a SD file is equal to the same
#    structure read from the *Chemical Component Dictionary* in PDBx
#    format.

#    In this case an SDF file is used, but it is compatible with the
#    SD format.
#    """
#    sdf_name = split(splitext(path)[0])[1]
#    ref_atoms = info.residue(sdf_name)
#    # The CCD contains information about aromatic bond types,
#    # but the SDF test files do not
#    ref_atoms.bonds.remove_aromaticity()

#    sdf_file = sdf.SDFile.read(path)
#    test_atoms = sdf_file.get_structure()

#    assert test_atoms.coord.shape == ref_atoms.coord.shape
#    assert test_atoms.coord.flatten().tolist() \
#        == ref_atoms.coord.flatten().tolist()
#    assert test_atoms.element.tolist() == ref_atoms.element.tolist()
#    assert test_atoms.charge.tolist() == ref_atoms.charge.tolist()
#    assert set(tuple(bond) for bond in test_atoms.bonds.as_array()) \
#        == set(tuple(bond) for bond in  ref_atoms.bonds.as_array())
#        
#    #header = sdf_file.get_header()
#    
#    try:
#        header = sdf_file.get_header()                
#    except:
#        assert False, "Could not get_header for SDFile [" +str(path)
#        
                
