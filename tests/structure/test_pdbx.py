# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite
import itertools
import numpy as np
import glob
from os.path import join
from .util import data_dir
import pytest


@pytest.mark.parametrize("category, key, exp_value", [
    ("audit_author", "name",
        ["Neidigh, J.W.", "Fesinmeyer, R.M.", "Andersen, N.H."]),
    ("struct_ref_seq", "pdbx_PDB_id_code", "1L2Y"),
    ("pdbx_nmr_ensemble", "conformer_selection_criteria",
        ("structures with acceptable covalent geometry, "
         "structures with the least restraint violations"))
])
def test_parsing(category, key, exp_value):
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(join(data_dir, "1l2y.cif"))
    cat_dict = pdbx_file[category]
    value = cat_dict[key]
    if isinstance(value, np.ndarray):
        assert value.tolist() == exp_value
    else:
        assert value == exp_value

@pytest.mark.parametrize("path, is_stack", itertools.product(
                            glob.glob(join(data_dir, "*.cif")),
                            [False, True],
                         ))
def test_conversion(path, is_stack):
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(path)
    if is_stack:
        array1 = pdbx.get_structure(pdbx_file)
    else:
        array1 = pdbx.get_structure(pdbx_file, model=1)
    pdbx_file = pdbx.PDBxFile()
    pdbx.set_structure(pdbx_file, array1, data_block="test")
    print(str(pdbx_file)[:10000])
    if is_stack:
        array2 = pdbx.get_structure(pdbx_file)
    else:
        array2 = pdbx.get_structure(pdbx_file, model=1)
    for category in array1.get_annotation_categories():
        assert array1.get_annotation(category).tolist() == \
               array2.get_annotation(category).tolist()
    assert array1.coord.tolist() == array2.coord.tolist()

def test_extra_fields():
    path = join(data_dir, "1l2y.cif")
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(path)
    stack1 = pdbx.get_structure(pdbx_file, extra_fields=["atom_id","b_factor",
                                "occupancy","charge"])
    pdbx_file = pdbx.PDBxFile()
    pdbx.set_structure(pdbx_file, stack1, data_block="test")
    stack2 = pdbx.get_structure(pdbx_file, extra_fields=["atom_id","b_factor",
                                "occupancy","charge"])
    assert stack1 == stack2

            
        
    