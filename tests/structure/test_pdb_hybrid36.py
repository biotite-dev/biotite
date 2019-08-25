# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.
import itertools
import glob
from os.path import join
import pytest
import numpy as np
import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from .util import data_dir


@pytest.mark.parametrize(
    "path, single_model",
    itertools.product(
        glob.glob(join(data_dir, "*.pdb")),
        [False, True]
    )
)
def test_array_conversion(path, single_model):
    model = 1 if single_model else None
    pdb_file = pdb.PDBFile()
    pdb_file.read(path)
    # Test also the thin wrapper around the methods
    # 'get_structure()' and 'set_structure()'
    array1 = pdb.get_structure(pdb_file, model=model)
    pdb_file = pdb.PDBFile()
    pdb.set_structure(pdb_file, array1, hybrid36=True)
    array2 = pdb.get_structure(pdb_file, model=model)
    if array1.box is not None:
        assert np.allclose(array1.box, array2.box)
    assert array1.bonds == array2.bonds
    for category in array1.get_annotation_categories():
        assert array1.get_annotation(category).tolist() == \
               array2.get_annotation(category).tolist()
    assert array1.coord.tolist() == array2.coord.tolist()


def test_id_overflow():
    # Create an atom array >= 100k atoms
    length = 100005
    a = struc.AtomArray(length)
    a.coord = np.zeros(a.coord.shape)
    a.chain_id = np.full(length, "A")
    # Create residue IDs over 10000
    a.res_id = np.arange(1, length+1)
    a.res_name = np.full(length, "GLY")
    a.hetero = np.full(length, False)
    a.atom_name = np.full(length, "CA")
    a.element = np.full(length, "C")

    # Write stack as hybrid-36 pdb file: no warning should be thrown
    with pytest.warns(None) as record:
        tmp_file_name = biotite.temp_file(".pdb")
        tmp_pdb_file = pdb.PDBFile()
        tmp_pdb_file.set_structure(a, hybrid36=True)
        tmp_pdb_file.write(tmp_file_name)
    assert len(record) == 0

    # Manually check if the output is written as correct hybrid-36
    with open(tmp_file_name) as output:
        last_line = output.readlines()[-1]
        atom_id = last_line.split()[1]
        assert(atom_id == "A0005")
        res_id = last_line.split()[4][1:]
        assert(res_id == "BXG5")
