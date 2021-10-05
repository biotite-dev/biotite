# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
All functionalities are tested against equivalent calls in
:class:`biotite.structure.io.pdb.PDBFile`.
"""

from tempfile import TemporaryFile
import itertools
import glob
from io import StringIO
from os.path import join, dirname, realpath
import pytest
import biotite
import numpy as np
import biotite.structure.io.pdb as pdb
import fastpdb as fastpdb


TEST_STRUCTURES = glob.glob(join(dirname(realpath(__file__)), "data", "*.pdb"))


@pytest.mark.parametrize(
    "path", TEST_STRUCTURES
)
def test_get_model_count(path):
    ref_file = pdb.PDBFile.read(path)
    
    test_file = fastpdb.PDBFile.read(path)
    

    assert ref_file.get_model_count() == test_file.get_model_count()


@pytest.mark.parametrize(
    "path, model",
    itertools.product(
        TEST_STRUCTURES,
        [None, 1, -1],
    )
)
def test_get_coord(path, model):
    ref_file = pdb.PDBFile.read(path)
    try:
        ref_coord = ref_file.get_coord(model)
    except biotite.InvalidFileError:
        if model is None:
            # Cannot create an AtomArrayStack
            # due to different number of atoms per model
            return
        else:
            raise
    
    test_file = fastpdb.PDBFile.read(path)
    test_coord = test_file.get_coord(model)


    assert np.allclose(test_coord, ref_coord)


@pytest.mark.parametrize(
    "path, model, altloc, extra_fields, include_bonds",
    itertools.product(
        TEST_STRUCTURES,
        [None, 1, -1],
        ["occupancy", "first", "all"],
        [False, True],
        [False, True],
    )
)
def test_get_structure(path, model, altloc, extra_fields, include_bonds):
    if extra_fields:
        extra_fields = ["atom_id", "b_factor", "occupancy", "charge"]
    else:
        extra_fields = None
    
    
    ref_file = pdb.PDBFile.read(path)
    try:
        ref_atoms = ref_file.get_structure(
            model, altloc, extra_fields, include_bonds
        )
    except biotite.InvalidFileError:
        if model is None:
            # Cannot create an AtomArrayStack
            # due to different number of atoms per model
            return
        else:
            raise

    
    test_file = fastpdb.PDBFile.read(path)
    test_atoms = test_file.get_structure(
        model, altloc, extra_fields, include_bonds
    )

    
    if ref_atoms.box is not None:
        assert np.allclose(test_atoms.box, ref_atoms.box)
    else:
        assert test_atoms == None
    
    assert test_atoms.bonds == ref_atoms.bonds
    
    for category in ref_atoms.get_annotation_categories():
        if np.issubdtype(ref_atoms.get_annotation(category).dtype, float):
            assert test_atoms.get_annotation(category).tolist() \
                == pytest.approx(ref_atoms.get_annotation(category).tolist())
        else:
            assert test_atoms.get_annotation(category).tolist() \
                ==  ref_atoms.get_annotation(category).tolist()
    
    assert np.allclose(test_atoms.coord, ref_atoms.coord)


@pytest.mark.parametrize(
    "path, model, altloc, extra_fields, include_bonds",
    itertools.product(
        TEST_STRUCTURES,
        [None, 1, -1],
        ["occupancy", "first", "all"],
        [False, True],
        [False, True],
    )
)
def test_set_structure(path, model, altloc, extra_fields, include_bonds):
    if extra_fields:
        extra_fields = ["atom_id", "b_factor", "occupancy", "charge"]
    else:
        extra_fields = None
    
    
    input_file = pdb.PDBFile.read(path)
    try:
        atoms = input_file.get_structure(
            model, altloc, extra_fields, include_bonds
        )
    except biotite.InvalidFileError:
        if model is None:
            # Cannot create an AtomArrayStack
            # due to different number of atoms per model
            return
        else:
            raise

    
    ref_file = pdb.PDBFile()
    ref_file.set_structure(atoms)
    ref_file_content = StringIO()
    ref_file.write(ref_file_content)

    test_file = fastpdb.PDBFile()
    test_file.set_structure(atoms)
    test_file_content = StringIO()
    test_file.write(test_file_content)


    assert test_file_content.getvalue() == ref_file_content.getvalue()