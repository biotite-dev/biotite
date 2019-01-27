from biotite.structure.io import load_structure
from biotite.structure.rdf import rdf
from biotite.structure.box import vectors_from_unitcell
from .util import data_dir
from os.path import join
import pytest

def test_rdf_atoms():
    assert(False)

def test_rdf_atomstack():
    assert(False)

def test_rdf_COM():
    assert(False)

def test_rdf_selection():
    assert(False)

def test_rdf_periodic():
    assert(False)

def test_rdf_box():
    stack = load_structure(join(data_dir, "1l2y.pdb"))
    box = vectors_from_unitcell(1, 1, 1, 90, 90, 90)

    # Use box attribute of stack
    rdf(stack[0, 0], stack)

    # Use box attribute and dont fail because stack has no box
    stack.box = None
    rdf(stack[0, 0], stack, box=box)

    # Fail if no box is present
    with pytest.raises(ValueError):
        rdf(stack[0, 0], stack)
