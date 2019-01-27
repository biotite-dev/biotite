from biotite.structure.io import load_structure
from biotite.structure.rdf import rdf
from biotite.structure.box import vectors_from_unitcell
from biotite.structure import mass_center
from .util import data_dir
from os.path import join
import pytest
import numpy as np


def test_rdf():
    stack = load_structure(join(data_dir, "waterbox.pdb"))
    center = stack[:, 0]
    bins, g_r = rdf(center, stack)
    # TODO confirm output


def test_rdf_bins():
    stack = load_structure(join(data_dir, "waterbox.pdb"))
    center = stack[:, 0]
    num_bins = 44
    bin_range = (0, 11.7)
    bins, g_r = rdf(center, stack, bins=num_bins, range=bin_range)
    assert(len(bins) == num_bins)
    assert(bins[0] > bin_range[0])
    assert(bins[1] < bin_range[1])


@pytest.mark.skip(reason="not implemented yet")
def test_rdf_with_selection():
    stack = load_structure(join(data_dir, "waterbox.pdb"))
    center = stack[:, 0]
    selection = stack[:, stack.atom_name == 'OW']
    bins, g_r = rdf(center, stack, selection=selection)
    # TODO confirm output


@pytest.mark.skip(reason="not implemented yet")
def test_rdf_atoms():
    stack = load_structure(join(data_dir, "waterbox.pdb"))
    atom = stack[0, 0]
    rdf(atom, stack[0])


@pytest.mark.skip(reason="not implemented yet")
def test_rdf_COM():
    stack = load_structure(join(data_dir, "waterbox.pdb"))
    com = mass_center(stack)
    bins, g_r = rdf(com, stack)
    # TODO confirm output


@pytest.mark.skip(reason="not implemented yet")
def test_rdf_periodic():
    # TODO implement
    assert(False)


def test_rdf_box():
    stack = load_structure(join(data_dir, "waterbox.pdb"))
    box = vectors_from_unitcell(1, 1, 1, 90, 90, 90)
    box = np.repeat(box[np.newaxis, :, :], len(stack), axis=0)

    # Use box attribute of stack
    rdf(stack[:, 0], stack)

    # Use box attribute and dont fail because stack has no box
    stack.box = None
    rdf(stack[:, 0], stack, box=box)

    # Fail if no box is present
    with pytest.raises(ValueError):
        rdf(stack[:, 0], stack)

    # Fail if box is of wrong size
    with pytest.raises(ValueError):
        rdf(stack[:, 0], stack, box=box[0])

