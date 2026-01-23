import json
from os.path import join
import numpy as np
import pytest
import biotite.structure.io.gro as gro
from biotite.structure.box import vectors_from_unitcell
from biotite.structure.io import load_structure
from biotite.structure.rdf import rdf
from tests.util import data_dir

TEST_FILE = join(data_dir("structure"), "waterbox.gro")


def test_rdf_consistency():
    """
    Check that oxygen RDF for a box of water reproduces results from MDTraj.
    """
    INTERVAL = [0, 10]
    N_BINS = 100

    # Load precomputed RDF from MDTraj
    with open(join(data_dir("structure"), "misc", "rdf.json")) as file:
        ref_data = json.load(file)
    ref_bins = ref_data["bins"]
    ref_g_r = ref_data["g_r"]

    gro_file = gro.GROFile.read(TEST_FILE)
    stack = gro_file.get_structure()
    # Calculate oxygen RDF for water
    oxygen = stack[:, stack.atom_name == "OW"]
    test_bins, test_g_r = rdf(
        oxygen[:, 0].coord, oxygen, interval=INTERVAL, bins=N_BINS, periodic=False
    )

    assert test_bins.tolist() == pytest.approx(ref_bins)
    assert test_g_r.tolist() == pytest.approx(ref_g_r, rel=0.01)


def test_rdf_bins():
    """
    Test if RDF produce correct bin ranges.
    """
    stack = load_structure(TEST_FILE)
    center = stack[:, 0]
    num_bins = 44
    bin_range = (0, 11.7)
    bins, g_r = rdf(center, stack, bins=num_bins, interval=bin_range)
    assert len(bins) == num_bins
    assert bins[0] > bin_range[0]
    assert bins[1] < bin_range[1]


def test_rdf_with_selection():
    """
    Test if the selection argument of rdf function works as expected.
    """
    stack = load_structure(TEST_FILE)

    # calculate oxygen RDF for water with and without a selection
    oxygen = stack[:, stack.atom_name == "OW"]
    interval = np.array([0, 10])
    n_bins = 100
    sele = (stack.atom_name == "OW") & (stack.res_id >= 3)
    bins, g_r = rdf(
        oxygen[:, 0].coord,
        stack,
        selection=sele,
        interval=interval,
        bins=n_bins,
        periodic=False,
    )

    nosel_bins, nosel_g_r = rdf(
        oxygen[:, 0].coord,
        oxygen[:, 1:],
        interval=interval,
        bins=n_bins,
        periodic=False,
    )

    assert np.allclose(bins, nosel_bins)
    assert np.allclose(g_r, nosel_g_r)


def test_rdf_atom_argument():
    """
    Test if the first argument allows using AtomArrayStack.
    """
    stack = load_structure(TEST_FILE)

    # calculate oxygen RDF for water with and without a selection
    oxygen = stack[:, stack.atom_name == "OW"]
    interval = np.array([0, 10])
    n_bins = 100

    bins, g_r = rdf(oxygen[:, 0], stack, interval=interval, bins=n_bins, periodic=False)

    atom_bins, atoms_g_r = rdf(
        oxygen[:, 0].coord, stack, interval=interval, bins=n_bins, periodic=False
    )

    assert np.allclose(g_r, atoms_g_r)


def test_rdf_multiple_center():
    """
    Test if the first argument allows using multiple centers.
    """
    stack = load_structure(TEST_FILE)

    # calculate oxygen RDF for water with and without a selection
    oxygen = stack[:, stack.atom_name == "OW"]
    interval = np.array([0, 10])
    n_bins = 100

    # averaging individual calculations
    bins1, g_r1 = rdf(
        oxygen[:, 1].coord,
        oxygen[:, 2:],
        interval=interval,
        bins=n_bins,
        periodic=False,
    )
    bins2, g_r2 = rdf(
        oxygen[:, 0].coord,
        oxygen[:, 2:],
        interval=interval,
        bins=n_bins,
        periodic=False,
    )
    mean = np.mean([g_r1, g_r2], axis=0)

    # this should give the same result as averaging for oxygen 0 and 1
    bins, g_r = rdf(
        oxygen[:, 0:2].coord,
        oxygen[:, 2:],
        interval=interval,
        bins=n_bins,
        periodic=False,
    )

    assert np.allclose(g_r, mean, rtol=0.0001)


def test_rdf_box():
    """
    Test correct use of simulation boxes.
    """
    stack = load_structure(TEST_FILE)
    box = vectors_from_unitcell(1, 1, 1, 90, 90, 90)
    box_stack = np.repeat(box[np.newaxis, :, :], len(stack), axis=0)

    # Use box attribute of stack
    rdf(stack[:, 0], stack)
    # Test proper stacking of single AtomArrays -> Use box attribute
    rdf(stack[0, 0], stack[0])

    # Use box attribute and don't fail because stack has no box
    stack.box = None
    rdf(stack[:, 0], stack, box=box_stack)

    # Fail if no box is present
    with pytest.raises(ValueError):
        rdf(stack[:, 0], stack)

    # Fail if box is of wrong size
    with pytest.raises(ValueError):
        rdf(stack[:, 0], stack, box=box)

    # Reshape (3,3) boxes to (1,3,3) to match stacked input AtomArrays
    rdf(stack[0, 0], stack[0], box=box)


def test_rdf_normalized():
    """
    Assert that the RDF tail is normalized to 1.
    """
    test_file = TEST_FILE
    stack = load_structure(test_file)

    # calculate oxygen RDF for water
    oxygen = stack[:, stack.atom_name == "OW"]
    interval = np.array([0, 5])
    n_bins = 100

    bins, g_r = rdf(oxygen.coord, oxygen, interval=interval, bins=n_bins, periodic=True)
    assert np.allclose(g_r[-10:], np.ones(10), atol=0.1)
