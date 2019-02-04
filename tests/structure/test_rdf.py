from biotite.structure.io import load_structure
from biotite.structure.rdf import rdf
from biotite.structure.box import vectors_from_unitcell
from .util import data_dir
from os.path import join
import pytest
import numpy as np
import itertools


def test_rdf():
    """ General test to reproduce oxygen RDF for a box of water"""
    test_file = join(data_dir, "waterbox.pdb")
    stack = load_structure(test_file)

    # calculate oxygen RDF for water
    oxygen = stack[:, stack.atom_name == 'OW']
    interval = np.array([0, 10])
    n_bins = 100
    bins, g_r = rdf(oxygen[:, 0].coord, oxygen[:, 1:], interval=interval,
                    bins=n_bins, periodic=False)

    # Compare with MDTraj
    import mdtraj
    traj = mdtraj.load(test_file)
    ow = [a.index for a in traj.topology.atoms if a.name == 'O']
    pairs = itertools.product([ow[0]], ow[1:])
    mdt_bins, mdt_g_r = mdtraj.compute_rdf(traj, list(pairs),
                                           r_range=interval/10, n_bins=n_bins,
                                           periodic=False)

    assert np.allclose(bins, mdt_bins*10)
    assert np.allclose(g_r, mdt_g_r, rtol=0.02)

def test_rdf_bins():
    """ Test if RDF produce correct bin ranges """
    stack = load_structure(join(data_dir, "waterbox.pdb"))
    center = stack[:, 0]
    num_bins = 44
    bin_range = (0, 11.7)
    bins, g_r = rdf(center, stack, bins=num_bins, interval=bin_range)
    assert(len(bins) == num_bins)
    assert(bins[0] > bin_range[0])
    assert(bins[1] < bin_range[1])


# @pytest.mark.skip(reason="not implemented yet")
def test_rdf_with_selection():
    """ Test if the selection argument of rdf function works correctly """
    test_file = join(data_dir, "waterbox.pdb")
    stack = load_structure(test_file)

    # calculate oxygen RDF for water with and without a selection
    oxygen = stack[:, stack.atom_name == 'OW']
    interval = np.array([0, 10])
    n_bins = 100
    sele = (stack.atom_name=='OW') & (stack.res_id >= 3)
    bins, g_r = rdf(oxygen[:, 0].coord, stack, selection=sele,
                    interval=interval, bins=n_bins, periodic=False)

    nosel_bins, nosel_g_r = rdf(oxygen[:, 0].coord, oxygen[:, 1:],
                                interval=interval, bins=n_bins, periodic=False)

    assert np.allclose(bins, nosel_bins)
    assert np.allclose(g_r, nosel_g_r)


def test_rdf_atom_argument():
    """ Test if the first argument allows to use AtomArrayStack """
    test_file = join(data_dir, "waterbox.pdb")
    stack = load_structure(test_file)

    # calculate oxygen RDF for water with and without a selection
    oxygen = stack[:, stack.atom_name == 'OW']
    interval = np.array([0, 10])
    n_bins = 100
    sele = (stack.atom_name == 'OW') & (stack.res_id >= 3)
    bins, g_r = rdf(oxygen[:, 0].coord, stack, selection=sele,
                    interval=interval, bins=n_bins, periodic=False)

    nosel_bins, nosel_g_r = rdf(oxygen[:, 0].coord, oxygen[:, 1:],
                                interval=interval, bins=n_bins, periodic=False)

    assert np.allclose(bins, nosel_bins)
    assert np.allclose(g_r, nosel_g_r)


@pytest.mark.skip(reason="not implemented yet")
def test_rdf_multiple_center():
    """ Test if the first argument allows to use multiple centers"""
    test_file = join(data_dir, "waterbox.pdb")
    stack = load_structure(test_file)

    # calculate oxygen RDF for water with and without a selection
    oxygen = stack[:, stack.atom_name == 'OW']
    interval = np.array([0, 10])
    n_bins = 100

    # assert no error is raised
    bins, g_r = rdf(oxygen[:, 0:2].coord, oxygen, interval=interval,
                    bins=n_bins, periodic=False)


@pytest.mark.skip(reason="not implemented yet")
def test_rdf_periodic():
    """ Test if the periodic argument gives the correct results"""
    # TODO implement
    assert(False)


def test_rdf_box():
    """ Test correct use of simulation boxes """
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

