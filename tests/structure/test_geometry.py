# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
from os.path import join
import numpy as np
import numpy.random as random
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
from .util import data_dir


def test_distance():
    coord1 = struc.coord([0,1,1])
    coord2 = struc.coord([0,2,2])
    assert struc.distance(coord1, coord2) == pytest.approx(np.sqrt(2))


def test_centroid():
    coord = struc.coord([[1,1,1],[0,-1,-1],[-1,0,0]])
    assert struc.centroid(coord).tolist() == [0,0,0]


def test_angle():
    coord1 = struc.coord([0,0,1])
    coord2 = struc.coord([0,0,0])
    coord3 = struc.coord([0,1,1])
    assert struc.angle(coord1, coord2, coord3) == pytest.approx(0.25*np.pi)


def test_dihedral():
    coord1 = struc.coord([-0.5,-1,0])
    coord2 = struc.coord([0,0,0])
    coord3 = struc.coord([1,0,0])
    coord4 = struc.coord([0,0,-1])
    assert struc.dihedral(coord1, coord2, coord3, coord4) \
           == pytest.approx(0.5*np.pi)


def test_dihedral_backbone():
    stack = strucio.load_structure(join(data_dir, "1l2y.mmtf"))
    array = stack[0]
    # Test array
    phi, psi, omega = struc.dihedral_backbone(array)
    assert omega.shape == (20,)
    # Remove nan values
    omega = np.abs(omega)[:-1]
    assert omega.tolist() == pytest.approx([np.pi] * len(omega), rel=0.05)
    # Test stack
    phi, psi, omega = struc.dihedral_backbone(stack)
    assert omega.shape == (38,20)
    # Remove nan values
    omega = np.abs(omega)[:, :-1]
    omega = np.average(omega, axis=0)
    assert omega.tolist() == pytest.approx([np.pi] * len(omega), rel=0.05)


def test_index_distance_non_periodic():
    """
    Without PBC the result should be equal to the normal distance
    calculation.
    """
    array = strucio.load_structure(join(data_dir, "3o5r.mmtf"))
    ref_dist = struc.distance(
        array.coord[np.newaxis, :, :],
        array.coord[:, np.newaxis, :]
    ).flatten()
    length = array.array_length()
    dist = struc.index_distance(
        array,
        indices = np.stack([
            np.repeat(np.arange(length), length),
              np.tile(np.arange(length), length)
        ], axis=1)
    )
    assert np.allclose(dist, ref_dist)


@pytest.mark.parametrize(
    "shift", [
        np.array([10, 20, 30]),
        np.array([-8, 12, 28]),
        np.array([ 0, 99, 54])
    ]
)
def test_index_distance_periodic_orthogonal(shift):
    """
    The PBC aware computation, should give the same results,
    irrespective of which atoms are centered in the box 
    """
    array = strucio.load_structure(join(data_dir, "3o5r.mmtf"))
    # Use a box based on the boundaries of the structure
    # '+1' to add a margin
    array.box = np.diag(
        np.max(array.coord, axis=0) - np.min(array.coord, axis=0) + 1
    )

    length = array.array_length()
    dist_indices = np.stack([
        np.repeat(np.arange(length), length),
            np.tile(np.arange(length), length)
    ], axis=1)
    ref_dist = struc.index_distance(array, dist_indices, periodic=True)
    
    array.coord += shift
    array.coord = struc.move_inside_box(array.coord, array.box)
    dist = struc.index_distance(array, dist_indices, periodic=True)
    assert np.allclose(dist, ref_dist, atol=1e-5)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.xfail(raises=ImportError)
@pytest.mark.parametrize(
    "shift, angles", itertools.product(
    [
        np.array([10, 20, 30]),
        np.array([-8, 12, 28]),
        np.array([ 0, 99, 54])
    ],
    [
        np.array([ 50,  90,  90]),
        np.array([ 90,  90, 120]),
        np.array([ 60,  60,  60])
    ]
    )
)
def test_index_distance_periodic_triclinic(shift, angles):
    """
    The PBC aware computation, should give the same results,
    irrespective of which atoms are centered in the box 
    """
    array = strucio.load_structure(join(data_dir, "3o5r.mmtf"))
    # Use a box based on the boundaries of the structure
    # '+1' to add a margin
    boundaries = np.max(array.coord, axis=0) - np.min(array.coord, axis=0) + 1
    angles = np.deg2rad(angles)
    array.box = struc.vectors_from_unitcell(
        boundaries[0], boundaries[1], boundaries[2],
        angles[0], angles[1], angles[2]
    )

    length = array.array_length()
    dist_indices = np.stack([
        np.repeat(np.arange(length), length),
          np.tile(np.arange(length), length)
    ], axis=1)
    ref_dist = struc.index_distance(array, dist_indices, periodic=True)

    # Compare with MDTraj
    import mdtraj
    traj = mdtraj.load(join(data_dir, "3o5r.pdb"))
    # Angstrom to Nanometers
    traj.unitcell_vectors = array.box[np.newaxis, :, :] / 10
    # Nanometers to Angstrom
    mdtraj_dist = mdtraj.compute_distances(traj, dist_indices)[0] * 10
    ind = np.where(~np.isclose(ref_dist, mdtraj_dist, atol=1e-5, rtol=1e-3))[0]
    assert np.allclose(ref_dist, mdtraj_dist, atol=1e-5, rtol=1e-3)
    
    # Compare with shifted variant
    array.coord += shift
    array.coord = struc.move_inside_box(array.coord, array.box)
    dist = struc.index_distance(array, dist_indices, periodic=True)
    assert np.allclose(dist, ref_dist, atol=1e-5)


def test_index_functions():
    """
    The `index_xxx()` functions should give the same result as the
    corresponding `xxx` functions.
    """
    stack = strucio.load_structure(join(data_dir, "1l2y.mmtf"))
    array = stack[0]
    # Test for atom array, stack and raw coordinates
    samples = (array, stack, struc.coord(array), struc.coord(stack))
    # Generate random indices
    random.seed(42)
    indices = random.randint(array.array_length(), size=(100,4), dtype=int)
    for sample in samples:
        if isinstance(sample, np.ndarray):
            atoms1 = sample[..., indices[:,0], :]
            atoms2 = sample[..., indices[:,1], :]
            atoms3 = sample[..., indices[:,2], :]
            atoms4 = sample[..., indices[:,3], :]
        else:
            atoms1 = sample[..., indices[:,0]]
            atoms2 = sample[..., indices[:,1]]
            atoms3 = sample[..., indices[:,2]]
            atoms4 = sample[..., indices[:,3]]
        assert np.allclose(
            struc.displacement(atoms1, atoms2),
            struc.index_displacement(sample, indices[:,:2]),
            atol=1e-5
        )
        assert np.allclose(
            struc.distance(atoms1, atoms2),
            struc.index_distance(sample, indices[:,:2]),
            atol=1e-5
        )
        assert np.allclose(
            struc.angle(atoms1, atoms2, atoms3),
            struc.index_angle(sample, indices[:,:3]),
            atol=1e-5
        )
        assert np.allclose(
            struc.dihedral(atoms1, atoms2, atoms3, atoms4),
            struc.index_dihedral(sample, indices[:,:4]),
            atol=1e-5
        )



