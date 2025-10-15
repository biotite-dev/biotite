# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import json
from os.path import join
import numpy as np
import numpy.random as random
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


def test_distance():
    coord1 = struc.coord([0, 1, 1])
    coord2 = struc.coord([0, 2, 2])
    assert struc.distance(coord1, coord2) == pytest.approx(np.sqrt(2))


def test_centroid():
    coord = struc.coord([[1, 1, 1], [0, -1, -1], [-1, 0, 0]])
    assert struc.centroid(coord).tolist() == [0, 0, 0]


def test_angle():
    coord1 = struc.coord([0, 0, 1])
    coord2 = struc.coord([0, 0, 0])
    coord3 = struc.coord([0, 1, 1])
    assert struc.angle(coord1, coord2, coord3) == pytest.approx(0.25 * np.pi)


def test_dihedral():
    coord1 = struc.coord([-0.5, -1, 0])
    coord2 = struc.coord([0, 0, 0])
    coord3 = struc.coord([1, 0, 0])
    coord4 = struc.coord([0, 0, -1])
    assert struc.dihedral(coord1, coord2, coord3, coord4) == pytest.approx(0.5 * np.pi)


@pytest.mark.parametrize("multi_model", [False, True])
def test_dihedral_backbone_general(multi_model):
    stack = strucio.load_structure(join(data_dir("structure"), "1l2y.bcif"))
    n_models = stack.stack_depth()
    n_res = stack.res_id[-1]

    if multi_model:
        atoms = stack
        expected_shape = (n_models, n_res)
    else:
        atoms = stack[0]
        expected_shape = (n_res,)
    # Test array
    phi, psi, omega = struc.dihedral_backbone(atoms)
    assert phi.shape == expected_shape
    assert psi.shape == expected_shape
    assert omega.shape == expected_shape
    _assert_plausible_omega(omega)


def _assert_plausible_omega(omega):
    # Remove nan values
    omega = omega.flatten()
    omega = np.abs(omega[~np.isnan(omega)])
    # Omega should be always approximately pi
    assert omega.tolist() == pytest.approx([np.pi] * len(omega), rel=0.6)


@pytest.mark.parametrize("multi_model", [False, True])
def test_dihedral_backbone_consistency(multi_model):
    """
    Check if the computed dihedral angles are equal to the reference computed with
    MDTraj.
    """
    with open(join(data_dir("structure"), "misc", "dihedrals.json")) as file:
        ref_dihedrals = json.load(file)
        ref_phi = np.array(ref_dihedrals["phi"])
        ref_psi = np.array(ref_dihedrals["psi"])
        ref_omega = np.array(ref_dihedrals["omega"])

    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1l2y.bcif"))
    atoms = pdbx.get_structure(pdbx_file, model=1)
    if multi_model:
        # Use models with equal coordinates
        atoms = struc.stack([atoms] * 2)

    test_phi, test_psi, test_ome = struc.dihedral_backbone(atoms)

    if multi_model:
        assert np.array_equal(test_phi[1], test_phi[0], equal_nan=True)
        assert np.array_equal(test_psi[1], test_psi[0], equal_nan=True)
        assert np.array_equal(test_ome[1], test_ome[0], equal_nan=True)
        test_phi = test_phi[0]
        test_psi = test_psi[0]
        test_ome = test_ome[0]
    assert test_phi == pytest.approx(ref_phi, abs=1e-3, nan_ok=True)
    assert test_psi == pytest.approx(ref_psi, abs=1e-3, nan_ok=True)
    assert test_ome == pytest.approx(ref_omega, abs=1e-3, nan_ok=True)


@pytest.mark.parametrize("multi_model", [False, True])
def test_dihedral_side_chain_consistency(multi_model):
    """
    Check if the computed dihedral angles are equal to the reference computed with
    MDTraj.
    """
    with open(join(data_dir("structure"), "misc", "dihedrals.json")) as file:
        ref_dihedrals = json.load(file)
        ref_chi = np.stack(
            [
                np.array(ref_dihedrals[name])
                for name in ("chi1", "chi2", "chi3", "chi4")
            ],
            axis=-1,
        )

    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1l2y.bcif"))
    atoms = pdbx.get_structure(pdbx_file, model=1)
    if multi_model:
        # Use models with equal coordinates
        atoms = struc.stack([atoms] * 2)

    test_chi = struc.dihedral_side_chain(atoms)

    if multi_model:
        assert np.array_equal(test_chi[1], test_chi[0], equal_nan=True)
        test_chi = test_chi[0]
    assert test_chi == pytest.approx(ref_chi, abs=1e-3, nan_ok=True)


def test_index_distance_non_periodic():
    """
    Without PBC the result should be equal to the normal distance
    calculation.
    """
    array = strucio.load_structure(join(data_dir("structure"), "3o5r.bcif"))
    ref_dist = struc.distance(
        array.coord[np.newaxis, :, :], array.coord[:, np.newaxis, :]
    ).flatten()
    length = array.array_length()
    dist = struc.index_distance(
        array,
        indices=np.stack(
            [np.repeat(np.arange(length), length), np.tile(np.arange(length), length)],
            axis=1,
        ),
    )
    assert np.allclose(dist, ref_dist)


@pytest.mark.parametrize(
    "shift, angles",
    itertools.product(
        [np.array([10, 20, 30]), np.array([-8, 12, 28]), np.array([0, 99, 54])],
        [
            np.array([90, 90, 90]),
            np.array([50, 90, 90]),
            np.array([90, 90, 120]),
            np.array([60, 60, 60]),
        ],
    ),
)
def test_index_distance_periodic(shift, angles):
    """
    The PBC aware computation, should give the same results,
    irrespective of which atoms are centered in the box
    """
    array = strucio.load_structure(join(data_dir("structure"), "3o5r.bcif"))
    # Use a box based on the boundaries of the structure
    # '+1' to add a margin
    boundaries = np.max(array.coord, axis=0) - np.min(array.coord, axis=0) + 1
    angles = np.deg2rad(angles)
    array.box = struc.vectors_from_unitcell(
        boundaries[0], boundaries[1], boundaries[2], angles[0], angles[1], angles[2]
    )

    length = array.array_length()
    # All-to-all indices
    dist_indices = np.stack(
        [np.repeat(np.arange(length), length), np.tile(np.arange(length), length)],
        axis=1,
    )
    # index_distance() creates a large ndarray
    try:
        ref_dist = struc.index_distance(array, dist_indices, periodic=True)
    except MemoryError:
        pytest.skip("Not enough memory")

    # Compare with shifted variant
    array.coord += shift
    array.coord = struc.move_inside_box(array.coord, array.box)
    # index_distance() creates a large ndarray
    try:
        test_dist = struc.index_distance(array, dist_indices, periodic=True)
    except MemoryError:
        pytest.skip("Not enough memory")
    assert np.allclose(test_dist, ref_dist, atol=2e-5)


def test_index_functions():
    """
    The `index_xxx()` functions should give the same result as the
    corresponding `xxx` functions.
    """
    stack = strucio.load_structure(join(data_dir("structure"), "1l2y.bcif"))
    array = stack[0]
    # Test for atom array, stack and raw coordinates
    samples = (array, stack, struc.coord(array), struc.coord(stack))
    # Generate random indices
    random.seed(42)
    indices = random.randint(array.array_length(), size=(100, 4), dtype=int)
    for sample in samples:
        if isinstance(sample, np.ndarray):
            atoms1 = sample[..., indices[:, 0], :]
            atoms2 = sample[..., indices[:, 1], :]
            atoms3 = sample[..., indices[:, 2], :]
            atoms4 = sample[..., indices[:, 3], :]
        else:
            atoms1 = sample[..., indices[:, 0]]
            atoms2 = sample[..., indices[:, 1]]
            atoms3 = sample[..., indices[:, 2]]
            atoms4 = sample[..., indices[:, 3]]
        assert np.allclose(
            struc.displacement(atoms1, atoms2),
            struc.index_displacement(sample, indices[:, :2]),
            atol=1e-5,
        )
        assert np.allclose(
            struc.distance(atoms1, atoms2),
            struc.index_distance(sample, indices[:, :2]),
            atol=1e-5,
        )
        assert np.allclose(
            struc.angle(atoms1, atoms2, atoms3),
            struc.index_angle(sample, indices[:, :3]),
            atol=1e-5,
        )
        assert np.allclose(
            struc.dihedral(atoms1, atoms2, atoms3, atoms4),
            struc.index_dihedral(sample, indices[:, :4]),
            atol=1e-5,
        )
