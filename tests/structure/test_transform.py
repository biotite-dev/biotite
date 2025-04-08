# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from biotite.structure.transform import _multi_matmul as multi_matmul
from tests.util import data_dir


@pytest.fixture(
    params=itertools.product(
        [1, 2, 3],  # ndim
        [False, True],  # as_coord
    )
)
def input_atoms(request):
    ndim, as_coord = request.param

    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1l2y.bcif"))
    atoms = pdbx.get_structure(pdbx_file)

    if ndim == 2:
        # Only one model
        atoms = atoms[0]
    elif ndim == 1:
        # Only one atom
        atoms = atoms[0, 0]

    if as_coord:
        return atoms.coord
    else:
        return atoms


def test_transform_as_matrix():
    """
    Check if the 4x4 matrix obtained from
    :meth:`AffineTransformation.as_matrix()` transforms coordinates in
    the same way as :meth:`AffineTransformation.apply()`.
    """
    N_MODELS = 10
    N_COORD = 100

    np.random.seed(0)
    orig_coord = np.random.rand(N_MODELS, N_COORD, 3)
    transform = struc.AffineTransformation(
        center_translation=np.random.rand(N_MODELS, 3),
        # This is not really a rotation matrix,
        # but the same maths apply
        rotation=np.random.rand(N_MODELS, 3, 3),
        target_translation=np.random.rand(N_MODELS, 3),
    )

    ref_coord = transform.apply(orig_coord)

    orig_coord_4 = np.concatenate(
        [orig_coord, np.ones((N_MODELS, N_COORD, 1))], axis=-1
    )
    test_coord_4 = multi_matmul(transform.as_matrix(), orig_coord_4)
    test_coord = test_coord_4[..., :3]

    assert test_coord.flatten().tolist() == pytest.approx(
        ref_coord.flatten().tolist(), abs=1e-6
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("random_seed", np.arange(5))
def test_translate(input_atoms, ndim, as_list, random_seed):
    """
    Translate and translate back and check if the coordinates are still
    the same.
    """
    if ndim > len(input_atoms.shape):
        # Cannot run tests if translation vector has more dimensions
        # as input coordinates/atoms
        return

    np.random.seed(random_seed)
    vectors = np.random.rand(*struc.coord(input_atoms).shape[-ndim:])
    vectors *= 10
    neg_vectors = -vectors
    if as_list:
        vectors = vectors.tolist()
        neg_vectors = neg_vectors.tolist()

    translated = struc.translate(input_atoms, vectors)
    restored = struc.translate(translated, neg_vectors)

    assert isinstance(restored, type(input_atoms))
    assert struc.coord(restored).shape == struc.coord(input_atoms).shape
    assert np.allclose(struc.coord(restored), struc.coord(input_atoms), atol=1e-5)


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("axis", [0, 1, 2])  # x, y, z
@pytest.mark.parametrize("random_seed", np.arange(5))
@pytest.mark.parametrize("centered", [False, True])
def test_rotate(input_atoms, as_list, axis, random_seed, centered):
    """
    Rotate and rotate back and check if the coordinates are still
    the same.
    """
    np.random.seed(random_seed)
    angles = np.zeros(3)
    angles[axis] = np.random.rand() * 2 * np.pi
    neg_angles = -angles
    if as_list:
        angles = angles.tolist()
        neg_angles = neg_angles.tolist()

    func = struc.rotate_centered if centered else struc.rotate
    rotated = func(input_atoms, angles)
    restored = func(rotated, neg_angles)

    assert isinstance(restored, type(input_atoms))
    assert struc.coord(restored).shape == struc.coord(input_atoms).shape
    print(np.max(np.abs(struc.coord(restored) - struc.coord(input_atoms))))
    assert np.allclose(struc.coord(restored), struc.coord(input_atoms), atol=1e-5)
    if centered and struc.coord(input_atoms).ndim > 1:
        assert np.allclose(
            struc.centroid(restored), struc.centroid(input_atoms), atol=1e-5
        )


@pytest.mark.parametrize("x", [0, 2 * np.pi])
@pytest.mark.parametrize("y", [0, 2 * np.pi])
@pytest.mark.parametrize("z", [0, 2 * np.pi])
@pytest.mark.parametrize("centered", [False, True])
def test_rotate_360(input_atoms, x, y, z, centered):
    """
    Rotate by 360 degrees and expect that the coordinates have not
    changed.
    """
    func = struc.rotate_centered if centered else struc.rotate
    rotated = func(input_atoms, [x, y, z])

    assert isinstance(rotated, type(input_atoms))
    assert struc.coord(rotated).shape == struc.coord(input_atoms).shape
    assert np.allclose(struc.coord(rotated), struc.coord(input_atoms), atol=1e-5)
    if centered and struc.coord(input_atoms).ndim > 1:
        assert np.allclose(
            struc.centroid(rotated), struc.centroid(input_atoms), atol=1e-5
        )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_rotate_known(ndim):
    """
    Rotate a vector at the Y-axis about the X-axis by 90 degrees and
    expect a rotated vector at the Z-axis.
    """
    shape = (1,) * (ndim - 1) + (3,)
    vector = np.zeros(shape)
    vector[...] = [0, 1, 0]

    exp_rotated = np.zeros(shape)
    exp_rotated[...] = [0, 0, 1]

    # Rotation by 90 degrees
    test_rotated = struc.rotate(vector, [0.5 * np.pi, 0, 0])

    assert test_rotated.shape == exp_rotated.shape
    assert np.allclose(test_rotated, exp_rotated, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1, 2])  # x, y, z
@pytest.mark.parametrize("random_seed", np.arange(5))
def test_rotate_measure(axis, random_seed):
    """
    Rotate and measure resulting angle that should be equal to the input
    angle.
    """
    np.random.seed(random_seed)
    # Maximum rotation is only 180 degrees,
    # as with higher angles the measured angle would decrease again
    ref_angle = np.random.rand() * np.pi
    angles = np.zeros(3)
    angles[axis] = ref_angle

    # The measured angle is only equal to the input angle,
    # if the input coordinates have no component on the rotation axis
    input_coord = np.ones(3)
    input_coord[axis] = 0

    rotated = struc.rotate(input_coord, angles)
    test_angle = struc.angle(rotated, 0, input_coord)

    # Vector length should be unchanged
    assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(input_coord))
    assert test_angle == pytest.approx(ref_angle)


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("use_support", [False, True])
@pytest.mark.parametrize("random_seed", np.arange(5))
def test_rotate_about_axis(input_atoms, as_list, use_support, random_seed):
    """
    Rotate and rotate back and check if the coordinates are still
    the same.
    """
    np.random.seed(random_seed)
    axis = np.random.rand(3)
    angle = np.random.rand()
    neg_angle = -angle
    support = np.random.rand(3) if use_support else None
    if as_list:
        axis = axis.tolist()
        support = support.tolist() if support is not None else None

    rotated = struc.rotate_about_axis(input_atoms, axis, angle, support)
    restored = struc.rotate_about_axis(rotated, axis, neg_angle, support)

    assert isinstance(restored, type(input_atoms))
    assert struc.coord(restored).shape == struc.coord(input_atoms).shape
    assert np.allclose(struc.coord(restored), struc.coord(input_atoms), atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1, 2])  # x, y, z
@pytest.mark.parametrize("random_seed", np.arange(5))
def test_rotate_about_axis_consistency(input_atoms, axis, random_seed):
    """
    Compare the outcome of :func:`rotate_about_axis()` with
    :func:`rotate()`.
    """
    np.random.seed(random_seed)
    angle = np.random.rand() * 2 * np.pi

    angles = np.zeros(3)
    angles[axis] = angle
    ref_rotated = struc.rotate(input_atoms, angles)

    rot_axis = np.zeros(3)
    # Length of axis should be irrelevant
    rot_axis[axis] = np.random.rand()
    test_rotated = struc.rotate_about_axis(
        input_atoms,
        rot_axis,
        angle,
    )

    assert isinstance(test_rotated, type(ref_rotated))
    assert struc.coord(test_rotated).shape == struc.coord(ref_rotated).shape
    assert np.allclose(struc.coord(test_rotated), struc.coord(ref_rotated), atol=1e-5)


@pytest.mark.parametrize("random_seed", np.arange(5))
@pytest.mark.parametrize("use_support", [False, True])
def test_rotate_about_axis_360(input_atoms, random_seed, use_support):
    """
    Rotate by 360 degrees around an arbitrary axis and expect that the
    coordinates have not changed.
    """
    np.random.seed(random_seed)
    axis = np.random.rand(3)
    support = np.random.rand(3) if use_support else None

    rotated = struc.rotate_about_axis(input_atoms, axis, 2 * np.pi, support)

    assert isinstance(rotated, type(input_atoms))
    assert struc.coord(rotated).shape == struc.coord(input_atoms).shape
    assert np.allclose(struc.coord(rotated), struc.coord(input_atoms), atol=1e-5)


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize(
    "order",
    (
        np.array([0, 1, 2]),
        np.array([0, 2, 1]),
        np.array([1, 0, 2]),
        np.array([2, 0, 1]),
        np.array([2, 1, 0]),
        np.array([1, 2, 0]),
    ),
)
def test_orient_principal_components(input_atoms, as_list, order):
    """
    Orient atoms such that the variance in each axis is greatest
    where the value for `order` is smallest.

    The number of dimensions in input_atoms must be 2.
    """
    if struc.coord(input_atoms).ndim != 2:
        with pytest.raises(ValueError):
            struc.orient_principal_components(input_atoms)
        return

    if as_list:
        order = order.tolist()

    result = struc.orient_principal_components(input_atoms, order=order)
    neg_variance = -struc.coord(result).var(axis=0)
    assert isinstance(result, type(input_atoms))
    assert (neg_variance.argsort() == np.argsort(order)).all()


@pytest.mark.parametrize("bad_order", (np.array([2, 3, 4]), np.arange(4)))
def test_orient_principal_components_bad_order(input_atoms, bad_order):
    """
    Ensure correct inputs for `order`.
    """
    with pytest.raises(ValueError):
        struc.orient_principal_components(input_atoms, order=bad_order)


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("use_support", [False, True])
@pytest.mark.parametrize("random_seed", np.arange(5))
def test_align_vectors(input_atoms, as_list, use_support, random_seed):
    """
    Align, swap alignment vectors and align again.
    Expect that the original coordinates have been restored.
    """
    np.random.seed(random_seed)
    source_direction = np.random.rand(3)
    target_direction = np.random.rand(3)
    if use_support:
        source_position = np.random.rand(3)
        target_position = np.random.rand(3)
    else:
        source_position = None
        target_position = None

    if as_list:
        source_direction = source_direction.tolist()
        target_direction = target_direction.tolist()
        if use_support:
            source_position = source_position.tolist()
            target_position = target_position.tolist()

    transformed = struc.align_vectors(
        input_atoms,
        source_direction,
        target_direction,
        source_position,
        target_position,
    )
    restored = struc.align_vectors(
        transformed,
        target_direction,
        source_direction,
        target_position,
        source_position,
    )

    assert isinstance(restored, type(input_atoms))
    assert struc.coord(restored).shape == struc.coord(input_atoms).shape
    assert np.allclose(struc.coord(restored), struc.coord(input_atoms), atol=1e-5)


def test_align_vectors_non_vector_inputs(input_atoms):
    """
    Ensure input vectors to ``struct.align_vectors`` have the correct shape.
    """
    source_direction = np.random.rand(2, 3)
    target_direction = np.random.rand(2, 3)
    with pytest.raises(ValueError):
        struc.align_vectors(
            input_atoms,
            source_direction,
            target_direction,
        )
    source_direction = np.random.rand(4)
    target_direction = np.random.rand(4)
    with pytest.raises(ValueError):
        struc.align_vectors(
            input_atoms,
            source_direction,
            target_direction,
        )
