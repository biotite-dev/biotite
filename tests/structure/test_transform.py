# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.npz as npz
from .util import data_dir


@pytest.fixture(
    params=itertools.product(
        [1, 2, 3],    # ndim
        [False, True]    # as_coord
    )
)
def input_atoms(request):
    ndim, as_coord = request.param

    file = npz.NpzFile()
    file.read(join(data_dir, "1l2y.npz"))
    atoms = file.get_structure()
    
    if ndim == 2:
        # Only one model
        atoms = atoms[0]
    elif ndim == 1:
        # Only one atom
        atoms = atoms[0,0]
    
    if as_coord:
        return atoms.coord
    else:
        return atoms


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

    assert type(restored) == type(input_atoms)
    assert struc.coord(restored).shape == struc.coord(input_atoms).shape
    assert np.allclose(
        struc.coord(restored), struc.coord(input_atoms), atol=1e-6
    )


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("random_seed", np.arange(5))
@pytest.mark.parametrize("centered", [False, True])
def test_rotate(input_atoms, as_list, random_seed, centered):
    """
    Rotate and rotate back and check if the coordinates are still
    the same.
    """
    np.random.seed(random_seed)
    angles = np.random.rand(3) * 2*np.pi
    neg_angles = -angles
    if as_list:
        angles = angles.tolist()
        neg_angles = neg_angles.tolist()
    
    func = struc.rotate_centered if centered else struc.rotate
    rotated = func(input_atoms, angles)
    restored = func(rotated, neg_angles)

    assert type(restored) == type(input_atoms)
    assert struc.coord(restored).shape == struc.coord(input_atoms).shape
    assert np.allclose(
        struc.coord(restored), struc.coord(input_atoms), atol=1e-6
    )
    if centered and struc.coord(input_atoms).ndim > 1:
        assert np.allclose(
            struc.centroid(restored), struc.centroid(input_atoms), atol=1e-6
        )


@pytest.mark.parametrize("x", [0, 2*np.pi])
@pytest.mark.parametrize("y", [0, 2*np.pi])
@pytest.mark.parametrize("z", [0, 2*np.pi])
@pytest.mark.parametrize("centered", [False, True])
def test_rotate_360(input_atoms, x, y, z, centered):
    """
    Rotate by 360 degrees and expect that the coordinates have not
    changed.
    """
    func = struc.rotate_centered if centered else struc.rotate
    rotated = func(input_atoms, [x, y, z])
    
    assert type(rotated) == type(input_atoms)
    assert struc.coord(rotated).shape == struc.coord(input_atoms).shape
    assert np.allclose(
        struc.coord(rotated), struc.coord(input_atoms), atol=1e-6
    )
    if centered and struc.coord(input_atoms).ndim > 1:
        assert np.allclose(
            struc.centroid(rotated), struc.centroid(input_atoms), atol=1e-6
        )