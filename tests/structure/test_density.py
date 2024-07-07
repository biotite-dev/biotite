# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.structure as struc
from biotite.structure import Atom


@pytest.fixture
def array():
    atom_list = []
    atom_list.append(Atom([0.5, 2.5, 1.0]))
    atom_list.append(Atom([0.5, 2.7, 1.0]))
    atom_list.append(Atom([1.5, 1.5, 1.0]))
    atom_list.append(Atom([2.5, 0.5, 1.0]))
    atom_list.append(Atom([2.5, 0.7, 1.0]))
    atom_list.append(Atom([2.5, 0.5, 1.1]))
    return struc.array(atom_list)


@pytest.fixture
def stack(array):
    return struc.stack([array, array.copy()])


def test_density(array, stack):
    density, (x, y, z) = struc.density(array)
    assert np.array_equal(x, [0.5, 1.5, 2.5])
    assert np.array_equal(y, [0.5, 1.5, 2.5, 3.5])
    assert np.array_equal(z, [1.0, 2.0])
    assert density.sum() == 6
    assert density[0, 2] == 2
    assert density[1, 0] == 3
    assert density[1, 1] == 1

    density, (x, y, z) = struc.density(stack)
    assert np.array_equal(x, [0.5, 1.5, 2.5])
    assert np.array_equal(y, [0.5, 1.5, 2.5, 3.5])
    assert np.array_equal(z, [1.0, 2.0])
    assert density.sum() == 12
    assert density[0, 2] == 4
    assert density[1, 0] == 6
    assert density[1, 1] == 2


def test_density_with_bins(array):
    bins = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    density, (x, y, z) = struc.density(array, bins=bins)
    assert np.array_equal(x, [0, 1, 2, 3])
    assert np.array_equal(y, [0, 1, 2, 3])
    assert np.array_equal(z, [0, 1, 2, 3])
    assert density.sum() == 6
    assert density[0, 2, 1] == 2
    assert density[1, 1, 1] == 1
    assert density[2, 0, 1] == 3


def test_density_with_delta(array):
    density, (x, y, z) = struc.density(array, delta=5.0)
    assert density.shape == (1, 1, 1)
    assert density.sum() == 6
    assert density[0, 0, 0] == 6


def test_density_normalized(array):
    density, (x, y, z) = struc.density(array, density=True)
    assert np.abs(density.sum() - 1.0) < 0.0001
    assert np.abs(density[0, 2] - 2.0 / 6.0) < 0.0001


def test_density_weights(array, stack):
    # assign weights to coordinates
    atomic_weights = np.arange(0.1, 0.7, 0.1)

    # weights should be applied correctly
    density, (x, y, z) = struc.density(array, weights=atomic_weights)
    assert density.sum() == atomic_weights.sum()
    assert density[0, 2] == atomic_weights[0] + atomic_weights[1]
    assert density[1, 0] == atomic_weights[3:].sum()
    assert density[1, 1] == atomic_weights[2]

    # weights should be repeated along stack dimensions and lead to the same
    # result independent of shape
    density, (x, y, z) = struc.density(stack, weights=atomic_weights)
    density2, (x, y, z) = struc.density(
        stack, weights=np.array([atomic_weights, atomic_weights])
    )
    assert density.sum() == density2.sum()
    assert density[0, 2] == density2[0, 2]
    assert density[1, 0] == density2[1, 0]
    assert density[1, 1] == density2[1, 1]
