# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
from biotite.structure import Atom
import numpy as np
import pytest

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
    assert density[0,2] == 2
    assert density[1,0] == 3
    assert density[1,1] == 1

    density, (x, y, z) = struc.density(stack)
    assert np.array_equal(x, [0.5, 1.5, 2.5])
    assert np.array_equal(y, [0.5, 1.5, 2.5, 3.5])
    assert np.array_equal(z, [1.0, 2.0])
    assert density.sum() == 12
    assert density[0,2] == 4
    assert density[1,0] == 6
    assert density[1,1] == 2

def test_density_with_bins(array):
    bins = np.array([[0, 1, 2, 3],[0, 1, 2, 3],[0, 1, 2, 3]])
    density, (x, y, z) = struc.density(array, bins=bins)
    assert np.array_equal(x, [0,1,2,3])
    assert np.array_equal(y, [0,1,2,3])
    assert np.array_equal(z, [0,1,2,3])
    assert density.sum() == 6
    assert density[0,2,1] == 2
    assert density[1,1,1] == 1
    assert density[2,0,1] == 3

def test_density_with_delta(array):
    density, (x, y, z) = struc.density(array, delta=5.0)
    assert density.shape == (1, 1, 1)
    assert density.sum() == 6
    assert density[0,0,0] == 6

def test_density_normalized(array):
    density, (x, y, z) = struc.density(array, density=True)
    assert np.abs(density.sum() - 1.0) < 0.0001
    assert np.abs(density[0,2] - 2.0/6.0) < 0.0001

def test_density_weights(array, stack):
    atomic_weights = np.array([1, 1, 1, 1, 1, 0.5])

    # weights should be applied correctly
    density, (x, y, z) = struc.density(array, weights=atomic_weights)
    assert density.sum() == 5.5
    assert density[0,2] == 2
    assert density[1,0] == 2.5
    assert density[1,1] == 1

    # weights should be repeated along stack dimension
    density, (x, y, z) = struc.density(stack, weights=atomic_weights)
    assert density.sum() == 11
    assert density[0,2] == 4
    assert density[1,0] == 5
    assert density[1,1] == 2

    # weights should not be repeated along stack dimension for NxM
    stack_weights = np.array([atomic_weights, atomic_weights])
    density, (x, y, z) = struc.density(stack, weights=stack_weights)
    assert density.sum() == 11
    assert density[0,2] == 4
    assert density[1,0] == 5
    assert density[1,1] == 2

