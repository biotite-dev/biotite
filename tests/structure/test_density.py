# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
from biotite.structure import Atom
import numpy as np
import pytest

@pytest.fixture
def stack():
    atom_list = []
    atom_list.append(Atom([0.5, 2.5, 1.0]))
    atom_list.append(Atom([0.5, 2.7, 1.0]))
    atom_list.append(Atom([1.5, 1.5, 1.0]))
    atom_list.append(Atom([2.5, 0.5, 1.0]))
    atom_list.append(Atom([2.5, 0.7, 1.0]))
    atom_list.append(Atom([2.5, 0.5, 1.1]))
    arr = struc.array(atom_list)
    return struc.stack([arr])

def test_density(stack):
    density, (x, y, z) = struc.density(stack)
    assert np.array_equal(x, [0.5, 1.5, 2.5])
    assert np.array_equal(y, [0.5, 1.5, 2.5, 3.5])
    assert np.array_equal(z, [1.0, 2.0])
    assert density.sum() == 6
    assert density[0,2] == 2
    assert density[1,0] == 3
    assert density[1,1] == 1

def test_density_with_bins(stack):
    bins = np.array([[0, 1, 2, 3],[0, 1, 2, 3],[0, 1, 2, 3]])
    density, (x, y, z) = struc.density(stack, bins=bins)
    assert np.array_equal(x, [0,1,2,3])
    assert np.array_equal(y, [0,1,2,3])
    assert np.array_equal(z, [0,1,2,3])
    assert density.sum() == 6
    assert density[0,2,1] == 2
    assert density[1,1,1] == 1
    assert density[2,0,1] == 3

def test_density_with_delta(stack):
    density, (x, y, z) = struc.density(stack, delta=5.0)
    assert density.shape == (1, 1, 1)
    assert density.sum() == 6
    assert density[0,0,0] == 6

def test_density_normalized(stack):
    density, (x, y, z) = struc.density(stack, density=True)
    assert np.abs(density.sum() - 1.0) < 0.0001
    assert np.abs(density[0,2] - 2.0/6.0) < 0.0001

