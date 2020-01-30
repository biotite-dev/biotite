# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import numpy as np
import pytest


@pytest.fixture
def stack():
    stack = struc.AtomArrayStack(depth=3, length=5)
    stack.coord = np.arange(45).reshape((3,5,3))
    return stack


@pytest.mark.parametrize("as_coord", [False, True])
def test_rmsd(stack, as_coord):
    if as_coord:
        stack = stack.coord
    assert struc.rmsd(stack[0], stack).tolist() \
           == pytest.approx([0.0, 25.98076211, 51.96152423])
    assert struc.rmsd(stack[0], stack[1]) \
            == pytest.approx(25.9807621135)


@pytest.mark.parametrize("as_coord", [False, True])
def test_rmsf(stack, as_coord):
    if as_coord:
        stack = stack.coord
    assert struc.rmsf(struc.average(stack), stack).tolist() \
           == pytest.approx([21.21320344] * 5)