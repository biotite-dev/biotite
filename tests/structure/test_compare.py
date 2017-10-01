# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import biopython.structure as struc
import numpy as np
import pytest

@pytest.fixture
def stack():
    stack = struc.AtomArrayStack(depth=3, length=5)
    stack.coord = np.arange(45).reshape((3,5,3))
    return stack
    
def test_rmsd(stack):
    assert struc.rmsd(stack[0], stack).tolist() \
           == pytest.approx([0.0, 25.98076211, 51.96152423])
    assert struc.rmsd(stack[0], stack[1]) \
            == pytest.approx(25.9807621135)

def test_rmsf(stack):
    assert struc.rmsf(struc.average(stack), stack).tolist() \
           == pytest.approx([21.21320344] * 5)