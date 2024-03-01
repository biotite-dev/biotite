import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
from os.path import join
from ..util import data_dir
import pytest

def test_gyration_radius():
    stack = strucio.load_structure(join(data_dir("structure"), "1l2y.bcif"))
    radii = struc.gyration_radius(stack)
    # Compare with results from MDTraj
    exp_radii = \
       [7.30527532, 7.34189463, 7.21863721, 7.29877736, 7.25389752, 7.22292189,
        7.20646252, 7.27215909, 7.30437723, 7.30455437, 7.37979331, 7.14176259,
        7.20674397, 7.27594995, 7.31665835, 7.29850786, 7.34378951, 7.2642137,
        7.20727158, 7.16336879, 7.3479218,  7.19362027, 7.24841519, 7.29229237,
        7.15243826, 7.31285673, 7.22585756, 7.25467109, 7.3493648,  7.34203588,
        7.3310182,  7.29236536, 7.20527373, 7.33138918, 7.2284936,  7.40374312,
        7.24856173, 7.25581809]
    assert radii.tolist() == pytest.approx(exp_radii, abs=2e-2)

    # Same for atom array instead of stack
    array = stack[0]
    radius = struc.gyration_radius(array)
    assert radius == pytest.approx(exp_radii[0], abs=2e-2)