# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.mmtf as mmtf
from os.path import join
import numpy as np
import pytest
from ..util import data_dir

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

@pytest.fixture
def load_stack_superimpose():
    stack = strucio.load_structure(join(
        data_dir("structure"), "1l2y.mmtf"
    ))
    # Superimpose with first frame
    bb_mask = struc.filter_peptide_backbone(stack[0])
    supimp, _ = struc.superimpose(stack[0], stack, atom_mask=bb_mask)
    return stack, supimp

def test_rmsd_gmx(load_stack_superimpose):
    """
    Comparison of RMSD values computed with Biotite with results 
    obtained from GROMACS 2021.5. 
    """
    stack, supimp = load_stack_superimpose
    rmsd = struc.rmsd(stack[0], supimp)/10
    
    # Gromacs RMSDs -> Without mass-weighting:
    # echo "Backbone Protein" | \
    # gmx rms -s 1l2y.gro -f 1l2y.xtc -o rmsd.xvg -mw no
    rmsd_gmx = np.array([
        0.0005037, 0.1957698, 0.2119313, 0.2226127, 0.184382, 
        0.2210998, 0.2712815, 0.1372861, 0.2348654, 0.1848784, 
        0.1893576, 0.2500543, 0.1946374, 0.2101624, 0.2180645, 
        0.1836762, 0.1681345, 0.2363865, 0.2287371, 0.2546207, 
        0.1604872, 0.2167119, 0.2176063, 0.2069806, 0.2535706, 
        0.2682233, 0.2252388, 0.2419151, 0.2343987, 0.1902994, 
        0.2334525, 0.2010523, 0.215444, 0.1786632, 0.2652018, 
        0.174061, 0.2591569, 0.2602662
    ])
    
    assert np.allclose(rmsd, rmsd_gmx, atol=1e-03)

def test_rmspd_gmx(load_stack_superimpose):
    """
    Comparison of the RMSPD computed with Biotite with results 
    obtained from GROMACS 2021.5.
    """
    stack, _ = load_stack_superimpose
    rmspd = struc.rmspd(stack[0], stack)/10
    
    # Gromacs RMSDist:
    # echo "Protein" | \
    # gmx rmsdist -f 1l2y.xtc -s 1l2y.gro -o rmsdist.xvg -sumh no -pbc no                    
    rmspd_gmx = np.array([
        0.000401147, 0.125482, 0.138913, 0.138847, 0.113917, 
        0.132915, 0.173084, 0.103089, 0.156309, 0.114694, 
        0.12964, 0.15875, 0.12876, 0.128983, 0.137031, 
        0.126059, 0.106726, 0.154244, 0.144405, 0.174041, 
        0.10417, 0.130936, 0.141216, 0.125559, 0.171342, 
        0.165306, 0.137616, 0.154447, 0.146337, 0.116433, 
        0.154976, 0.128477, 0.150537, 0.111494, 0.173234, 
        0.116638, 0.169524, 0.15953
    ])

    assert np.allclose(rmspd, rmspd_gmx, atol=1e-03)

def test_rmsf_gmx(load_stack_superimpose):
    """
    Comparison of RMSF values computed with Biotite with results 
    obtained from GROMACS 2021.5.     
    """
    stack, supimp = load_stack_superimpose
    ca_mask = ((stack[0].atom_name == "CA") & (stack[0].element == "C"))
    rmsf = struc.rmsf(struc.average(supimp[:, ca_mask]), supimp[:, ca_mask])/10
    
    # Gromacs RMSF:
    # echo "C-alpha" | gmx rmsf -s 1l2y.gro -f 1l2y.xtc -o rmsf.xvg -res                    
    rmsf_gmx = np.array([
        0.1379, 0.036, 0.0261, 0.0255, 0.029, 0.0204, 0.0199, 
        0.0317, 0.0365, 0.0249, 0.0269, 0.032, 0.0356, 0.0446, 
        0.059, 0.037, 0.0331, 0.0392, 0.0403, 0.0954
    ])

    assert np.allclose(rmsf, rmsf_gmx, atol=1e-02)