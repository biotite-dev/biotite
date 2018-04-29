# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from os.path import join
from .util import data_dir
import pytest

# Expected values from: http://curie.utmb.edu/getarea.html
@pytest.mark.parametrize("pdb_id, expect",
                         [("1igy", 60765.54),
                          ("1l2y", 1815.51),
                          pytest.mark.xfail(("3o5r", 6391.02)),
                          ("5h73", 14963.58)])
def test_sasa(pdb_id, expect):
    file = pdbx.PDBxFile()
    file.read(join(data_dir, pdb_id+".cif"))
    array = pdbx.get_structure(file, model=1)
    sasa = struc.sasa(array)
    assert np.nansum(sasa) == pytest.approx(expect, rel=0.05)