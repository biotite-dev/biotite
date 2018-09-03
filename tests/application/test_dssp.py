# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import glob
import shutil
from subprocess import SubprocessError
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.mmtf as mmtf
from biotite.application.dssp import DsspApp
from ..structure.util import data_dir


@pytest.mark.skipif(shutil.which("mkdssp") is None,
                    reason="DSSP is not installed")
@pytest.mark.parametrize("path", glob.glob(join(data_dir, "*.mmtf")))
def test_dssp(path):
    sec_struct_codes = {0 : "I",
                        1 : "S",
                        2 : "H",
                        3 : "E",
                        4 : "G",
                        5 : "B",
                        6 : "T",
                        7 : "C"}

    mmtf_file = mmtf.MMTFFile()
    mmtf_file.read(path)
    array = mmtf.get_structure(mmtf_file, model=1)
    array = array[array.hetero == False]
    first_chain_id = array.chain_id[0]
    chain = array[array.chain_id == first_chain_id]

    n_residues = struc.get_residue_count(chain)
    # Secondary structure annotation in PDB use also DSSP
    # -> compare PDB and local DSSP
    sse = mmtf_file["secStructList"]
    sse = sse[:n_residues]
    if (sse == -1).all():
        # First chain is not a polypeptide chain (presumably DNA/RNA)
        # DSSP not applicable -> return
        return
    sse = np.array([sec_struct_codes[code] for code in sse],
                    dtype="U1")
    
    chain = array[array.chain_id == first_chain_id]
    sse_from_app = DsspApp.annotate_sse(chain)
    np.set_printoptions(threshold=10000)
    # PDB uses different DSSP version -> slight differences possible
    # -> only 95% must be identical
    assert np.count_nonzero(sse_from_app == sse) / len(sse) > 0.95


@pytest.mark.skipif(shutil.which("mkdssp") is None,
                    reason="DSSP is not installed")
def test_invalid_structure():
    array = strucio.load_structure(join(data_dir, "5ugo.mmtf"))
    # Get DNA chain -> Invalid for DSSP
    chain = array[array.chain_id == "T"]
    with pytest.raises(SubprocessError):
        DsspApp.annotate_sse(chain)

