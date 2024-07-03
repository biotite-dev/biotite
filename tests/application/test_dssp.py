# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
from subprocess import SubprocessError
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdbx as pdbx
from biotite.application.dssp import DsspApp
from ..util import data_dir, is_not_installed


@pytest.mark.skipif(is_not_installed("mkdssp"), reason="DSSP is not installed")
def test_multiple_chains():
    atoms = pdbx.get_structure(
        pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1igy.bcif")), model=1
    )
    atoms = atoms[struc.filter_canonical_amino_acids(atoms)]
    sse = DsspApp.annotate_sse(atoms)
    assert np.all(np.isin(sse, ["C", "H", "B", "E", "G", "I", "T", "S"]))
    assert len(sse) == struc.get_residue_count(atoms)


@pytest.mark.skipif(is_not_installed("mkdssp"), reason="DSSP is not installed")
def test_invalid_structure():
    array = strucio.load_structure(join(data_dir("structure"), "5ugo.bcif"))
    # Get DNA chain -> Invalid for DSSP
    chain = array[array.chain_id == "T"]
    with pytest.raises(SubprocessError):
        DsspApp.annotate_sse(chain)
