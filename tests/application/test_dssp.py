# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdbx as pdbx
from biotite.application.dssp import DsspApp
from tests.util import data_dir, is_not_installed


@pytest.mark.parametrize(
    "pdb_id",
    [
        "1aki",  # Single chain
        "1igy",  # Multiple chains
        "5eil",  # Contains non-canonical amino acid
    ],
)
@pytest.mark.skipif(is_not_installed("mkdssp"), reason="DSSP is not installed")
def test_annotation(pdb_id):
    """
    Check if the DSSP annotation has the correct length and reasonable values.
    """
    atoms = pdbx.get_structure(
        pdbx.BinaryCIFFile.read(join(data_dir("structure"), f"{pdb_id}.bcif")), model=1
    )
    atoms = atoms[struc.filter_amino_acids(atoms)]
    sse = DsspApp.annotate_sse(atoms)

    assert np.all(np.isin(sse, ["C", "H", "B", "E", "G", "I", "T", "S", "P"]))
    # One SSE per residue
    assert len(sse) == struc.get_residue_count(atoms)


@pytest.mark.skipif(is_not_installed("mkdssp"), reason="DSSP is not installed")
def test_invalid_structure():
    """
    Check if an exception is raised, if the input structure contains non-amino-acid
    residues.
    """
    array = strucio.load_structure(join(data_dir("structure"), "5ugo.bcif"))
    # Get DNA chain -> Invalid for DSSP
    chain = array[array.chain_id == "T"]
    with pytest.raises(struc.BadStructureError):
        DsspApp.annotate_sse(chain)
