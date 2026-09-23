import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from biotite.application_v2.dssp import DsspApp, DsspElement
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
    Check if the DSSP annotation has the correct length and reasonable
    values.
    """
    atoms = pdbx.get_structure(
        pdbx.BinaryCIFFile.read(data_dir("structure") / "pdb" / f"{pdb_id}.bcif"),
        model=1,
    )
    atoms = atoms[struc.filter_amino_acids(atoms)]
    sse = DsspApp().run(atoms).result()

    assert np.all(np.isin(sse, list(DsspElement)))
    # One SSE per residue
    assert len(sse) == struc.get_residue_count(atoms)


def test_symbol_conversion():
    """
    Converting :class:`DsspElement` values to symbols and back must
    restore the original values and unknown symbols must be rejected.
    """
    ref_sse = np.array([sse.value for sse in DsspElement] * 3)
    symbols = DsspElement.to_symbols(ref_sse)
    assert DsspElement.from_symbols(symbols).tolist() == ref_sse.tolist()
    for sse in DsspElement:
        assert DsspElement.from_symbol(sse.symbol) == sse
    with pytest.raises(ValueError):
        DsspElement.from_symbols(["H", "x"])


@pytest.mark.skipif(is_not_installed("mkdssp"), reason="DSSP is not installed")
def test_invalid_structure():
    """
    Check if an exception is raised, if the input structure contains
    non-amino-acid residues.
    """
    array = pdbx.get_structure(
        pdbx.BinaryCIFFile.read(data_dir("structure") / "pdb" / "5ugo.bcif"), model=1
    )
    # Get DNA chain -> Invalid for DSSP
    chain = array[array.chain_id == "T"]
    with pytest.raises(struc.BadStructureError):
        DsspApp().run(chain)
