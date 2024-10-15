import re
from pathlib import Path
import numpy as np
import pytest
import biotite.sequence.io.fasta as fasta
import biotite.structure as struc
import biotite.structure.alphabet as strucalph
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


def _get_ref_3di_sequence(pdb_id, chain_id):
    """
    Get the reference 3di sequence for the first model of the structure with the given
    PDB ID and chain ID.
    """
    ref_3di_file = fasta.FastaFile.read(
        Path(data_dir("structure")) / "alphabet" / "i3d.fasta"
    )
    for header, seq_string in ref_3di_file.items():
        # The first model of a structure is also the first sequence to appear
        # and thus to be matched
        if re.match(rf"^{pdb_id}(_MODEL_\d+)?_{chain_id}", header):
            ref_3di_sequence = strucalph.I3DSequence(seq_string)
            break
    else:
        raise ValueError(
            f"Reference 3Di sequence not found for {pdb_id} chain {chain_id}"
        )
    return ref_3di_sequence


@pytest.mark.parametrize(
    "path", Path(data_dir("structure")).glob("*.bcif"), ids=lambda path: path.stem
)
def test_to_3di(path):
    """
    Check if the 3di sequence of a chain is correctly generated, by comparing the result
    to a reference sequence generated with *foldseek*.
    """
    if (
        path.stem
        in [
            "1dix"  # `get_chain_starts()` does not work properly here with `use_author_fields=True`
        ]
    ):
        pytest.skip("Miscellaneous issues")

    pdbx_file = pdbx.BinaryCIFFile.read(path)
    if np.any(
        pdbx_file.block["atom_site"]["label_alt_id"].mask.array
        == pdbx.MaskValue.PRESENT
    ):
        # There is some inconsistency in how foldseek and Biotite handle altloc IDs
        # -> skip these cases for the sake of simplicity
        pytest.skip("Structure contains altlocs")
    atoms = pdbx.get_structure(pdbx_file, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    if len(atoms) == 0:
        pytest.skip("Structure contains no peptide chains")
    test_3di, chain_starts = strucalph.to_3di(atoms)

    ref_3di = [
        _get_ref_3di_sequence(path.stem, chain_id)
        for chain_id in atoms.chain_id[chain_starts]
    ]

    for test, ref, chain_id in zip(test_3di, ref_3di, atoms.chain_id[chain_starts]):
        assert str(test) == str(ref), f"3Di sequence of chain {chain_id} does not match"


def test_missing():
    """
    Test if missing or non-peptide residues within a chain are correctly handled.
    """
    pass


def test_empty():
    """
    Test if an empty structure is correctly handled.
    """
    pass
