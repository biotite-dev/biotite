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
    altloc_ids = pdbx_file.block["atom_site"]["label_alt_id"]
    if altloc_ids.mask is None or np.any(
        altloc_ids.mask.array == pdbx.MaskValue.PRESENT
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


def test_missing_residues():
    """
    Like, `test_to_protein_blocks()`, but in some residues backbone atoms are missing.
    Expect that these and adjacent residues get the unknown symbol 'Z' in the
    PB sequence.
    """
    PDB_ID = "1aki"
    N_DELETIONS = 5
    MAX_MISMATCH_PERCENTAGE = 0.1
    UNDEFINED_SYMBOL = strucalph.I3DSequence.undefined_symbol

    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / f"{PDB_ID}.bcif")
    atoms = pdbx.get_structure(pdbx_file, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]

    # Randomly delete some backbone atoms
    rng = np.random.default_rng(1)
    del_backbone_residue_ids = rng.choice(
        np.unique(atoms.res_id), N_DELETIONS, replace=False
    )
    atoms = atoms[
        ~np.isin(atoms.res_id, del_backbone_residue_ids)
        | ~np.isin(atoms.atom_name, ("N", "CA", "CB", "C"))
    ]
    test_sequences, _ = strucalph.to_3di(atoms)

    # Apply the same deletions to the reference sequence
    ref_sequence = _get_ref_3di_sequence(PDB_ID, atoms.chain_id[0])
    for res_id in del_backbone_residue_ids:
        seq_index = res_id - atoms.res_id[0]
        # Convert the PDB symbol for residue and adjacent ones to 'Z'
        start_index = max(0, seq_index - 1)
        end_index = min(len(ref_sequence), seq_index + 1)
        ref_sequence[start_index : end_index + 1] = UNDEFINED_SYMBOL

    assert len(test_sequences) == 1
    # 3Di sequences are quite complex, i.e. removing backbone atoms at some position
    # might alter the symbols in remote positions
    # -> Allow for mismatches
    n_mismatches = np.count_nonzero(test_sequences[0].code != ref_sequence.code)
    assert n_mismatches / len(ref_sequence) <= MAX_MISMATCH_PERCENTAGE
