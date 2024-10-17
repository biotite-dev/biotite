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
    Get the reference CLePAPS sequence for the the structure with the given
    PDB ID and chain ID.
    """
    ref_3di_file = fasta.FastaFile.read(
        Path(data_dir("structure")) / "alphabet" / "clepaps.fasta"
    )
    return strucalph.ClepapsSequence(ref_3di_file[f"{pdb_id.lower()}_{chain_id}"])


@pytest.mark.parametrize("pdb_id, chain_id", [("1mol", "A"), ("1cew", "I")])
def test_to_clepaps(pdb_id, chain_id):
    """
    Test the structure conversion to CLePAPS based on a reference example from
    presentation slides.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(
        Path(data_dir("structure")) / "alphabet" / f"{pdb_id}.bcif"
    )
    atoms = pdbx.get_structure(pdbx_file, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    chain = atoms[atoms.chain_id == chain_id]
    test_sequences, _ = strucalph.to_clepaps(chain)

    ref_sequence = _get_ref_3di_sequence(pdb_id, chain_id)

    # Only a single chain was used as input -> expect only one sequence
    assert len(test_sequences) == 1
    assert str(test_sequences[0]) == str(ref_sequence)


@pytest.mark.parametrize("pdb_id, chain_id", [("1mol", "A"), ("1cew", "I")])
def test_missing_residues(pdb_id, chain_id):
    """
    Like, `test_to_clepaps()`, but in some residues backbone atoms are missing.
    Expect that these and adjacent residues get the unknown symbol 'R' in the
    CLePAPs sequence.
    """
    N_DELETIONS = 5
    # The 'R' symbol
    UKNOWN_SYMBOL = strucalph.ClepapsSequence.unknown_symbol

    pdbx_file = pdbx.BinaryCIFFile.read(
        Path(data_dir("structure")) / "alphabet" / f"{pdb_id}.bcif"
    )
    atoms = pdbx.get_structure(pdbx_file, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    chain = atoms[atoms.chain_id == chain_id]

    # Randomly delete some backbone atoms
    rng = np.random.default_rng(1)
    del_backbone_residue_ids = rng.choice(
        np.unique(chain.res_id), N_DELETIONS, replace=False
    )
    chain = chain[
        ~np.isin(chain.res_id, del_backbone_residue_ids) | ~(chain.atom_name == "CA")
    ]

    test_sequences = strucalph.to_clepaps(chain)

    # Apply the same deletions to the reference sequence
    ref_sequence, _ = strucalph.to_clepaps(chain)
    for res_id in del_backbone_residue_ids:
        seq_index = res_id - chain.res_id[0]
        # Convert the symbol for residue and adjacent ones to 'R'
        start_index = max(0, seq_index - 2)
        end_index = min(len(ref_sequence), seq_index + 1)
        ref_sequence[start_index : end_index + 1] = UKNOWN_SYMBOL

    assert len(test_sequences) == 1
    assert str(test_sequences[0]) == str(ref_sequence)
