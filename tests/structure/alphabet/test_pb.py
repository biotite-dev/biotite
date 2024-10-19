from pathlib import Path
import numpy as np
import pytest
import biotite.sequence.io.fasta as fasta
import biotite.structure as struc
import biotite.structure.alphabet as strucalph
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture
def reference_sequence():
    """
    Get the reference Protein Blocks sequence for the alphabet example structure.
    """
    _, seq_string = next(
        fasta.FastaFile.read_iter(Path(data_dir("structure")) / "alphabet" / "pb.fasta")
    )
    return strucalph.ProteinBlocksSequence(seq_string)


@pytest.fixture
def reference_chain():
    pdbx_file = pdbx.BinaryCIFFile.read(
        Path(data_dir("structure")) / "alphabet" / "1ay7.bcif"
    )
    atoms = pdbx.get_structure(pdbx_file, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    chain = atoms[atoms.chain_id == "B"]
    return chain


def test_to_protein_blocks(reference_chain, reference_sequence):
    """
    Test the structure conversion to protein blocks based on a reference example from
    the PBexplore documentation
    (https://pbxplore.readthedocs.io/en/latest/intro_PB.html).
    """
    test_pb_sequences, _ = strucalph.to_protein_blocks(reference_chain)

    assert len(test_pb_sequences) == 1
    assert str(test_pb_sequences[0]) == str(reference_sequence)


def test_missing_residues(reference_chain, reference_sequence):
    """
    Like, `test_to_protein_blocks()`, but in some residues backbone atoms are missing.
    Expect that these and adjacent residues get the unknown symbol 'Z' in the
    PB sequence.
    """
    N_DELETIONS = 5
    # The 'Z' symbol
    UNDEFINED_SYMBOL = strucalph.ProteinBlocksSequence.undefined_symbol

    # Randomly delete some backbone atoms
    rng = np.random.default_rng(1)
    del_backbone_residue_ids = rng.choice(
        np.unique(reference_chain.res_id), N_DELETIONS, replace=False
    )
    reference_chain = reference_chain[
        ~np.isin(reference_chain.res_id, del_backbone_residue_ids)
        | ~np.isin(reference_chain.atom_name, ("N", "CA", "C"))
    ]

    # Apply the same deletions to the reference sequence
    for res_id in del_backbone_residue_ids:
        seq_index = res_id - reference_chain.res_id[0]
        # Convert the PB symbol for residue and adjacent ones to 'Z'
        start_index = max(0, seq_index - 2)
        end_index = min(len(reference_sequence), seq_index + 2)
        reference_sequence[start_index : end_index + 1] = UNDEFINED_SYMBOL

    test_pb_sequences, _ = strucalph.to_protein_blocks(reference_chain)

    assert len(test_pb_sequences) == 1
    assert str(test_pb_sequences[0]) == str(reference_sequence)
