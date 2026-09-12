import numpy as np
import pytest
import biotite.sequence.align as align
import biotite.sequence.io.fasta as fasta
import biotite.structure as struc
import biotite.structure.alphabet as strucalph
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


def _get_reference_sequence(pdb_id, chain_id):
    """
    Get the reference CLePAPS sequence for the structure with the given PDB ID and
    chain ID.
    """
    ref_file = fasta.FastaFile.read(
        data_dir("structure") / "alphabet" / "clepaps.fasta"
    )
    return strucalph.ClepapsSequence(ref_file[f"{pdb_id.lower()}_{chain_id}"])


def _load_chain(pdb_id, chain_id):
    pdbx_file = pdbx.BinaryCIFFile.read(
        data_dir("structure") / "alphabet" / f"{pdb_id}.bcif"
    )
    atoms = pdbx.get_structure(pdbx_file, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    return atoms[atoms.chain_id == chain_id]


@pytest.mark.parametrize("pdb_id, chain_id", [("1mol", "A")])
def test_to_clepaps(pdb_id, chain_id):
    """
    Test the structure conversion to CLePAPS based on a reference example from
    presentation slides.
    """
    # The conformational letter parameters published in the CLePAPS paper are rounded to
    # only a few significant digits and the original software is not available anymore.
    # Hence, the reference sequence cannot be reproduced exactly.
    # Instead, a high fraction of the symbols must match and each deviating symbol must be
    # geometrically similar to the expected one, i.e. its CLESUM substitution score must
    # be sufficiently high.
    MIN_PERCENTAGE = 0.95
    MIN_SCORE = 10

    chain = _load_chain(pdb_id, chain_id)
    test_sequences, _ = strucalph.to_clepaps(chain)
    ref_sequence = _get_reference_sequence(pdb_id, chain_id)

    # Only a single chain was used as input -> expect only one sequence
    assert len(test_sequences) == 1
    test_sequence = test_sequences[0]
    assert len(test_sequence) == len(ref_sequence)
    n_matches = np.count_nonzero(test_sequence.code == ref_sequence.code)
    assert n_matches / len(ref_sequence) >= MIN_PERCENTAGE
    score_matrix = align.SubstitutionMatrix.std_clepaps_matrix().score_matrix()
    for i in np.where(test_sequence.code != ref_sequence.code)[0]:
        assert score_matrix[test_sequence.code[i], ref_sequence.code[i]] >= MIN_SCORE


@pytest.mark.parametrize("pdb_id, chain_id", [("1mol", "A")])
def test_missing_residues(pdb_id, chain_id):
    """
    Like `test_to_clepaps()`, but in some residues the ``CA`` atom is missing.
    Expect that these and adjacent residues get the unknown symbol 'R' in the
    CLePAPS sequence.
    """
    N_DELETIONS = 5
    # The 'R' symbol
    UNKNOWN_SYMBOL = strucalph.ClepapsSequence.unknown_symbol

    chain = _load_chain(pdb_id, chain_id)
    ref_sequence = strucalph.to_clepaps(chain)[0][0]

    # Randomly delete the CA atom of some residues
    rng = np.random.default_rng(1)
    del_residue_ids = rng.choice(np.unique(chain.res_id), N_DELETIONS, replace=False)
    chain = chain[~np.isin(chain.res_id, del_residue_ids) | (chain.atom_name != "CA")]

    # A CLePAPS symbol is defined by four consecutive CA atoms, hence a missing CA
    # turns the symbol of its residue and the three surrounding ones into 'R'
    for res_id in del_residue_ids:
        seq_index = res_id - chain.res_id[0]
        start_index = max(0, seq_index - 1)
        end_index = min(len(ref_sequence), seq_index + 3)
        ref_sequence[start_index:end_index] = UNKNOWN_SYMBOL

    test_sequences, _ = strucalph.to_clepaps(chain)

    assert len(test_sequences) == 1
    assert str(test_sequences[0]) == str(ref_sequence)
