# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


def test_alignment_from_string():
    """
    Test reading alignments from string.
    """
    ali_str = [
        "A-CCTGA----",
        "----T-ATGCT"
    ]  # fmt: skip
    alignment = align.Alignment.from_strings(ali_str, seq.NucleotideSequence)
    assert str(alignment).split("\n") == ali_str


def test_conversion_to_symbols():
    """
    Test conversion of alignments to strings.
    """
    seq_strings = [
        "HAKLPRDD--WKL--",
        "HA--PRDDADWKLHH",
        "HA----DDADWKLHH",
    ]
    alignment = align.Alignment.from_strings(seq_strings, seq.ProteinSequence)
    # Test the conversion bach to strings of symbols
    symbols = align.get_symbols(alignment)
    symbols = [
        "".join([sym if sym is not None else "-" for sym in sym_list])
        for sym_list in symbols
    ]
    assert symbols == seq_strings


def test_identity():
    """
    Test correct calculation of `get_sequence_identity()` via a known
    test case.
    """
    seq_strings = [
        "--HAKLPRDD--WL--",
        "FRHA--QRTDADWLHH",
    ]
    alignment = align.Alignment.from_strings(seq_strings, seq.ProteinSequence)
    # Assert correct sequence identity calculation
    modes = ["all", "not_terminal", "shortest"]
    values = [6 / 16, 6 / 12, 6 / 10]
    for mode, value in zip(modes, values):
        assert align.get_sequence_identity(alignment, mode=mode) == value


@pytest.mark.parametrize("mode", ["all", "not_terminal", "shortest"])
def test_pairwise_identity(sequences, mode):
    """
    Test correct calculation of `get_pairwise_sequence_identity()` via
    pairwise calls of `get_sequence_identity()`.
    """
    sequences = sequences
    msa, _, _, _ = align.align_multiple(
        sequences, matrix=align.SubstitutionMatrix.std_protein_matrix()
    )

    ref_identity_matrix = np.zeros((len(sequences), len(sequences)))
    for i in range(len(sequences)):
        for j in range(len(sequences)):
            ref_identity_matrix[i, j] = align.get_sequence_identity(
                msa[:, [i, j]], mode=mode
            )

    test_identity_matrix = align.get_pairwise_sequence_identity(msa, mode=mode)

    # Identity of two equal sequences should be 1, if only the length of
    # the sequence is counted
    if mode == "shortest":
        assert (np.diag(test_identity_matrix) == 1).all()
    # Identity must be between 0 and 1
    assert ((test_identity_matrix <= 1) & (test_identity_matrix >= 0)).all()
    # Identity matrix is symmetric
    assert (test_identity_matrix == test_identity_matrix.T).all()
    # Pairwise identity must be equal in the two functions
    assert (test_identity_matrix == ref_identity_matrix).all()
