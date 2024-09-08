# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


@pytest.mark.parametrize(
    "db_entry",
    [
        entry
        for entry in align.SubstitutionMatrix.list_db()
        if entry not in ["NUC", "GONNET"]
    ],
)
def test_matrices(db_entry):
    """
    Test for exceptions when reading matrix files.
    """
    alph1 = seq.ProteinSequence.alphabet
    alph2 = seq.ProteinSequence.alphabet
    align.SubstitutionMatrix(alph1, alph2, db_entry)


def test_matrix_str():
    """
    Test conversion of substitution matrix to string via a small
    constructed test case.
    """
    alph1 = seq.Alphabet("abc")
    alph2 = seq.Alphabet("def")
    score_matrix = np.arange(9).reshape((3, 3))
    matrix = align.SubstitutionMatrix(alph1, alph2, score_matrix)
    assert str(matrix) == "\n".join(
        ["    d   e   f",
         "a   0   1   2",
         "b   3   4   5",
         "c   6   7   8"]
    )  # fmt: skip


@pytest.mark.parametrize("seed", range(10))
def test_as_positional(seed):
    """
    Check if the score from a positional substitution matrix for symbols from
    positional sequences is equal to the score from the original matrix for the
    corresponding symbols from the original sequences.
    """
    MAX_SEQ_LENGTH = 10

    np.random.seed(seed)
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    sequences = []
    for _ in range(2):
        seq_length = np.random.randint(1, MAX_SEQ_LENGTH)
        sequence = seq.ProteinSequence()
        sequence.code = np.random.randint(0, len(sequence.alphabet), size=seq_length)
        sequences.append(sequence)
    pos_matrix, *pos_sequences = matrix.as_positional(*sequences)

    # Sanity check if the positional sequences do correspond to the original sequences
    for i in range(2):
        assert str(pos_sequences[i]) == str(sequences[i])
    for i in range(len(sequences[0])):
        for j in range(len(sequences[1])):
            ref_score = matrix.get_score(sequences[0][i], sequences[1][j])
            test_score = pos_matrix.get_score(pos_sequences[0][i], pos_sequences[1][j])
            assert test_score == ref_score
