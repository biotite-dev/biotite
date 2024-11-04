# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.structure.alphabet as strucalph


@pytest.mark.parametrize(
    "db_entry",
    [
        entry
        for entry in align.SubstitutionMatrix.list_db()
        if entry not in ["NUC", "GONNET", "3Di", "PB"]
    ],
)
def test_matrices(db_entry):
    """
    Test for exceptions when reading matrix files.
    """
    alph1 = seq.ProteinSequence.alphabet
    alph2 = seq.ProteinSequence.alphabet
    align.SubstitutionMatrix(alph1, alph2, db_entry)


@pytest.mark.parametrize(
    "matrix_name, alphabet",
    [
        ("3Di", strucalph.I3DSequence.alphabet),
    ],
)
def test_structural_alphabet_matrices(matrix_name, alphabet):
    """
    Test for exceptions when reading structural alphabet matrix files.
    """
    align.SubstitutionMatrix(alphabet, alphabet, matrix_name)


@pytest.mark.parametrize(
    "method_name",
    [
        "std_protein_matrix",
        "std_nucleotide_matrix",
        "std_3di_matrix",
        "std_protein_blocks_matrix",
    ],
)
def test_default_matrices(method_name):
    """
    Test for exceptions when using the static methods for getting default matrices.
    """
    matrix = getattr(align.SubstitutionMatrix, method_name)()
    assert isinstance(matrix, align.SubstitutionMatrix)


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
