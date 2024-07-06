# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


@pytest.mark.parametrize("db_entry", [entry for entry
                                      in align.SubstitutionMatrix.list_db()
                                      if entry not in ["NUC","GONNET"]])
def test_matrices(db_entry):
    """
    Test for exceptions when reading matrix files.
    """
    alph1 = seq.ProteinSequence.alphabet
    alph2 = seq.ProteinSequence.alphabet
    matrix = align.SubstitutionMatrix(alph1, alph2, db_entry)

def test_matrix_str():
    """
    Test conversion of substitution matrix to string via a small
    constructed test case.
    """
    alph1 = seq.Alphabet("abc")
    alph2 = seq.Alphabet("def")
    score_matrix = np.arange(9).reshape((3,3))
    matrix = align.SubstitutionMatrix(alph1, alph2, score_matrix)
    assert str(matrix) == "\n".join(
        ["    d   e   f",
         "a   0   1   2",
         "b   3   4   5",
         "c   6   7   8"]
    ) # fmt: skip