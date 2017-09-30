# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import biopython.sequence as seq
import biopython.sequence.align as align
import numpy as np
import pytest


def test_align_global():
    seq1 = seq.NucleotideSequence("ATACGCTTGCT")
    seq2 = seq.NucleotideSequence("AGGCGCAGCT")
    alignments = align.align_global(seq1, seq2,
                       align.SubstitutionMatrix.std_nucleotide_matrix(),
                       gap_opening=-6,
                       gap_extension=-2)
    alignment = alignments[0]
    assert alignment.trace.tolist() == [[0,0], [1,1], [2,2], [3,3], [4,4],
                                        [5,5], [6,6], [7,-1], [8,7], [9,8],
                                        [10,9]]

def test_align_local():
    seq1 = seq.NucleotideSequence("ATGGCACATGATCTTA")
    seq2 = seq.NucleotideSequence("ACTTGCTTACGAT")
    alignments = align.align_local(seq1, seq2,
                       align.SubstitutionMatrix.std_nucleotide_matrix(),
                       gap_opening=-6,
                       gap_extension=-2)
    alignment = alignments[0]
    assert alignment.trace.tolist() == [[5,0], [6,1], [7,2], [8,3], [9,4],
                                        [10,-1], [11,-1], [12,5], [13,6],
                                        [14,7], [15,8]]

@pytest.mark.parametrize("db_entry", [entry for entry
                                      in align.SubstitutionMatrix.list_db()
                                      if entry not in ["NUC","GONNET"]])
def test_matrices(db_entry):
    alph1 = seq.ProteinSequence.alphabet
    alph2 = seq.ProteinSequence.alphabet
    matrix = align.SubstitutionMatrix(alph1, alph2, db_entry)