# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


@pytest.mark.parametrize(
    "seq_type, seq1, seq2, seed, threshold, ref_range1, ref_range2,"
    "direction, score_only",
    [list(itertools.chain(*e)) for e in itertools.product(
        [
            (
                seq.NucleotideSequence,  # seq_type
                "TTTAAAAAAATTTTT",       # seq1
                "CCCCCAAAAAAACCC",       # seq1
                (5, 7),                  # seed
                0,                       # threshold
                (3, 10),                 # ref_range1
                (5, 12),                 # ref_range2
            ),
            (
                seq.NucleotideSequence,
                "TTTAAAAAAATTTTT",
                "CCCCCAAAAAAACCC",
                (3, 5),
                100,
                (3, 10),
                (5, 12),
            ),
            # Mismatch should lead to alignment termination
            (
                seq.ProteinSequence,
                "NLYIQWLKDGGPSSGRPPPS",
                  "YIQWLKDGGPSLGRPPPS",
                (8, 6),
                0,
                (2, 13),
                (0, 11),
            ),
            # Mismatch should not terminate alignment
            (
                seq.ProteinSequence,
                "NLYIQWLKDGGPSSGRPPPS",
                  "YIQWLKDGGPSLGRPPPS",
                (8, 6),
                2,
                (2, 20),
                (0, 18),
            ),
            # Mismatch should terminate alignment
            (
                seq.ProteinSequence,
                "NLYIQWLKDGGPSSGRPPPS",
                  "YIQWLKDWWPSSGRPPPS",
                (6, 4),
                3,
                (2, 9),
                (0, 7),
            ),
        ],

        [["both"], ["upstream"], ["downstream"]],  # direction
        
        [[False], [True]]  # score_only
    )]
)
def test_algin_local_ungapped(seq_type, seq1, seq2, seed, threshold,
                              ref_range1, ref_range2,
                              direction, score_only):
    """
    Check if `algin_local_ungapped()` produces correct alignments based on
    known examples.
    """
    # Limit start or stop reference alignment range to seed
    # if the alignment does not extend in both directions
    if direction == "upstream":
        ref_range1 = (ref_range1[0], seed[0] + 1)
        ref_range2 = (ref_range2[0], seed[1] + 1)
    elif direction == "downstream":
        ref_range1 = (seed[0], ref_range1[1])
        ref_range2 = (seed[1], ref_range2[1])


    seq1 = seq_type(seq1)
    seq2 = seq_type(seq2)
    
    if seq_type == seq.NucleotideSequence:
        matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    else:
        matrix = align.SubstitutionMatrix.std_protein_matrix()
    
    ref_alignment = align.Alignment(
        [seq1, seq2],
        np.stack([
            np.arange(*ref_range1),
            np.arange(*ref_range2)
        ], axis=-1)
    )
    ref_score = align.score(ref_alignment, matrix)
    ref_alignment.score = ref_score

    test_result = align.align_local_ungapped(
        seq1, seq2, matrix, seed, threshold, direction, score_only)
    
    if score_only:
        assert test_result == ref_score
    else:
        print(test_result.trace)
        print()
        print(test_result.trace)
        assert test_result == ref_alignment
