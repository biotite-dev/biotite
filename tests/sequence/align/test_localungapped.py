# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
from biotite.sequence.seqtypes import ProteinSequence


@pytest.mark.parametrize(
    "seq_type, seq1, seq2, seed, threshold, ref_range1, ref_range2,"
    "direction, score_only, uint8_code",
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
        
        [[False], [True]],  # score_only

        [[False], [True]],  # uint8_code
    )]
)
def test_simple_alignments(seq_type, seq1, seq2, seed, threshold,
                           ref_range1, ref_range2,
                           direction, score_only, uint8_code):
    """
    Check if `algin_local_ungapped()` produces correct alignments based on
    simple known examples.
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
    
    if not uint8_code:
        seq1, seq2, matrix = _convert_to_uint16_code(seq1, seq2, matrix)

    
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
        assert test_result == ref_alignment


@pytest.mark.parametrize(
    "seed, uint8_code", itertools.product(
        range(100),
        [False, True]
    )
)
def test_random_alignment(seed, uint8_code):
    """
    Create two randomized sequences and place a conserved region into
    each sequence, where both conserved regions are similar to each
    other.
    The conserved regions only contain point mutations and no indels.
    Expect that the alignment score found by `align_local_ungapped()` is
    equal to the alignment score found by `align_optimal()`.
    """
    MIN_SIZE = 200
    MAX_SIZE = 1000
    MIN_CONSERVED_SIZE = 20
    MAX_CONSERVED_SIZE = 100
    CONSERVED_ENDS = 5
    MUTATION_PROB = 0.1
    THRESHOLD = 100
    
    np.random.seed(seed)

    # Create conserved regions
    conserved1 = ProteinSequence()
    conserved_len = np.random.randint(MIN_CONSERVED_SIZE, MAX_CONSERVED_SIZE+1)
    conserved1.code = np.random.randint(
        # Do not include stop symbol for aesthetic reasons -> -1
        len(conserved1.alphabet)-1,
        size=conserved_len
    )
    conserved2 = ProteinSequence()
    # The second conserved regions is equal to the first one,
    # except a few point mutations
    conserved2.code = conserved1.code.copy()
    mutation_mask = np.random.choice(
        [False, True],
        size=conserved_len,
        p = [1 - MUTATION_PROB, MUTATION_PROB]
    )
    conserved2.code[mutation_mask] = np.random.randint(
        len(conserved2.alphabet)-1,
        size=np.count_nonzero(mutation_mask)
    )
    # Flank the conserved regions with equal termini to ensure
    # that the alignment extends from start to end of the region
    conserved2.code[:CONSERVED_ENDS] = conserved1.code[:CONSERVED_ENDS]
    conserved2.code[-CONSERVED_ENDS:] = conserved1.code[-CONSERVED_ENDS:]

    # Create randomized sequences
    seq1 = ProteinSequence()
    seq2 = ProteinSequence()
    offset = []
    for sequence, conserved in zip(
        (seq1, seq2), (conserved1, conserved2)
    ):
        sequence.code = np.random.randint(
            len(sequence.alphabet)-1,
            size=np.random.randint(MIN_SIZE, MAX_SIZE+1)
        )
        # Place conserved region randomly within the sequence
        conserved_pos = np.random.randint(0, len(sequence) - len(conserved))
        sequence.code[conserved_pos : conserved_pos + len(conserved)] \
            = conserved.code
        offset.append(conserved_pos)
    # The seed is placed somewhere in the conserved region
    seed = np.array(offset) + np.random.randint(len(conserved))


    matrix = align.SubstitutionMatrix.std_protein_matrix()
    if not uint8_code:
        seq1, seq2, matrix = _convert_to_uint16_code(seq1, seq2, matrix)
    
    ref_score = align.align_optimal(
        seq1, seq2, matrix, local=True, max_number=1,
        # High gap penalty to prevent introduction of gaps, 
        # since 'align_local_ungapped()' is also no able to place gaps
        gap_penalty=-1000
    )[0].score

    test_alignment = align.align_local_ungapped(
        seq1, seq2, matrix, seed, THRESHOLD
    )

    assert test_alignment.score == ref_score
    # Test if the score is also correctly calculated
    assert align.score(test_alignment, matrix) == ref_score


def _convert_to_uint16_code(seq1, seq2, matrix):
        """
        Adjust sequences, so that they use 'uint16' as dtype for the
        code.
        This is a necessary test, since 'uint8' uses a separate
        implementation.
        """
        new_alph = seq.Alphabet(np.arange(500))
        code = seq1.code
        seq1 = seq.GeneralSequence(new_alph)
        seq1.code = code
        code = seq2.code
        seq2 = seq.GeneralSequence(new_alph)
        seq2.code = code
        # Adjust the substitution matrix as well,
        # so that it is compatible with the new alphabet
        score_matrix = np.zeros((len(new_alph), len(new_alph)), dtype=np.int32)
        orig_len = len(matrix.score_matrix())
        score_matrix[:orig_len, :orig_len] = matrix.score_matrix()
        matrix = align.SubstitutionMatrix(new_alph, new_alph, score_matrix)
        return seq1, seq2, matrix