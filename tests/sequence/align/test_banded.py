# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


@pytest.mark.parametrize(
    "gap_penalty, local, band_width",
    itertools.product([-10, (-10, -1)], [False, True], [2, 5, 20, 100]),
)
def test_simple_alignment(gap_penalty, local, band_width):
    """
    Test `align_banded()` by comparing the output to `align_optimal()`.
    This test uses a pair of highly similar short sequences.
    """
    # Cyclotide C, Uniprot: P86843
    seq1 = seq.ProteinSequence("gvpcaescvwipctvtallgcsckdkvcyld")
    # Cyclotide F, Uniprot: P86846
    seq2 = seq.ProteinSequence("gipcgescvfipcissvvgcsckskvcyld")
    matrix = align.SubstitutionMatrix.std_protein_matrix()

    ref_alignments = align.align_optimal(
        seq1, seq2, matrix, gap_penalty=gap_penalty, local=local, terminal_penalty=False
    )
    # Remove terminal gaps in reference to obtain a true semi-global
    # alignment, as returned by align_banded()
    ref_alignments = [align.remove_terminal_gaps(al) for al in ref_alignments]

    test_alignments = align.align_banded(
        seq1,
        seq2,
        matrix,
        (-band_width, band_width),
        gap_penalty=gap_penalty,
        local=local,
    )

    assert len(test_alignments) == len(ref_alignments)
    for alignment in test_alignments:
        assert alignment in ref_alignments


@pytest.mark.parametrize(
    "gap_penalty, local, seq_indices",
    itertools.product(
        [-10, (-10, -1)],
        [False, True],
        [(i, j) for i in range(10) for j in range(i + 1)],
    ),
)
def test_complex_alignment(sequences, gap_penalty, local, seq_indices):
    """
    Test `align_banded()` by comparing the output to `align_optimal()`.
    This test uses a set of long sequences, which are pairwise compared.
    The band should be chosen sufficiently large so `align_banded()`
    can return the optimal alignment(s).
    """
    MAX_NUMBER = 100

    matrix = align.SubstitutionMatrix.std_protein_matrix()
    index1, index2 = seq_indices
    seq1 = sequences[index1]
    seq2 = sequences[index2]

    ref_alignments = align.align_optimal(
        seq1,
        seq2,
        matrix,
        gap_penalty=gap_penalty,
        local=local,
        terminal_penalty=False,
        max_number=MAX_NUMBER,
    )
    # Remove terminal gaps in reference to obtain a true semi-global
    # alignment, as returned by align_banded()
    ref_alignments = [align.remove_terminal_gaps(al) for al in ref_alignments]

    identity = align.get_sequence_identity(ref_alignments[0])
    # Use a relatively small band width, if the sequences are similar,
    # otherwise use the entire search space
    band_width = 100 if identity > 0.5 else len(seq1) + len(seq2)
    test_alignments = align.align_banded(
        seq1,
        seq2,
        matrix,
        (-band_width, band_width),
        gap_penalty=gap_penalty,
        local=local,
        max_number=MAX_NUMBER,
    )

    try:
        assert test_alignments[0].score == ref_alignments[0].score
        if len(ref_alignments) < MAX_NUMBER:
            # Only test if the exact same alignments were created,
            # if the number of traces was not limited by MAX_NUMBER
            assert len(test_alignments) == len(ref_alignments)
            for alignment in test_alignments:
                assert alignment in ref_alignments
    except AssertionError:
        print("First tested alignment:")
        print()
        print(test_alignments[0])
        print("\n")
        print("First reference alignment:")
        print()
        print(ref_alignments[0])
        raise


@pytest.mark.parametrize(
    "length, excerpt_length, seed",
    itertools.product([1_000, 1_000_000], [50, 500], range(10)),
)
def test_large_sequence_mapping(length, excerpt_length, seed):
    """
    Test whether an excerpt of a very large sequence is aligned to that
    sequence at the position, where the excerpt was taken from.
    """
    BAND_WIDTH = 100

    np.random.seed(seed)

    sequence = seq.NucleotideSequence()
    sequence.code = np.random.randint(len(sequence.alphabet), size=length)
    excerpt_pos = np.random.randint(len(sequence) - excerpt_length)
    excerpt = sequence[excerpt_pos : excerpt_pos + excerpt_length]

    diagonal = np.random.randint(excerpt_pos - BAND_WIDTH, excerpt_pos + BAND_WIDTH)
    band = (diagonal - BAND_WIDTH, diagonal + BAND_WIDTH)
    print(band)
    print(len(sequence), len(excerpt))

    matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    test_alignments = align.align_banded(excerpt, sequence, matrix, band=band)
    # The excerpt should be uniquely mappable to a single location on
    # the long sequence
    assert len(test_alignments) == 1
    test_alignment = test_alignments[0]
    test_trace = test_alignment.trace

    ref_trace = np.stack(
        [np.arange(len(excerpt)), np.arange(excerpt_pos, len(excerpt) + excerpt_pos)],
        axis=1,
    )
    assert np.array_equal(test_trace, ref_trace)


@pytest.mark.parametrize(
    "gap_penalty, local, seed",
    itertools.product([-10, (-10, -1)], [False, True], range(100)),
)
def test_swapping(gap_penalty, local, seed):
    """
    Check if `align_banded()` returns a 'swapped' alignment, if
    the order of input sequences is swapped.
    """
    np.random.seed(seed)
    band = (np.random.randint(-30, -10), np.random.randint(10, 30))

    seq1, seq2 = _create_random_pair(seed)
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    ref_alignments = align.align_banded(
        seq1, seq2, matrix, band=band, local=local, gap_penalty=gap_penalty
    )

    test_alignments = align.align_banded(
        seq2, seq1, matrix, band=band, local=local, gap_penalty=gap_penalty
    )

    if len(ref_alignments) != 1 or len(test_alignments) != 1:
        # If multiple optimal alignments exist,
        # it is not easy to assign a swapped one to an original one
        # therefore, simply return in this case
        # the number of tested seeds should be large enough to generate
        # a reasonable number of suitable test cases
        return
    ref_alignment = ref_alignments[0]
    test_alignment = test_alignments[0]

    assert test_alignment.sequences[0] == ref_alignment.sequences[1]
    assert test_alignment.sequences[1] == ref_alignment.sequences[0]
    assert np.array_equal(test_alignment.trace, ref_alignment.trace[:, ::-1])


def _create_random_pair(
    seed,
    length=100,
    max_subsitutions=5,
    max_insertions=5,
    max_deletions=5,
    max_truncations=5,
):
    """
    generate a pair of protein sequences.
    Each pair contains

        1. a randomly generated sequence
        2. a sequence created by randomly introducing
           deletions/insertions/substitutions into the first sequence.
    """
    np.random.seed(seed)

    original = seq.ProteinSequence()
    original.code = np.random.randint(len(original.alphabet), size=length)

    mutant = original.copy()

    # Random Substitutions
    n_subsitutions = np.random.randint(max_subsitutions)
    subsitution_indices = np.random.choice(
        np.arange(len(mutant)), size=n_subsitutions, replace=False
    )
    subsitution_values = np.random.randint(len(original.alphabet), size=n_subsitutions)
    mutant.code[subsitution_indices] = subsitution_values

    # Random insertions
    n_insertions = np.random.randint(max_insertions)
    insertion_indices = np.random.choice(
        np.arange(len(mutant)), size=n_insertions, replace=False
    )
    insertion_values = np.random.randint(len(original.alphabet), size=n_insertions)
    mutant.code = np.insert(mutant.code, insertion_indices, insertion_values)

    # Random deletions
    n_deletions = np.random.randint(max_deletions)
    deletion_indices = np.random.choice(
        np.arange(len(mutant)), size=n_deletions, replace=False
    )
    mutant.code = np.delete(mutant.code, deletion_indices)

    # Truncate at both ends of original and mutant
    original = original[
        np.random.randint(max_truncations) : -(1 + np.random.randint(max_truncations))
    ]
    mutant = mutant[
        np.random.randint(max_truncations) : -(1 + np.random.randint(max_truncations))
    ]

    return original, mutant
