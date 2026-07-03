# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


@pytest.mark.parametrize("gap_penalty", [-10, (-10, -1)])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("band_width", [2, 5, 20, 100])
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


@pytest.mark.parametrize("length", [1_000, 1_000_000])
@pytest.mark.parametrize("excerpt_length", [50, 500])
@pytest.mark.parametrize("seed", range(10))
def test_large_sequence_mapping(length, excerpt_length, seed):
    """
    Test whether an excerpt of a very large sequence is aligned to that
    sequence at the position, where the excerpt was taken from.
    """
    BAND_WIDTH = 100

    rng = np.random.default_rng(seed)

    sequence = seq.NucleotideSequence()
    sequence.code = rng.integers(len(sequence.alphabet), size=length)
    excerpt_pos = rng.integers(len(sequence) - excerpt_length)
    excerpt = sequence[excerpt_pos : excerpt_pos + excerpt_length]

    diagonal = rng.integers(excerpt_pos - BAND_WIDTH, excerpt_pos + BAND_WIDTH)
    band = (diagonal - BAND_WIDTH, diagonal + BAND_WIDTH)

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
