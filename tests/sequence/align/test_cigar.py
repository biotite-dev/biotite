# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import re
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


def _generate_cigar(seed):
    np.random.seed(seed)
    cigar = ""
    n_op_pairs = np.random.choice(range(1, 10))
    for _ in range(n_op_pairs):
        # Alternatingly insert matches and insertions/deletions
        cigar += f"{np.random.randint(1, 100)}M"
        op = align.CigarOp(
            np.random.choice(
                [
                    align.CigarOp.INSERTION,
                    align.CigarOp.DELETION,
                ]
            )
        ).to_cigar_symbol()
        cigar += f"{np.random.randint(1, 100)}{op}"
    # Alignment must end with a match
    cigar += f"{np.random.randint(1, 100)}M"
    # Possibly add a soft clip at the alignment start
    # Soft clip at alignment end would be ignored
    # and thus could not be mapped back
    if np.random.choice([True, False]):
        cigar = f"{np.random.randint(1, 100)}S" + cigar
    return cigar


def _mutate_sequence(
    original, max_subsitutions=50, max_insertions=50, max_deletions=50
):
    """
    Introduce random deletions, insertions and substitutions into a
    sequence.
    """
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

    return mutant


@pytest.mark.parametrize("cigar", [_generate_cigar(i) for i in range(20)])
def test_cigar_conversion(cigar):
    """
    Check whether a CIGAR string converted into an :class:`Alignment`
    and back again into a CIGAR
    string gives the same result.
    """
    LENGTH = 10000
    # The sequences are arbitrary, only the alignment trace matters
    # However, they still need to be long enough for the number of CIGAR
    # operations
    ref = seq.NucleotideSequence(["A"] * LENGTH)
    seg = seq.NucleotideSequence(["A"] * LENGTH)
    alignment = align.read_alignment_from_cigar(cigar, 0, ref, seg)
    print(alignment)

    test_cigar = align.write_alignment_to_cigar(alignment)
    # Remove the soft clip at the end,
    # that appears due to the `seg` sequence length
    test_cigar = re.sub(r"\d+S$", "", test_cigar)

    assert test_cigar == cigar


@pytest.mark.parametrize(
    "seed, local, distinguish_matches, include_terminal_gaps",
    itertools.product(
        range(20),
        [False, True],
        [False, True],
        [False, True],
    ),
)
def test_alignment_conversion(seed, local, distinguish_matches, include_terminal_gaps):
    """
    Check whether an :class:`Alignment` converted into a CIGAR string
    and back again into an :class:`Alignment` gives the same result.
    """
    REF_LENGTH = 1000
    np.random.seed(seed)
    ref = seq.NucleotideSequence(ambiguous=False)
    ref.code = np.random.randint(0, len(ref.alphabet), REF_LENGTH, dtype=np.uint8)

    excerpt_start = np.random.randint(0, 200)
    excerpt_stop = np.random.randint(REF_LENGTH - 200, REF_LENGTH)
    seg = ref[excerpt_start:excerpt_stop]
    seg = _mutate_sequence(seg)

    matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    if local:
        ref_ali = align.align_optimal(ref, seg, matrix, local=True, max_number=1)[0]
    else:
        ref_ali = align.align_optimal(
            ref, seg, matrix, terminal_penalty=False, max_number=1
        )[0]
        if not include_terminal_gaps:
            # Turn into a semi-global alignment
            ref_ali = align.remove_terminal_gaps(ref_ali)
    # Remove score as the compared reconstructed alignment does not
    # contain it either
    ref_ali.score = None
    start_position = ref_ali.trace[0, 0]

    cigar = align.write_alignment_to_cigar(
        ref_ali,
        distinguish_matches=distinguish_matches,
        include_terminal_gaps=include_terminal_gaps,
    )

    test_ali = align.read_alignment_from_cigar(cigar, start_position, ref, seg)

    print(cigar)
    print("\n\n")
    print(ref_ali)
    print("\n\n")
    print(test_ali)
    print("\n\n")
    assert test_ali == ref_ali
