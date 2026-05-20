# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"

import numbers
import textwrap
from collections.abc import Callable, Iterator
from collections.abc import Sequence as SequenceABC
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from biotite.sequence.alphabet import Alphabet
from biotite.sequence.sequence import Sequence
from biotite.typing import K, M, N, NDArray2

if TYPE_CHECKING:
    from biotite.sequence.align.matrix import SubstitutionMatrix

__all__ = [
    "Alignment",
    "get_codes",
    "get_symbols",
    "get_sequence_identity",
    "get_pairwise_sequence_identity",
    "score",
    "find_terminal_gaps",
    "remove_terminal_gaps",
    "remove_gaps",
]


class Alignment(object):
    """
    An :class:`Alignment` object stores information about which symbols
    of *n* sequences are aligned to each other and it stores the
    corresponding alignment score.

    Instead of saving a list of aligned symbols, this class saves the
    original *n* sequences, that were aligned, and a so called *trace*,
    which indicate the aligned symbols of these sequences.
    The trace is a *(m x n)* :class:`ndarray` with alignment length
    *m* and sequence count *n*.
    Each element of the trace is the index in the corresponding
    sequence.
    A gap is represented by the value -1.

    Furthermore this class provides multiple utility functions for
    conversion into strings in order to make the alignment human
    readable.

    Unless an :class:`Alignment` object is the result of a multiple
    sequence alignment, the object will contain only two sequences.

    All attributes of this class are publicly accessible.

    Parameters
    ----------
    sequences : list
        A list of aligned sequences.
    trace : ndarray, dtype=int, shape=(n,m)
        The alignment trace.
    score : int, optional
        Alignment score.

    Attributes
    ----------
    sequences : list
        A list of aligned sequences.
    trace : ndarray, dtype=int, shape=(n,m)
        The alignment trace.
    score : int
        Alignment score.

    Examples
    --------

    >>> seq1 = NucleotideSequence("CGTCAT")
    >>> seq2 = NucleotideSequence("TCATGC")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> ali = align_optimal(seq1, seq2, matrix)[0]
    >>> print(ali)
    CGTCAT--
    --TCATGC
    >>> print(ali.trace)
    [[ 0 -1]
     [ 1 -1]
     [ 2  0]
     [ 3  1]
     [ 4  2]
     [ 5  3]
     [-1  4]
     [-1  5]]
    >>> print(ali[1:4].trace)
    [[ 1 -1]
     [ 2  0]
     [ 3  1]]
    >>> print(ali[1:4, 0:1].trace)
    [[1]
     [2]
     [3]]
    """

    def __init__(
        self,
        sequences: list[Sequence],
        trace: NDArray2[K, N, np.integer],
        score: int | None = None,
    ) -> None:
        self.sequences = sequences.copy()
        self.trace = trace
        self.score = score

    @staticmethod
    def from_strings(
        sequence_strings: SequenceABC[str],
        sequence_factory: Callable[[str], Sequence],
        gap_character: str = "-",
    ) -> Alignment:
        """
        Create an :class:`Alignment` from strings that represent aligned sequences.

        **DEPRECATED**: Use :meth:`Alignment.from_strings()` instead.

        Parameters
        ----------
        sequence_strings : list of str
            The strings, where each each one represents a sequence
            (with gaps) in an alignment.
            All strings must have the same length.
        sequence_factory : Callable (str -> Sequence)
            Callable that takes a sequence string (with gaps already removed) and
            produces a :class:`Sequence` object.
        gap_character : str, optional
            This character is interpreted as gap.

        Returns
        -------
        alignment : Alignment
            The created alignment.

        Examples
        --------

        >>> alignment = Alignment.from_strings(
        ...     [
        ...         "BIQTITE",
        ...         "-IQLITE"
        ...     ],
        ...     ProteinSequence,
        ... )
        >>> print(alignment)
        BIQTITE
        -IQLITE
        >>> print(alignment.sequences[0])
        BIQTITE
        >>> print(alignment.sequences[1])
        IQLITE
        """
        sequences = [
            sequence_factory(seq_str.replace(gap_character, ""))
            for seq_str in sequence_strings
        ]
        return Alignment(
            sequences, Alignment.trace_from_strings(sequence_strings, gap_character)
        )

    @staticmethod
    def trace_from_strings(
        sequence_strings: SequenceABC[str], gap_character: str = "-"
    ) -> NDArray2[K, N, np.integer]:
        """
        Create a trace from strings that represent aligned sequences.

        Parameters
        ----------
        sequence_strings : list of str
            The strings, where each each one represents a sequence
            (with gaps) in an alignment.
        gap_character : str, optional
            This character is interpreted as gap.

        Returns
        -------
        trace : ndarray, dtype=int, shape=(n,2)
            The created trace.

        See Also
        --------
        from_strings:
            Creates directly an :class:`Alignment` object.
        """
        if len(sequence_strings) < 2:
            raise ValueError("An alignment must contain at least two sequences")
        if any(
            len(seq_str) != len(sequence_strings[0]) for seq_str in sequence_strings
        ):
            raise IndexError("All sequence strings must have the same length")

        # Start with a trace filled with gaps (-1)
        trace = np.full(
            (len(sequence_strings[0]), len(sequence_strings)), -1, dtype=int
        )
        for i, seq_str in enumerate(sequence_strings):
            # Convert into NumPy byte array to use vectorized operations
            byte_array = np.frombuffer(seq_str.encode("ASCII"), dtype=np.ubyte)
            # Fill the trace with the positions of each sequence where there is no gap
            not_gap_mask = byte_array != ord(gap_character)
            trace[not_gap_mask, i] = np.arange(np.count_nonzero(not_gap_mask))

        return trace  # pyright: ignore[reportReturnType]

    def __repr__(self) -> str:
        """Represent Alignment a string for debugging."""
        return (
            f"Alignment([{', '.join([seq.__repr__() for seq in self.sequences])}], "
            f"np.{np.array_repr(self.trace)}, score={self.score})"
        )

    def _gapped_str(self, seq_index: int) -> str:
        seq_str = ""
        for i in range(len(self.trace)):
            j = self.trace[i][seq_index]
            if j != -1:
                seq_str += str(self.sequences[seq_index][j])
            else:
                seq_str += "-"
        return seq_str

    def get_gapped_sequences(self) -> list[str]:
        """
        Get a the string representation of the gapped sequences.

        Returns
        -------
        sequences : list of str
            The list of gapped sequence strings. The order is the same
            as in `Alignment.sequences`.
        """
        return [self._gapped_str(i) for i in range(len(self.sequences))]

    def __str__(self) -> str:
        # Check if any of the sequences
        # has an non-single letter alphabet
        all_single_letter = True
        for seq in self.sequences:
            if not _is_single_letter(seq.alphabet):
                all_single_letter = False
        if all_single_letter:
            # First dimension: sequence number,
            # second dimension: line number
            seq_str_lines_list = []
            wrapper = textwrap.TextWrapper(break_on_hyphens=False)
            for i in range(len(self.sequences)):
                seq_str_lines_list.append(wrapper.wrap(self._gapped_str(i)))
            ali_str = ""
            for row_i in range(len(seq_str_lines_list[0])):
                for seq_j in range(len(seq_str_lines_list)):
                    ali_str += seq_str_lines_list[seq_j][row_i] + "\n"
                ali_str += "\n"
            # Remove final line breaks
            return ali_str[:-2]
        else:
            return super().__str__()

    def __getitem__(self, index: Any) -> Alignment:
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("Only 2D indices are allowed")
            column_index, seq_index = index
            if isinstance(column_index, numbers.Integral) or isinstance(
                seq_index, numbers.Integral
            ):
                raise IndexError(
                    "Integers are invalid indices for alignments, "
                    "a single sequence or alignment column cannot be "
                    "selected"
                )
            return Alignment(
                Alignment._index_sequences(self.sequences, seq_index),
                self.trace[index],
                self.score,
            )
        else:
            return Alignment(self.sequences, self.trace[index], self.score)

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError("'Alignment' object is not iterable")

    def __len__(self) -> int:
        return len(self.trace)

    def __eq__(self, item: object) -> bool:
        if not isinstance(item, Alignment):
            return False
        if self.sequences != item.sequences:
            return False
        if not np.array_equal(self.trace, item.trace):
            return False
        if self.score != item.score:
            return False
        return True

    @staticmethod
    def _index_sequences(sequences: list[Sequence], index: Any) -> list[Sequence]:
        if isinstance(index, (list, tuple)) or (
            isinstance(index, np.ndarray) and index.dtype != bool
        ):
            return [sequences[i] for i in index]
        elif isinstance(index, np.ndarray) and index.dtype == bool:
            return [seq for seq, mask in zip(sequences, index) if mask]
        if isinstance(index, slice):
            return sequences[index]
        else:
            raise IndexError(f"Invalid alignment index type '{type(index).__name__}'")


def get_codes(alignment: Alignment) -> NDArray2[N, M, np.integer]:
    """
    Get the sequence codes of the sequences in the alignment.

    The codes are built from the trace:
    Instead of the indices of the aligned symbols (trace), the return
    value contains the corresponding symbol codes for each index.
    Gaps are still represented by *-1*.

    Parameters
    ----------
    alignment : Alignment
        The alignment to get the sequence codes for.

    Returns
    -------
    codes : ndarray, dtype=int, shape=(n,m)
        The sequence codes for the alignment.
        The shape is *(n,m)* for *n* sequences and *m* alignment cloumn.
        The array uses *-1* values for gaps.

    Examples
    --------

    >>> seq1 = NucleotideSequence("CGTCAT")
    >>> seq2 = NucleotideSequence("TCATGC")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> ali = align_optimal(seq1, seq2, matrix)[0]
    >>> print(ali)
    CGTCAT--
    --TCATGC
    >>> print(get_codes(ali))
    [[ 1  2  3  1  0  3 -1 -1]
     [-1 -1  3  1  0  3  2  1]]
    """
    trace = alignment.trace
    sequences = alignment.sequences

    # The number of sequences is the first dimension
    codes = np.zeros((trace.shape[1], trace.shape[0]), dtype=np.int64)
    for i in range(len(sequences)):
        # Mark -1 explicitly as int64 to avoid that the unsigned dtype
        # of the sequence code is used
        # (https://numpy.org/neps/nep-0050-scalar-promotion.html)
        codes[i] = np.where(
            trace[:, i] != -1, sequences[i].code[trace[:, i]], np.int64(-1)
        )

    return codes  # pyright: ignore[reportReturnType]


def get_symbols(alignment: Alignment) -> list[list[Any]]:
    """
    Similar to :func:`get_codes()`, but contains the decoded symbols
    instead of codes.
    Gaps are still represented by *None* values.

    Parameters
    ----------
    alignment : Alignment
        The alignment to get the symbols for.

    Returns
    -------
    symbols : list of list
        The nested list of symbols.

    See Also
    --------
    get_codes : Get the sequence codes of the sequences in the alignment.

    Examples
    --------

    >>> seq1 = NucleotideSequence("CGTCAT")
    >>> seq2 = NucleotideSequence("TCATGC")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> ali = align_optimal(seq1, seq2, matrix)[0]
    >>> print(ali)
    CGTCAT--
    --TCATGC
    >>> print(get_symbols(ali))
    [['C', 'G', 'T', 'C', 'A', 'T', None, None], [None, None, 'T', 'C', 'A', 'T', 'G', 'C']]
    """
    codes = get_codes(alignment)
    symbols = [None] * codes.shape[0]
    for i in range(codes.shape[0]):
        alphabet = alignment.sequences[i].get_alphabet()
        codes_wo_gaps = codes[i, codes[i] != -1]
        symbols_wo_gaps = alphabet.decode_multiple(codes_wo_gaps)
        if isinstance(symbols_wo_gaps, np.ndarray):
            symbols_wo_gaps = symbols_wo_gaps.tolist()
        symbols_for_seq = np.full(len(codes[i]), None, dtype=object)
        symbols_for_seq[codes[i] != -1] = symbols_wo_gaps
        symbols[i] = symbols_for_seq.tolist()
    return symbols


def get_sequence_identity(
    alignment: Alignment,
    mode: Literal["all", "not_terminal", "shortest"] = "not_terminal",
) -> float:
    """
    Calculate the sequence identity for an alignment.

    The identity is equal to the matches divided by a measure for the
    length of the alignment that depends on the `mode` parameter.

    Parameters
    ----------
    alignment : Alignment
        The alignment to calculate the identity for.
    mode : {'all', 'not_terminal', 'shortest'}, optional
        The calculation mode for alignment length.

            - **all** - The number of matches divided by the number of
              all alignment columns.
            - **not_terminal** - The number of matches divided by the
              number of alignment columns that are not terminal gaps in
              any of the sequences.
            - **shortest** - The number of matches divided by the
              length of the shortest sequence.

        Default is *not_terminal*.

    Returns
    -------
    identity : float
        The sequence identity, ranging between 0 and 1.

    See Also
    --------
    get_pairwise_sequence_identity : Get sequence identity for each pair of alignment rows.
    """
    codes = get_codes(alignment)

    # Count matches
    matches = 0
    for i in range(codes.shape[1]):
        column = codes[:, i]
        # One unique value -> all symbols match
        unique_symbols = np.unique(column)
        if len(unique_symbols) == 1 and unique_symbols[0] != -1:
            matches += 1

    # Calculate length
    if mode == "all":
        length = len(alignment)
    elif mode == "not_terminal":
        start, stop = find_terminal_gaps(alignment)
        if stop <= start:
            raise ValueError(
                "Cannot calculate non-terminal identity, "
                "at least two sequences have no overlap"
            )
        length = stop - start
    elif mode == "shortest":
        length = min([len(seq) for seq in alignment.sequences])
    else:
        raise ValueError(f"'{mode}' is an invalid calculation mode")

    return matches / length


def get_pairwise_sequence_identity(
    alignment: Alignment,
    mode: Literal["all", "not_terminal", "shortest"] = "not_terminal",
) -> NDArray2[N, N, np.floating]:
    """
    Calculate the pairwise sequence identity for an alignment.

    The identity is equal to the matches divided by a measure for the
    length of the alignment that depends on the `mode` parameter.

    Parameters
    ----------
    alignment : Alignment, length=n
        The alignment to calculate the pairwise sequence identity for.
    mode : {'all', 'not_terminal', 'shortest'}, optional
        The calculation mode for alignment length.

            - **all** - The number of matches divided by the number of
              all alignment columns.
            - **not_terminal** - The number of matches divided by the
              number of alignment columns that are not terminal gaps in
              any of the two considered sequences.
            - **shortest** - The number of matches divided by the
              length of the shortest one of the two sequences.

        Default is *not_terminal*.

    Returns
    -------
    identity : ndarray, dtype=float, shape=(n,n)
        The pairwise sequence identity, ranging between 0 and 1.

    See Also
    --------
    get_sequence_identity : Get sequence identity over all alignment rows.
    """
    codes = get_codes(alignment)
    n_seq = len(codes)

    # Count matches
    # Calculate at which positions the sequences are identical
    # and are not gaps
    equality_matrix = (
        (codes[:, np.newaxis, :] == codes[np.newaxis, :, :])
        & (codes[:, np.newaxis, :] != -1)
        & (codes[np.newaxis, :, :] != -1)
    )
    # Sum these positions up
    matches = np.count_nonzero(equality_matrix, axis=-1)

    # Calculate length
    if mode == "all":
        length = len(alignment)
    elif mode == "not_terminal":
        length = np.zeros((n_seq, n_seq))
        for i in range(n_seq):
            for j in range(n_seq):
                # Find latest start and earliest stop of all sequences
                start, stop = find_terminal_gaps(alignment[:, [i, j]])
                if stop <= start:
                    raise ValueError(
                        "Cannot calculate non-terminal identity, "
                        "as the two sequences have no overlap"
                    )
                length[i, j] = stop - start
    elif mode == "shortest":
        length = np.zeros((n_seq, n_seq))
        for i in range(n_seq):
            for j in range(n_seq):
                length[i, j] = min(
                    [len(alignment.sequences[i]), len(alignment.sequences[j])]
                )
    else:
        raise ValueError(f"'{mode}' is an invalid calculation mode")

    return matches / length


def score(
    alignment: Alignment,
    matrix: SubstitutionMatrix,
    gap_penalty: int | tuple[int, int] = -10,
    terminal_penalty: bool = True,
) -> int:
    """
    Calculate the similarity score of an alignment.

    If the alignment contains more than two sequences,
    all pairwise scores are counted.

    Parameters
    ----------
    alignment : Alignment
        The alignment to calculate the identity for.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
    gap_penalty : int or (tuple, dtype=int), optional
        If an integer is provided, the value will be interpreted as
        general gap penalty. If a tuple is provided, an affine gap
        penalty is used. The first integer in the tuple is the gap
        opening penalty, the second integer is the gap extension
        penalty.
        The values need to be negative.
    terminal_penalty : bool, optional
        If true, gap penalties are applied to terminal gaps.

    Returns
    -------
    score : int
        The similarity score.
    """
    codes = get_codes(alignment)
    score_matrix = matrix.score_matrix()

    # Sum similarity scores (without gaps)
    score: int = 0
    # Iterate over all positions
    for pos in range(codes.shape[1]):
        column = codes[:, pos]
        # Iterate over all possible pairs
        # Do not count self-similarity
        # and do not count similarity twice (not S(i,j) and S(j,i))
        for i in range(codes.shape[0]):
            for j in range(i + 1, codes.shape[0]):
                code_i = column[i]
                code_j = column[j]
                # Ignore gaps
                if code_i != -1 and code_j != -1:
                    score += int(score_matrix[code_i, code_j])

    # Sum gap penalties
    gap_open: int
    gap_ext: int
    if isinstance(gap_penalty, SequenceABC):
        gap_open = gap_penalty[0]
        gap_ext = gap_penalty[1]
    elif isinstance(gap_penalty, numbers.Integral):
        gap_open = int(gap_penalty)
        gap_ext = int(gap_penalty)
    else:
        raise TypeError("Gap penalty must be either integer or tuple")
    # Iterate over all sequences
    for seq_code in codes:
        in_gap = False
        if terminal_penalty:
            start_index = 0
            stop_index = len(seq_code)
        else:
            # Find a start and stop index excluding terminal gaps
            start_index, stop_index = find_terminal_gaps(alignment)
        for i in range(start_index, stop_index):
            if seq_code[i] == -1:
                if in_gap:
                    score += gap_ext
                else:
                    score += gap_open
                in_gap = True
            else:
                in_gap = False
    return score


def find_terminal_gaps(alignment: Alignment) -> tuple[int, int]:
    """
    Find the slice indices that would remove terminal gaps from an
    alignment.

    Terminal gaps are gaps that appear before all sequences start and
    after any sequence ends.

    Parameters
    ----------
    alignment : Alignment
        The alignment, where the slice indices should be found in.

    Returns
    -------
    start, stop : int
        Indices that point to the start and exclusive stop of the
        alignment columns without terminal gaps.
        When these indices are used as slice index for an alignment or
        trace, the index would remove terminal gaps.

    See Also
    --------
    remove_terminal_gaps : Remove terminal gap columns directly.

    Examples
    --------

    >>> sequences = [
    ...     NucleotideSequence(seq_string) for seq_string in (
    ...         "AAAAACTGATTC",
    ...         "AAACTGTTCA",
    ...         "CTGATTCAAA"
    ...     )
    ... ]
    >>> trace = np.transpose([
    ...     ( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, -1, -1, -1),
    ...     (-1, -1,  0,  1,  2,  3,  4,  5, -1,  6,  7,  8,  9, -1, -1),
    ...     (-1, -1, -1, -1, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9),
    ... ])
    >>> alignment = Alignment(sequences, trace)
    >>> print(alignment)
    AAAAACTGATTC---
    --AAACTG-TTCA--
    -----CTGATTCAAA
    >>> print(find_terminal_gaps(alignment))
    (5, 12)
    """
    trace = alignment.trace
    # Find for each sequence the positions of non-gap symbols
    no_gap_pos = [np.where(trace[:, i] != -1)[0] for i in range(trace.shape[1])]
    # Find for each sequence the positions of the sequence start and end
    # in the alignment
    firsts = [no_gap_pos[i][0] for i in range(trace.shape[1])]
    lasts = [no_gap_pos[i][-1] for i in range(trace.shape[1])]
    # The terminal gaps are before all sequences start and after any
    # sequence ends
    # Use exclusive stop -> -1
    return np.max(firsts).item(), np.min(lasts).item() + 1


def remove_terminal_gaps(alignment: Alignment) -> Alignment:
    """
    Remove terminal gaps from an alignment.

    Terminal gaps are gaps that appear before all sequences start and
    after any sequence ends.

    Parameters
    ----------
    alignment : Alignment
        The alignment, where the terminal gaps should be removed from.

    Returns
    -------
    truncated_alignment : Alignment
        A shallow copy of the input `alignment` with an truncated trace,
        that does not contain alignment columns with terminal gaps.

    See Also
    --------
    find_terminal_gaps : Only find terminal gap columns.

    Examples
    --------

    >>> sequences = [
    ...     NucleotideSequence(seq_string) for seq_string in (
    ...         "AAAAACTGATTC",
    ...         "AAACTGTTCA",
    ...         "CTGATTCAAA"
    ...     )
    ... ]
    >>> trace = np.transpose([
    ...     ( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, -1, -1, -1),
    ...     (-1, -1,  0,  1,  2,  3,  4,  5, -1,  6,  7,  8,  9, -1, -1),
    ...     (-1, -1, -1, -1, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9),
    ... ])
    >>> alignment = Alignment(sequences, trace)
    >>> print(alignment)
    AAAAACTGATTC---
    --AAACTG-TTCA--
    -----CTGATTCAAA
    >>> truncated_alignment = remove_terminal_gaps(alignment)
    >>> print(truncated_alignment)
    CTGATTC
    CTG-TTC
    CTGATTC
    """
    start, stop = find_terminal_gaps(alignment)
    if stop < start:
        raise ValueError(
            "Cannot remove terminal gaps, since at least two sequences have "
            "no overlap and the resulting alignment would be empty"
        )
    return alignment[start:stop]


def remove_gaps(alignment: Alignment) -> Alignment:
    """
    Remove all gap columns from an alignment.

    Parameters
    ----------
    alignment : Alignment
        The alignment to be modified.

    Returns
    -------
    truncated_alignment : Alignment
        The alignment without gap columns.

    See Also
    --------
    remove_terminal_gaps : Remove only terminal gap columns.
    """
    non_gap_mask = (alignment.trace != -1).all(axis=1)
    return alignment[non_gap_mask]


def _is_single_letter(alphabet: Alphabet) -> bool:
    """
    More relaxed version of :func:`biotite.sequence.alphabet.is_letter_alphabet()`:
    It is sufficient that only only the string representation of each symbol is only
    a single character.
    """
    if alphabet.is_letter_alphabet():
        return True
    for symbol in alphabet:
        if len(str(symbol)) != 1:
            return False
    return True
