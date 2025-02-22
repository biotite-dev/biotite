# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["CigarOp", "read_alignment_from_cigar", "write_alignment_to_cigar"]

import enum
import numpy as np
from biotite.sequence.align.alignment import Alignment, get_codes


class CigarOp(enum.IntEnum):
    """
    An enum for the different CIGAR operations.
    """

    MATCH = 0
    INSERTION = 1
    DELETION = 2
    INTRON = 3
    SOFT_CLIP = 4
    HARD_CLIP = 5
    PADDING = 6
    EQUAL = 7
    DIFFERENT = 8
    BACK = 9

    @staticmethod
    def from_cigar_symbol(symbol):
        """
        Get the enum value from the CIGAR symbol.

        Parameters
        ----------
        symbol : str
            The CIGAR symbol.

        Returns
        -------
        op : CigarOp
            The enum value.
        """
        return _str_to_op[symbol]

    def to_cigar_symbol(self):
        return _op_to_str[self]


_str_to_op = {
    "M": CigarOp.MATCH,
    "I": CigarOp.INSERTION,
    "D": CigarOp.DELETION,
    "N": CigarOp.INTRON,
    "S": CigarOp.SOFT_CLIP,
    "H": CigarOp.HARD_CLIP,
    "P": CigarOp.PADDING,
    "=": CigarOp.EQUAL,
    "X": CigarOp.DIFFERENT,
    "B": CigarOp.BACK,
}
_op_to_str = {v: k for k, v in _str_to_op.items()}


def read_alignment_from_cigar(cigar, position, reference_sequence, segment_sequence):
    """
    Create an :class:`Alignment` from a CIGAR string.

    Parameters
    ----------
    cigar : str
        The CIGAR string.
    position : int
        0-based position of the first aligned base in the reference.
        0-based equivalent to the ``POS`` field in the SAM/BAM file.
    reference_sequence : Sequence
        The reference sequence.
    segment_sequence : Sequence
        The segment, read or query sequence.

    Returns
    -------
    alignment : Alignment
        The alignment.

    See Also
    --------
    write_alignment_to_cigar : The reverse operation.

    Notes
    -----
    This function expects that the `segment_sequence` was taken from the
    SAM/BAM file, hence hard-clipped bases are not part of the sequence.
    Therefore, hard clipped bases are simply ignored in the CIGAR
    string.

    Examples
    --------

    >>> ref = NucleotideSequence("TATAAAAGGTTTCCGACCGTAGGTAGCTGA")
    >>> seg = NucleotideSequence("CCCCGGTTTGACCGTATGTAG")
    >>> print(read_alignment_from_cigar("9M2D12M", 3, ref, seg))
    AAAAGGTTTCCGACCGTAGGTAG
    CCCCGGTTT--GACCGTATGTAG
    >>> print(read_alignment_from_cigar("4X5=2D7=1X4=", 3, ref, seg))
    AAAAGGTTTCCGACCGTAGGTAG
    CCCCGGTTT--GACCGTATGTAG

    Explicit terminal deletions are also possible.
    Note that in this case the deleted positions count as aligned bases
    with respect to the `position` parameter.

    >>> print(read_alignment_from_cigar("3D9M2D12M4D", 0, ref, seg))
    TATAAAAGGTTTCCGACCGTAGGTAGCTGA
    ---CCCCGGTTT--GACCGTATGTAG----

    If bases in the segment sequence are soft-clipped, they do not
    appear in the alignment.
    Furthermore, the start of the reference sequence must be adapted.

    >>> print(read_alignment_from_cigar("4S5M2D12M", 7, ref, seg))
    GGTTTCCGACCGTAGGTAG
    GGTTT--GACCGTATGTAG

    Hard-clipped bases are not part of the segment sequence.
    Hence `H` operations are completely ignored.

    >>> seg = NucleotideSequence("GGTTTGACCGTATGTAG")
    >>> print(read_alignment_from_cigar("4H5M2D12M", 7, ref, seg))
    GGTTTCCGACCGTAGGTAG
    GGTTT--GACCGTATGTAG

    Reading from BAM codes is also possible.

    >>> seg = NucleotideSequence("CCCCGGTTTGACCGTATGTAG")
    >>> op_tuples = [
    ...     (CigarOp.MATCH, 9),
    ...     (CigarOp.DELETION, 2),
    ...     (CigarOp.MATCH, 12)
    ... ]
    >>> print(read_alignment_from_cigar(op_tuples, 3, ref, seg))
    AAAAGGTTTCCGACCGTAGGTAG
    CCCCGGTTT--GACCGTATGTAG
    """
    if isinstance(cigar, str):
        operations = _op_tuples_from_cigar(cigar)
    else:
        operations = np.asarray(cigar, dtype=int)
        if operations.ndim != 2:
            raise ValueError("Expected array with shape (n,2)")
        if operations.shape[1] != 2:
            raise ValueError("Expected (operation, length) pairs")

    if len(operations) == 0:
        return Alignment(
            [reference_sequence, segment_sequence], np.zeros((0, 2), dtype=int)
        )

    trace = np.zeros((np.sum(operations[:, 1]), 2), dtype=int)
    clip_mask = np.ones(trace.shape[0], dtype=bool)

    i = 0
    ref_pos = position
    seg_pos = 0
    for op, length in operations:
        op = CigarOp(op)
        if op in (CigarOp.MATCH, CigarOp.EQUAL, CigarOp.DIFFERENT):
            trace[i : i + length, 0] = np.arange(ref_pos, ref_pos + length)
            trace[i : i + length, 1] = np.arange(seg_pos, seg_pos + length)
            ref_pos += length
            seg_pos += length
        elif op == CigarOp.INSERTION:
            trace[i : i + length, 0] = -1
            trace[i : i + length, 1] = np.arange(seg_pos, seg_pos + length)
            seg_pos += length
        elif op in (CigarOp.DELETION, CigarOp.INTRON):
            trace[i : i + length, 0] = np.arange(ref_pos, ref_pos + length)
            trace[i : i + length, 1] = -1
            ref_pos += length
        elif op == CigarOp.SOFT_CLIP:
            clip_mask[i : i + length] = False
            seg_pos += length
        elif op == CigarOp.HARD_CLIP:
            clip_mask[i : i + length] = False
        else:
            raise ValueError(f"CIGAR operation {op} is not implemented")
        i += length
    # Remove clipped positions
    trace = trace[clip_mask]
    return Alignment([reference_sequence, segment_sequence], trace)


def write_alignment_to_cigar(
    alignment,
    reference_index=0,
    segment_index=1,
    introns=(),
    distinguish_matches=False,
    hard_clip=False,
    include_terminal_gaps=False,
    as_string=True,
):
    """
    Convert an :class:`Alignment` into a CIGAR string.

    Parameters
    ----------
    alignment : Alignment
        The alignment to be converted.
    reference_index : int, optional
        The index of the reference sequence in the alignment.
        By default the first sequence is used.
    segment_index : int, optional
        The index of the segment, read or query sequence in the
        alignment.
        By default the second sequence is used.
    introns : iterable object of tuple(int, int), optional
        The introns in the reference sequence.
        The introns are given as tuples of start and exclusive stop
        index.
        In those regions gaps in the reference sequence are reflected by
        `'N'` in the CIGAR string.
        By default no introns are assumed.
    distinguish_matches : bool, optional
        If true, matches (`'='`) are distinguished from mismatches
        (`'X'`).
        Otherwise, matches and mismatches are reflected equally by an
        `'M'` in the CIGAR string.
    hard_clip : bool, optional
        If true, clipped bases are hard-clipped.
        Otherwise, clipped bases are soft-clipped.
    include_terminal_gaps : bool, optional
        If true, terminal gaps in the segment sequence are included in
        the CIGAR string.
        These are represented by ``D`` operations at the start and/or
        end of the string.
        By default, those terminal gaps are omitted in the CIGAR, which
        is the way SAM/BAM expects a CIGAR to be.
    as_string : bool, optional
        If true, the CIGAR string is returned.
        Otherwise, a list of tuples is returned, where the first element
        of each tuple specifies the :class:`CigarOp` and the second
        element specifies the number of repetitions.

    Returns
    -------
    cigar : str or ndarray, shape=(n,2) dtype=int
        If `as_string` is true, the CIGAR string is returned.
        Otherwise, an array is returned, where the first column
        specifies the :class:`CigarOp` and the second column specifies
        the number of repetitions of that operation.

    See Also
    --------
    read_alignment_from_cigar : The reverse operation.

    Notes
    -----
    If `include_terminal_gaps` is set to true, you usually want to set
    ``position=0`` in :func:`read_alignment_from_cigar` to get the
    correct alignment.

    Examples
    --------

    >>> ref = NucleotideSequence("TATAAAAGGTTTCCGACCGTAGGTAGCTGA")
    >>> seg = NucleotideSequence("CCCCGGTTTGACCGTATGTAG")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> semiglobal_alignment = align_optimal(
    ...     ref, seg, matrix, local=False, terminal_penalty=False
    ... )[0]
    >>> print(semiglobal_alignment)
    TATAAAAGGTTTCCGACCGTAGGTAGCTGA
    ---CCCCGGTTT--GACCGTATGTAG----
    >>> print(write_alignment_to_cigar(semiglobal_alignment))
    9M2D12M
    >>> print(write_alignment_to_cigar(semiglobal_alignment, introns=[(12, 14)]))
    9M2N12M
    >>> print(write_alignment_to_cigar(semiglobal_alignment, distinguish_matches=True))
    4X5=2D7=1X4=
    >>> print(write_alignment_to_cigar(semiglobal_alignment, include_terminal_gaps=True))
    3D9M2D12M4D
    >>> local_alignment = align_optimal(ref, seg, matrix, local=True)[0]
    >>> print(local_alignment)
    GGTTTCCGACCGTAGGTAG
    GGTTT--GACCGTATGTAG
    >>> print(write_alignment_to_cigar(local_alignment, hard_clip=False))
    4S5M2D12M
    >>> print(write_alignment_to_cigar(local_alignment, hard_clip=True))
    4H5M2D12M

    Writing operations as BAM codes is also possible:

    >>> op_tuples = write_alignment_to_cigar(semiglobal_alignment, as_string=False)
    >>> for op, length in op_tuples:
    ...     print(CigarOp(op).name, length)
    MATCH 9
    DELETION 2
    MATCH 12
    """
    if not include_terminal_gaps:
        alignment = _remove_terminal_segment_gaps(alignment, segment_index)

    ref_trace = alignment.trace[:, reference_index]
    seg_trace = alignment.trace[:, segment_index]
    operations = np.full(alignment.trace.shape[0], CigarOp.MATCH, dtype=int)

    insertion_mask = ref_trace == -1
    deletion_mask = seg_trace == -1
    if np.any(insertion_mask & deletion_mask):
        raise ValueError(
            "Alignment contains insertion and deletion at the same position"
        )
    operations[insertion_mask] = CigarOp.INSERTION
    operations[deletion_mask] = CigarOp.DELETION

    if introns is not None:
        intron_mask = np.zeros(operations.shape[0], dtype=bool)
        for start, stop in introns:
            if start >= stop:
                raise ValueError("Intron start must be smaller than intron stop")
            if start < 0:
                raise ValueError("Intron start must not be negative")
            intron_mask[(ref_trace >= start) & (ref_trace < stop)] = True
        if np.any(intron_mask & ~deletion_mask):
            raise ValueError("Introns must be within gaps in the reference sequence")
        operations[intron_mask] = CigarOp.INTRON

    if distinguish_matches:
        symbol_codes = get_codes(alignment)
        ref_codes = symbol_codes[reference_index, :]
        seg_codes = symbol_codes[segment_index, :]
        equal_mask = ref_codes == seg_codes
        match_mask = operations == CigarOp.MATCH
        operations[equal_mask & match_mask] = CigarOp.EQUAL
        operations[~equal_mask & match_mask] = CigarOp.DIFFERENT

    op_tuples = _aggregate_consecutive(operations)

    clip_op = CigarOp.HARD_CLIP if hard_clip else CigarOp.SOFT_CLIP
    start_clip_length, end_clip_length = _find_clipped_bases(alignment, segment_index)
    if start_clip_length != 0:
        start_clip = [(clip_op, start_clip_length)]
    else:
        start_clip = np.zeros((0, 2), dtype=int)
    if end_clip_length != 0:
        end_clip = [(clip_op, end_clip_length)]
    else:
        end_clip = np.zeros((0, 2), dtype=int)
    op_tuples = np.concatenate((start_clip, op_tuples, end_clip))

    if as_string:
        cigar = _cigar_from_op_tuples(op_tuples)
        return cigar
    else:
        return op_tuples


def _remove_terminal_segment_gaps(alignment, segment_index):
    """
    Remove terminal gaps in the segment sequence.
    """
    no_gap_pos = np.where(alignment.trace[:, segment_index] != -1)[0]
    return alignment[no_gap_pos[0] : no_gap_pos[-1] + 1]


def _find_clipped_bases(alignment, segment_index):
    """
    Find the number of clipped bases at the start and end of the segment.
    """
    # Finding the clipped part is easier, when the terminal segment gaps
    # are removed (if not already done)
    alignment = _remove_terminal_segment_gaps(alignment, segment_index)
    seg_trace = alignment.trace[:, segment_index]
    # Missing bases at the beginning and end of the segment are
    # interpreted as clipped
    # As first element in the segment trace is the first aligned base,
    # all previous bases are clipped...
    start_clip_length = seg_trace[0]
    # ...and the same applies for the last base
    end_clip_length = len(alignment.sequences[segment_index]) - seg_trace[-1] - 1
    return start_clip_length, end_clip_length


def _aggregate_consecutive(operations):
    """
    Aggregate consecutive operations of the same type.
    """
    op_start_indices = np.where(operations[:-1] != operations[1:])[0]
    # Also include the first operation
    op_start_indices += 1
    op_start_indices = np.concatenate(([0], op_start_indices))
    ops = operations[op_start_indices]
    length = np.diff(np.append(op_start_indices, len(operations)))
    return np.stack((ops, length), axis=-1)


def _cigar_from_op_tuples(op_tuples):
    """
    Create a CIGAR string from a list of BAM integer tuples.

    The first element of each tuple specifies the operation and the
    second element specifies the number of repetitions.
    """
    cigar = ""
    for op, count in op_tuples:
        cigar += str(count) + CigarOp(op).to_cigar_symbol()
    return cigar


def _op_tuples_from_cigar(cigar):
    """
    Create a list of tuples from a CIGAR string.
    """
    op_tuples = []
    count = ""
    for char in cigar:
        if char.isdigit():
            count += char
        else:
            op = CigarOp.from_cigar_symbol(char)
            op_tuples.append((op, count))
            count = ""
    return np.array(op_tuples, dtype=int)
