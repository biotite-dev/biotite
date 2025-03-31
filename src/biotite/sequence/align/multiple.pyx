# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["align_multiple"]

cimport cython
cimport numpy as np
from libc.math cimport log

import numpy as np
from .matrix import SubstitutionMatrix
from .alignment import Alignment
from .pairwise import align_optimal
from ..sequence import Sequence
from ..alphabet import Alphabet
from ..phylo.upgma import upgma
from ..phylo.tree import Tree, TreeNode, as_binary


ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64
ctypedef np.float32_t float32

ctypedef fused CodeType:
    uint8
    uint16
    uint32
    uint64


cdef float32 MAX_FLOAT = np.finfo(np.float32).max


class GapSymbol:

    _instance = None

    def __init__(self):
        if GapSymbol._instance is not None:
            raise ValueError(
                "Cannot instantiate this singleton more than one time"
            )
        else:
            GapSymbol._instance = self

    @staticmethod
    def instance():
        if GapSymbol._instance is None:
            GapSymbol._instance = GapSymbol()
        return GapSymbol._instance

    def __str__(self):
        return "-"

    def __hash__(self):
        return 0


def align_multiple(sequences, matrix, gap_penalty=-10, terminal_penalty=True,
                   distances=None, guide_tree=None):
    r"""
    align_multiple(sequences, matrix, gap_penalty=-10,
                   terminal_penalty=True, distances=None,
                   guide_tree=None)

    Perform a multiple sequence alignment using a progressive
    alignment algorithm. :footcite:`Feng1987`

    Based on pairwise sequence distances a guide tree is constructed.
    The sequences are progessively aligned according to the tree,
    following the rule 'Once a gap, always a gap'.

    Parameters
    ----------
    sequences : list of Sequence
        The sequences to be aligned.
        The alpahbet of the substitution matrix must be equal or
        extend the alphabet of each sequence.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
        Must be symmetric.
    gap_penalty : int or tuple(int, int), optional
        If an integer is provided, the value will be interpreted as
        general gap penalty. If a tuple is provided, an affine gap
        penalty is used. The first integer in the tuple is the gap
        opening penalty, the second integer is the gap extension
        penalty.
        The values need to be negative.
    terminal_penalty : bool, optional
        If true, gap penalties are applied to terminal gaps.
    distances : ndarray, shape=(n,n)
        Pairwise distances of the sequences.
        The matrix must be symmetric and all entries must be larger
        than 0.
        By default the pairwise distances are calculated from
        similarities obtained from optimal global pairwise alignments
        (:func:`align_optimal()`).
        The similarities are converted into distances using the method
        proposed by Feng & Doolittle :footcite:`Feng1996`.
    guide_tree : Tree
        The guide tree to be used for the progressive alignment.
        By default the guide tree is constructed from `distances`
        via the UPGMA clustering method.

    Returns
    -------
    alignment : Alignment
        The global multiple sequence alignment of the input sequences.
    order : ndarray, dtype=int
        The sequence order represented by the guide tree.
        When this order is applied to alignment sequence order,
        similar sequences are adjacent to each other.
    tree : Tree
        The guide tree used for progressive alignment.
        Equal to `guide_tree` if provided.
    distance_matrix : ndarray, shape=(n,n), dtype=float32
        The pairwise distance matrix used to construct the guide tree.
        Equal to `distances` if provided.

    Notes
    -----
    The similarity to distance conversion is performed according to the
    following formula:

    .. math:: D_{a,b} = -\ln\left(
                 \frac
                    { S_{a,b} - S_{a,b}^{rand} }
                    { S_{a,b}^{max} - S_{a,b}^{rand} }
              \right)

    .. math:: S_{a,b}^{max} = \frac{ S_{a,a} + S_{b,b} }{ 2 }

    .. math:: S_{a,b}^{rand} = \frac{1}{L_{a,b}}
              \left(
                 \sum_{x \in \Omega} \sum_{y \in \Omega}
                 s_{x,y} \cdot N_a(x) \cdot N_b(y)
              \right)
              + N_{a,b}^{open} \cdot p^{open} + N_{a,b}^{ext} \cdot p^{ext}

    :math:`D_{a,b}` - The distance between the sequences *a* and *b*.

    :math:`S_{a,b}` - The similarity score between the sequences *a* and *b*.

    :math:`s_{x,y}` - The similarity score between the symbols *x* and *y*.

    :math:`\Omega` - The sequence alphabet.

    :math:`N_a(x)` - Number of occurences of symbol *x* in sequence *a*.

    :math:`N_{a,b}^{open}, N_{a,b}^{ext}` - Number of gap openings/
    extensions, in the alignment of *a* and *b*.

    :math:`p^{open}, p^{ext}` - The penalty for a gap opening/extension.

    :math:`L_{a,b}` - Number of columns in the alignment of *a* and *b*.

    In rare cases of extremely unrelated sequences, :math:`S_{a,b}`
    can be lower than :math:`S_{a,b}^{rand}`.
    In this case the logarithm cannot be calculated and a
    :class:`ValueError` is raised.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> seq1 = ProteinSequence("BIQTITE")
    >>> seq2 = ProteinSequence("TITANITE")
    >>> seq3 = ProteinSequence("BISMITE")
    >>> seq4 = ProteinSequence("IQLITE")
    >>> matrix = SubstitutionMatrix.std_protein_matrix()
    >>>
    >>> alignment, order, tree, distances = align_multiple(
    ...     [seq1, seq2, seq3, seq4], matrix
    ... )
    >>>
    >>> print(alignment)
    BIQT-ITE
    TITANITE
    BISM-ITE
    -IQL-ITE
    >>> print(alignment[:, order.tolist()])
    -IQL-ITE
    BISM-ITE
    BIQT-ITE
    TITANITE
    >>> print(distances)
    [[0.000 1.034 0.382 0.560]
     [1.034 0.000 0.923 1.132]
     [0.382 0.923 0.000 0.632]
     [0.560 1.132 0.632 0.000]]
    >>>
    >>> print(tree.to_newick(
    ...     labels=["seq1", "seq2", "seq3", "seq4"], include_distance=False
    ... ))
    ((seq4,(seq3,seq1)),seq2);
    """
    if not matrix.is_symmetric():
        raise ValueError("A symmetric substitution matrix is required")
    alphabet = matrix.get_alphabet1()
    for i, seq in enumerate(sequences):
        if seq.code is None:
            raise ValueError(f"Code of sequence {i} is 'None'")
        if not alphabet.extends(seq.get_alphabet()):
            raise ValueError(
                f"The substitution matrix and sequence {i} have "
                f"incompatible alphabets"
            )

    # Create guide tree
    # Template parameter workaround
    _T = sequences[0].code
    if distances is None:
        distances = _get_distance_matrix(
            _T, sequences, matrix, gap_penalty, terminal_penalty
        )
    else:
        distances = distances.astype(np.float32, copy=True)
    if guide_tree is None:
        guide_tree = upgma(distances)
    else:
        # Assure that every node in the guide tree is binary
        guide_tree = as_binary(guide_tree)

    # Create new matrix with neutral gap symbol
    gap_symbol = GapSymbol.instance()
    new_alphabet = Alphabet(
        matrix.get_alphabet1().get_symbols() + (gap_symbol,)
    )
    new_score_matrix = np.zeros(
        (len(new_alphabet), len(new_alphabet)), dtype=np.int32
    )
    # New substitution matrix is the same as the old one,
    # except the neutral ghap symbol,
    # that scores 0 with all other symbols
    new_score_matrix[:-1,:-1] = matrix.score_matrix()
    new_matrix = SubstitutionMatrix(
        new_alphabet, new_alphabet, new_score_matrix
    )

    # Progressive alignment
    gap_symbol_code = new_alphabet.encode(gap_symbol)
    order, aligned_seqs = _progressive_align(
        _T, sequences, guide_tree.root, distances, new_matrix,
        gap_symbol_code, gap_penalty, terminal_penalty
    )
    aligned_seq_codes = [seq.code for seq in aligned_seqs]

    # Remove neutral gap symbols and create actual trace
    seq_i = np.zeros(len(aligned_seqs))
    trace = np.full(
        (len(aligned_seqs[0]), len(aligned_seqs)), -1, dtype=np.int64)
    for j in range(trace.shape[1]):
        seq_code = aligned_seq_codes[j]
        seq_i = 0
        for i in range(trace.shape[0]):
            if seq_code[i] == gap_symbol_code:
                trace[i,j] = -1
            else:
                trace[i,j] = seq_i
                seq_i += 1
    aligned_seq_codes = [
        code[code != gap_symbol_code] for code in aligned_seq_codes
    ]
    for i in range(len(aligned_seqs)):
        aligned_seqs[i].code = aligned_seq_codes[i]

    # Reorder alignmets into original alignemnt
    new_order = np.argsort(order)
    aligned_seqs = [aligned_seqs[pos] for pos in new_order]
    trace = trace[:, new_order]

    return Alignment(aligned_seqs, trace), order, guide_tree, distances


def _get_distance_matrix(CodeType[:] _T, sequences, matrix,
                         gap_penalty, terminal_penalty):
    """
    Create all pairwise alignments for the given sequences and use the
    method proposed by Feng & Doolittle to calculate the pairwise
    distance matrix

    Parameters
    ----------
    _T : ndarray, dtype=VARAIBLE
        A little bit hacky workaround to get the correct dtype for the
        sequence code of the sequences in a static way
        (important for Cython).
    sequences : list of Sequence, length=n
        The sequences to get the distance matrix for.
    matrix : SubstitutionMatrix
        The substitution matrix used for the alignments.
    gap_penalty : int or tuple(int, int)
        A linear or affine gap penalty for the alignments.
    terminal_penalty : bool
        Whether to or not count terminal gap penalties for the
        alignments.

    Returns
    -------
    distances : ndarray, shape=(n,n), dtype=float32
        The pairwise distance matrix.
    """
    cdef int i, j

    cdef np.ndarray scores = np.zeros(
        (len(sequences), len(sequences)), dtype=np.int32
    )
    cdef np.ndarray alignments = np.full(
        (len(sequences), len(sequences)), None, dtype=object
    )
    for i in range(len(sequences)):
        # Inclusive range
        for j in range(i+1):
            # For this method we only consider one alignment:
            # Score is equal for all alignments
            # Alignment length is equal for most alignments
            alignment = align_optimal(
                sequences[i], sequences[j], matrix,
                gap_penalty, terminal_penalty, max_number=1
            )[0]
            scores[i,j] = alignment.score
            alignments[i,j] = alignment

    ### Distance calculation from similarity scores ###
    # Calculate the occurences of each symbol code in each sequence
    # This is used later for the random score
    # Both alphabets are the same
    cdef CodeType alphabet_size = len(matrix.get_alphabet1())
    cdef np.ndarray code_count = np.zeros(
        (len(sequences), alphabet_size), dtype=np.int32
    )
    cdef int32[:,:] code_count_v = code_count
    for i in range(len(sequences)):
        code_count[i] = np.bincount(sequences[i].code, minlength=alphabet_size)

    cdef int gap_open=0, gap_ext=0
    if type(gap_penalty) == int:
        gap_open = gap_penalty
        gap_ext = gap_penalty
    elif type(gap_penalty) == tuple:
        gap_open = gap_penalty[0]
        gap_ext = gap_penalty[1]
    else:
        raise TypeError("Gap penalty must be either integer or tuple")

    cdef const int32[:,:] score_matrix = matrix.score_matrix()
    cdef int32[:,:] scores_v = scores
    cdef np.ndarray distances = np.zeros(
        (scores.shape[0], scores.shape[1]), dtype=np.float32
    )
    cdef float32[:,:] distances_v = distances
    cdef CodeType[:] seq_code1, seq_code2
    cdef CodeType code1, code2
    cdef float32 score_rand, score_max

    # Calculate distance
    # i and j are indicating the alignment between the sequences i and j
    for i in range(scores_v.shape[0]):
        for j in range(i):
            score_max =  (scores_v[i,i] + scores_v[j,j]) / 2.0
            score_rand = 0
            for code1 in range(alphabet_size):
                for code2 in range(alphabet_size):
                    score_rand += score_matrix[code1,code2] \
                                  * code_count[i,code1] \
                                  * code_count[j,code2]
            score_rand /= alignments[i,j].trace.shape[0]
            gap_open_count, gap_ext_count = _count_gaps(
                alignments[i,j].trace.astype(np.int64, copy=False),
                terminal_penalty
            )
            score_rand += gap_open_count * gap_open
            score_rand += gap_ext_count * gap_ext
            if scores_v[i,j] < score_rand:
                # Randomized alignment is better than actual alignment
                # -> the logaritmus argument would become negative
                # resulting in an NaN distance
                raise ValueError(
                    f"The randomized alignment of sequences {j} and {i} "
                    f"scores better than the real pairwise alignment, "
                    f"cannot calculate proper pairwise distance"
                )
            else:
                distances_v[i,j] = -log(
                    (scores_v[i,j] - score_rand) / (score_max - score_rand)
                )
            # Pairwise distance matrix is symmetric
            distances_v[j,i] = distances_v[i,j]
    return distances


def _count_gaps(int64[:,:] trace_v, bint terminal_penalty):
    """
    Count the number of gap openings and gap extensions in an alignment
    trace.

    Parameters
    ----------
    trace_v : ndarary, shape=(n,2), dtype=int
        The alignemnt trace.
    terminal_penalty : bool
        Whether to or not count terminal gap penalties.

    Returns
    -------
    gap_open_count, gap_ext_count: int
        The number of gap opening and gap extension columns
    """
    cdef int i, j
    cdef int gap_open_count=0, gap_ext_count=0
    cdef int start_index=-1, stop_index=-1

    if not terminal_penalty:
        # Ignore terminal gaps
        # -> get start and exclusive stop column of the trace
        # excluding terminal gaps
        for i in range(trace_v.shape[0]):
            # Check if all sequences have no gap at the given position
            if trace_v[i,0] != -1 and trace_v[i,1] != -1:
                start_index = i
                break
        # Reverse iteration
        for i in range(trace_v.shape[0]-1, -1, -1):
            # Check if all sequences have no gap at the given position
             if trace_v[i,0] != -1 and trace_v[i,1] != -1:
                stop_index = i+1
                break
        if start_index == -1 or stop_index == -1:
            return 0, 0
        trace_v = trace_v[start_index : stop_index]

    if trace_v[0,0] == -1:
        gap_open_count += 1
    if trace_v[0,1] == -1:
        gap_open_count += 1
    for i in range(1, trace_v.shape[0]):
        # trace_v.shape[1] = 2 due to pairwise alignemt
        for j in range(trace_v.shape[1]):
            if trace_v[i,j] == -1:
                if trace_v[i-1,j] == -1:
                    gap_ext_count += 1
                else:
                    gap_open_count += 1
    return gap_open_count, gap_ext_count


def _progressive_align(CodeType[:] _T, sequences, tree_node,
                       float32[:,:]distances_v, matrix,
                       int gap_symbol_code, gap_penalty, terminal_penalty):
    """
    Conduct the progressive alignemt of the sequences that are
    referred to by the given guide tree node.

    At first the the two sub-MSAs are calculated from the child nodes
    of the given node.
    Then the sub-MSAs are combined to one MSA by aligning the two
    sequences from both sub-MSAs with the lowest distance to each other,
    taken from the pairwise distance matrix.
    The gaps inserted in this pairwise alignment are also inserted
    into all other sequences in the respective sub-MSA at the same
    position.

    Parameters
    ----------
    _T : ndarray, dtype=VARAIBLE
        A little bit hacky workaround to get the correct dtype for the
        sequence code of the sequences in a static way
        (important for Cython).
    sequences : list of Sequence, lebgth=n
        All sequences that should be aligned in the MSA.
    tree_node : TreeNode
        This guide tree node defines, which of sequences in the
        `sequences` parameter should be aligned in this call.
        This is the only parameter that changes in the series of
        recursive calls of this function.
    distances_v : ndarray, shape=(n,n)
        The pairwise distance matrix.
    matrix : SubstitutionMatrix
        The substitution matrix used for the alignments.
    gap_symbol_code : int
        The symbol code for the gap symbol.
    gap_penalty : int or tuple(int, int)
        A linear or affine gap penalty for the alignments.
    terminal_penalty : bool
        Whether to or not count terminal gap penalties for the
        alignments.

    Returns
    -------
    order : ndarray, shape=(m,), dtype=int
        The index of each element in `aligned_sequences` in the
        orginal `sequences` parameter.
    aligned_sequences : list of Sequence, length=m
        A list of the sequences that were aligned.
        Instead of an :class:`Alignment` object that represents the gaps
        as ``-1`` in the trace, the gaps are represented as dedicated
        gap symbols in this case.
        This allows for the pairwise alignemt of gapped sequences.
    """
    cdef int i=0, j=0
    cdef int i_min=0, j_min=0
    cdef float32 dist_min, dist
    cdef int32[:] indices1_v, indices2_v
    cdef np.ndarray incides1, incides2
    cdef list aligned_seqs1, aligned_seqs2

    if tree_node.is_leaf():
        # Child node -> Cannot do an alignment
        # -> Just return the sequence corresponding to the leaf node
        # Copy sequences to avoid modification of input sequences
        # when neutral gap character is inserted
        return np.array([tree_node.index], dtype=np.int32), \
               [sequences[tree_node.index].copy()]

    else:
        # Multiple alignment of sequences corresponding to both child nodes
        child1, child2 = tree_node.children
        incides1, aligned_seqs1 = _progressive_align(
            _T, sequences, child1, distances_v, matrix,
            gap_symbol_code, gap_penalty, terminal_penalty
        )
        indices1_v = incides1
        incides2, aligned_seqs2 = _progressive_align(
            _T, sequences, child2, distances_v, matrix,
            gap_symbol_code, gap_penalty, terminal_penalty
        )
        indices2_v = incides2

        # Find sequence pair with lowest distance
        dist_min = MAX_FLOAT
        for i in range(indices1_v.shape[0]):
            for j in range(indices2_v.shape[0]):
                dist = distances_v[indices1_v[i], indices2_v[j]]
                if dist < dist_min:
                    dist_min = dist
                    i_min = i
                    j_min = j
        # Alignment of sequence pair with lowest distance
        # For this method we only consider one alignment:
        alignment = align_optimal(
            aligned_seqs1[i_min], aligned_seqs2[j_min], matrix,
            gap_penalty, terminal_penalty, max_number=1
        )[0]
        # Place neutral gap symbol for position of new gaps
        # in both sequence groups
        for i in range(len(aligned_seqs1)):
            seq = aligned_seqs1[i]
            seq.code = _replace_gaps(
                _T, alignment.trace[:,0], seq.code, gap_symbol_code
            )
        for i in range(len(aligned_seqs2)):
            seq = aligned_seqs2[i]
            seq.code = _replace_gaps(
                _T, alignment.trace[:,1], seq.code, gap_symbol_code
            )
        return np.append(incides1, incides2), \
               aligned_seqs1 + aligned_seqs2



def _replace_gaps(CodeType[:] _T,
                  int64[:] partial_trace_v,
                  np.ndarray seq_code,
                  int gap_symbol_code):
    """
    Replace gaps in a sequence in an :class:`Alignment` with a dedicated
    gap symbol.

    The replacement is required by the progressive alignment algorithm
    to be able to align gapped sequences with each other.

    Parameters
    ----------
    _T : ndarray, dtype=VARAIBLE
        A little bit hacky workaround to get the correct dtype for the
        sequence code of the sequences in a static way
        (important for Cython).
    partial_trace_v : ndarary, shape=(m,), dtype=int
        The row of the alignemnt trace reffering to the given sequence.
    seq_code : ndarary, shape=(n,)
        The sequence code representing the given sequence.
    gap_symbol_code : int
        The symbol code for the gap symbol.

    Returns
    -------
    new_seq_code : ndarary, shape=(m,)
        The sequence code representing a new sequence, that is the given
        sequence with inserted gap symbols.
    """
    cdef int i
    cdef int64 index
    cdef CodeType code

    cdef CodeType[:] seq_code_v = seq_code
    cdef np.ndarray new_seq_code = np.zeros(
        partial_trace_v.shape[0], dtype=seq_code.dtype
    )
    cdef CodeType[:] new_seq_code_v = new_seq_code

    for i in range(partial_trace_v.shape[0]):
        index = partial_trace_v[i]
        if index == -1:
            new_seq_code_v[i] = gap_symbol_code
        else:
            new_seq_code_v[i] = seq_code[index]

    return new_seq_code
