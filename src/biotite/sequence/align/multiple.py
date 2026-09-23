from __future__ import annotations

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["align_multiple"]

from collections.abc import Sequence as SequenceABC
import networkx as nx
import numpy as np
from biotite.rust.sequence.align import align_optimal as rust_align_optimal
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.align.pairwise import align_optimal
from biotite.sequence.alphabet import common_alphabet
from biotite.sequence.phylo.tree import as_binary, get_leaves, get_root
from biotite.sequence.phylo.upgma import upgma
from biotite.sequence.sequence import Sequence
from biotite.typing import K, N, NDArray1, NDArray2, S


def align_multiple(
    sequences: SequenceABC[Sequence[S]],
    matrix: SubstitutionMatrix[S, S] | tuple[int, int] = (1, -1),
    gap_penalty: int | tuple[int, int] = -1,
    terminal_penalty: bool = True,
    distances: NDArray2[N, N, np.floating] | None = None,
    guide_tree: nx.DiGraph | None = None,
) -> tuple[
    Alignment, NDArray1[K, np.integer], nx.DiGraph, NDArray2[int, int, np.float32]
]:
    r"""
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
    matrix : SubstitutionMatrix or tuple(int, int), optional
        Either a substitution matrix or a ``(match, mismatch)`` pair of
        scores.
        By default a match scores ``1`` and a mismatch scores ``-1``.
        A substitution matrix must be symmetric.
        The pair of scores does not require a substitution matrix over
        the potentially large alphabet of the sequences, which saves
        memory and run time.
        In this case at least one sequence must have an alphabet that
        extends the alphabets of all other sequences.
    gap_penalty : int or tuple(int, int), optional
        If an integer is provided, the value will be interpreted as
        general gap penalty. If a tuple is provided, an affine gap
        penalty is used. The first integer in the tuple is the gap
        opening penalty, the second integer is the gap extension
        penalty.
        The values need to be negative.
        By default a linear gap penalty of ``-1`` is used.
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
    guide_tree : DiGraph
        The guide tree to be used for the progressive alignment.
        The leaf nodes must be the sequence indices.
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
    tree : DiGraph
        The guide tree used for progressive alignment.
        The binary version of `guide_tree` if provided
        (see :func:`biotite.sequence.phylo.as_binary()`).
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

    >>> import biotite.sequence.phylo as phylo
    >>> seq1 = ProteinSequence("BIQTITE")
    >>> seq2 = ProteinSequence("TITANITE")
    >>> seq3 = ProteinSequence("BISMITE")
    >>> seq4 = ProteinSequence("IQLITE")
    >>> matrix = SubstitutionMatrix.std_protein_matrix()
    >>>
    >>> alignment, order, tree, distances = align_multiple(
    ...     [seq1, seq2, seq3, seq4], matrix, gap_penalty=-10
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
    >>> print(phylo.to_newick(
    ...     tree, labels=["seq1", "seq2", "seq3", "seq4"], include_distance=False
    ... ))
    ((seq4,(seq3,seq1)),seq2);
    """
    if len(sequences) == 0:
        raise ValueError("At least one sequence is required")
    for i, seq in enumerate(sequences):
        if seq.code is None:
            raise ValueError(f"Code of sequence {i} is 'None'")
    if isinstance(matrix, tuple):
        alphabet = common_alphabet([seq.get_alphabet() for seq in sequences])
        if alphabet is None:
            raise ValueError(
                "The sequences have incompatible alphabets, "
                "a substitution matrix is required"
            )
    else:
        if not matrix.is_symmetric():
            raise ValueError("A symmetric substitution matrix is required")
        alphabet = matrix.get_alphabet1()
        for i, seq in enumerate(sequences):
            if not alphabet.extends(seq.get_alphabet()):
                raise ValueError(
                    f"The substitution matrix and sequence {i} "
                    "have incompatible alphabets"
                )

    # Create guide tree
    if distances is None:
        distance_matrix = _get_distance_matrix(
            sequences, matrix, gap_penalty, terminal_penalty
        )
    else:
        distance_matrix = np.array(distances, dtype=np.float32)
    if guide_tree is None:
        guide_tree = upgma(distance_matrix)
    else:
        # Assure that every node in the guide tree is binary
        guide_tree = as_binary(guide_tree)
        if sorted(get_leaves(guide_tree)) != list(range(len(sequences))):
            raise ValueError(
                "The leaf nodes of the guide tree must be the sequence indices"
            )

    # The neutral gap symbol gets the code after the last alphabet symbol
    # and scores 0 with all other symbols
    gap_symbol_code = len(alphabet)
    if isinstance(matrix, tuple):
        scoring = (int(matrix[0]), int(matrix[1]), gap_symbol_code)
    else:
        extended_score_matrix = np.zeros(
            (gap_symbol_code + 1, gap_symbol_code + 1), dtype=np.int32
        )
        extended_score_matrix[:-1, :-1] = matrix.score_matrix()
        scoring = extended_score_matrix

    # Progressive alignment
    order, gapped_codes = _progressive_align(
        sequences,
        guide_tree,
        distance_matrix,
        scoring,
        gap_symbol_code,
        gap_penalty,
        terminal_penalty,
    )

    # Convert the gap symbols into the actual trace
    gapped_codes = np.stack(gapped_codes, axis=-1)
    is_gap = gapped_codes == gap_symbol_code
    trace = np.cumsum(~is_gap, axis=0, dtype=np.int64) - 1
    trace[is_gap] = -1
    # Reorder the alignment into the original sequence order
    trace = trace[:, np.argsort(order)]

    return Alignment(list(sequences), trace), order, guide_tree, distance_matrix


def _get_distance_matrix(
    sequences: SequenceABC[Sequence],
    matrix: SubstitutionMatrix | tuple[int, int],
    gap_penalty: int | tuple[int, int],
    terminal_penalty: bool,
) -> NDArray2[int, int, np.float32]:
    """
    Create all pairwise alignments for the given sequences and use the
    method proposed by Feng & Doolittle to calculate the pairwise
    distance matrix.

    Parameters
    ----------
    sequences : list of Sequence, length=n
        The sequences to get the distance matrix for.
    matrix : SubstitutionMatrix or tuple(int, int)
        The substitution matrix or ``(match, mismatch)`` scores used for
        the alignments.
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
    if isinstance(gap_penalty, int):
        gap_open = gap_penalty
        gap_ext = gap_penalty
    elif isinstance(gap_penalty, tuple):
        gap_open, gap_ext = gap_penalty
    else:
        raise TypeError("Gap penalty must be either integer or tuple")

    n_seq = len(sequences)
    scores = np.zeros((n_seq, n_seq), dtype=np.int64)
    ali_lengths = np.zeros((n_seq, n_seq), dtype=np.int64)
    gap_open_counts = np.zeros((n_seq, n_seq), dtype=np.int64)
    gap_ext_counts = np.zeros((n_seq, n_seq), dtype=np.int64)
    for i in range(n_seq):
        # Inclusive range, as the self-alignment scores are required
        # for the maximum score
        for j in range(i + 1):
            # For this method we only consider one alignment:
            # Score is equal for all alignments
            # Alignment length is equal for most alignments
            alignment = align_optimal(
                sequences[i],
                sequences[j],
                matrix,
                gap_penalty,
                terminal_penalty,
                max_number=1,
            )[0]
            if alignment.score is None:
                raise ValueError("The pairwise alignment has no score")
            scores[i, j] = alignment.score
            ali_lengths[i, j] = alignment.trace.shape[0]
            gap_open_counts[i, j], gap_ext_counts[i, j] = _count_gaps(
                alignment.trace, terminal_penalty
            )

    ### Distance calculation from similarity scores ###
    # The random score is the sum over all symbol pairs of
    # score times occurrences in the respective sequences
    rand_scores = _random_scores(sequences, matrix)

    # i and j are indicating the alignment between the sequences i and j
    i, j = np.tril_indices(n_seq, k=-1)
    score_max = (scores[i, i] + scores[j, j]) / 2.0
    score_rand = (
        rand_scores[i, j] / ali_lengths[i, j]
        + gap_open_counts[i, j] * gap_open
        + gap_ext_counts[i, j] * gap_ext
    )
    score = scores[i, j]
    invalid = np.where(score < score_rand)[0]
    if len(invalid) > 0:
        # Randomized alignment is better than actual alignment
        # -> the logarithm argument would become negative
        # resulting in an NaN distance
        raise ValueError(
            f"The randomized alignment of sequences {j[invalid[0]]} and "
            f"{i[invalid[0]]} scores better than the real pairwise alignment, "
            f"cannot calculate proper pairwise distance"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = -np.log((score - score_rand) / (score_max - score_rand))
    # Pairwise distance matrix is symmetric
    distances = np.zeros((n_seq, n_seq), dtype=np.float32)
    distances[i, j] = distance
    distances[j, i] = distance
    return distances


def _random_scores(
    sequences: SequenceABC[Sequence], matrix: SubstitutionMatrix | tuple[int, int]
) -> NDArray2[int, int, np.float64]:
    """
    Calculate the expected score for each pair of sequences, if the
    symbols were randomly aligned, without normalization by the
    alignment length.

    Parameters
    ----------
    sequences : list of Sequence, length=n
        The sequences.
    matrix : SubstitutionMatrix or tuple(int, int)
        The substitution matrix or ``(match, mismatch)`` scores.

    Returns
    -------
    rand_scores : ndarray, shape=(n,n), dtype=float
        The sum over all symbol pairs of score times occurrences.
    """
    # Count the occurrences of each symbol code in each sequence
    # Only symbol codes that actually appear in the sequences are
    # considered, to avoid arrays of the alphabet size
    n_seq = len(sequences)
    lengths = np.array([len(seq) for seq in sequences])
    distinct_codes, compact_codes = np.unique(
        np.concatenate([seq.code for seq in sequences]), return_inverse=True
    )
    seq_indices = np.repeat(np.arange(n_seq), lengths)
    code_counts = np.bincount(
        seq_indices * len(distinct_codes) + compact_codes,
        minlength=n_seq * len(distinct_codes),
    ).reshape(n_seq, len(distinct_codes))
    code_counts = code_counts.astype(np.float64)

    if isinstance(matrix, tuple):
        # The score matrix is the mismatch score for all symbol pairs
        # plus the difference to the match score for equal symbols
        match, mismatch = matrix
        return mismatch * np.outer(lengths, lengths) + (match - mismatch) * (
            code_counts @ code_counts.T
        )
    else:
        score_matrix = matrix.score_matrix()[np.ix_(distinct_codes, distinct_codes)]
        return code_counts @ score_matrix.astype(np.float64) @ code_counts.T


def _count_gaps(
    trace: NDArray2[N, N, np.integer], terminal_penalty: bool
) -> tuple[int, int]:
    """
    Count the number of gap openings and gap extensions in an alignment
    trace.

    Parameters
    ----------
    trace : ndarary, shape=(n,2), dtype=int
        The alignment trace.
    terminal_penalty : bool
        Whether to or not count terminal gap penalties.

    Returns
    -------
    gap_open_count, gap_ext_count: int
        The number of gap opening and gap extension columns
    """
    gaps = trace == -1
    if not terminal_penalty:
        # Ignore terminal gaps
        # -> only consider the columns between the first and last
        # column where no sequence has a gap
        gapless_columns = np.where(~gaps.any(axis=1))[0]
        if len(gapless_columns) == 0:
            return 0, 0
        gaps = gaps[gapless_columns[0] : gapless_columns[-1] + 1]
    # A gap in the first column is always an opening,
    # afterwards a gap is an extension if the previous column is also a gap
    gap_open_count = np.count_nonzero(gaps[0]) + np.count_nonzero(gaps[1:] & ~gaps[:-1])
    gap_ext_count = np.count_nonzero(gaps[1:] & gaps[:-1])
    return int(gap_open_count), int(gap_ext_count)


def _progressive_align(
    sequences: SequenceABC[Sequence],
    guide_tree: nx.DiGraph,
    distances: NDArray2[int, int, np.float32],
    scoring: NDArray2[int, int, np.int32] | tuple[int, int, int],
    gap_symbol_code: int,
    gap_penalty: int | tuple[int, int],
    terminal_penalty: bool,
) -> tuple[NDArray1[K, np.integer], list[NDArray1[K, np.integer]]]:
    """
    Conduct the progressive alignment of the sequences according to the
    given guide tree.

    The tree is traversed bottom-up.
    At each intermediate node the sub-MSAs of its two children are
    combined into one MSA by aligning the two sequences from both
    sub-MSAs with the lowest distance to each other, taken from the
    pairwise distance matrix.
    The gaps inserted in this pairwise alignment are also inserted
    into all other sequences in the respective sub-MSA at the same
    position.

    Parameters
    ----------
    sequences : list of Sequence, length=n
        All sequences that should be aligned in the MSA.
    guide_tree : DiGraph
        The binary guide tree, whose leaf nodes are the indices of
        `sequences`.
    distances : ndarray, shape=(n,n)
        The pairwise distance matrix.
    scoring : ndarray, shape=(k+1,k+1), dtype=int32 or tuple(int, int, int)
        The scoring scheme used for the alignments, as accepted by the
        Rust alignment function:
        Either a score matrix extended by the gap symbol or
        ``(match, mismatch, gap_symbol_code)`` scores.
    gap_symbol_code : int
        The symbol code for the gap symbol.
    gap_penalty : int or tuple(int, int)
        A linear or affine gap penalty for the alignments.
    terminal_penalty : bool
        Whether to or not count terminal gap penalties for the
        alignments.

    Returns
    -------
    order : ndarray, shape=(n,), dtype=int
        The index of each element in `gapped_codes` in the
        original `sequences` parameter.
    gapped_codes : list of ndarray, length=n
        The sequence codes of the aligned sequences.
        Instead of an :class:`Alignment` object that represents the gaps
        as ``-1`` in the trace, the gaps are represented as the
        dedicated gap symbol in this case.
        This allows for the pairwise alignment of gapped sequences.
    """
    code_dtype = Sequence.dtype(gap_symbol_code + 1)
    root = get_root(guide_tree)
    # For each processed node the indices of the sequences in its subtree
    # and the corresponding gapped sequence codes
    sub_msas: dict[int, tuple[NDArray1, list[NDArray1]]] = {}
    for node in nx.dfs_postorder_nodes(guide_tree, root):
        children = list(guide_tree.successors(node))
        if len(children) == 0:
            # Leaf node -> Cannot do an alignment
            # -> Just take the sequence corresponding to the leaf node
            sub_msas[node] = (
                np.array([node], dtype=np.int64),
                [sequences[node].code.astype(code_dtype, copy=False)],
            )
            continue

        # Multiple alignment of the sequences corresponding to both child nodes
        indices1, codes1 = sub_msas.pop(children[0])
        indices2, codes2 = sub_msas.pop(children[1])
        # Find sequence pair with lowest distance
        sub_distances = distances[np.ix_(indices1, indices2)]
        i_min, j_min = np.unravel_index(np.argmin(sub_distances), sub_distances.shape)
        # Alignment of the sequence pair with lowest distance
        # For this method we only consider one alignment
        # The gapped sequences are aligned as plain codes, so the Rust
        # function is called directly
        traces, _ = rust_align_optimal(
            codes1[i_min],
            codes2[j_min],
            scoring,
            gap_penalty,
            terminal_penalty,
            False,
            False,
            1,
        )
        trace = traces[0]
        # Place neutral gap symbol for position of new gaps
        # in both sequence groups
        trace1 = trace[:, 0]
        trace2 = trace[:, 1]
        codes1 = [_insert_gaps(code, trace1, gap_symbol_code) for code in codes1]
        codes2 = [_insert_gaps(code, trace2, gap_symbol_code) for code in codes2]
        sub_msas[node] = (np.concatenate([indices1, indices2]), codes1 + codes2)

    return sub_msas[root]


def _insert_gaps(
    seq_code: NDArray1[K, np.integer],
    partial_trace: NDArray1[N, np.integer],
    gap_symbol_code: int,
) -> NDArray1[N, np.integer]:
    """
    Insert gap symbols into a sequence code according to the given
    alignment trace.

    Parameters
    ----------
    seq_code : ndarary, shape=(n,)
        The sequence code representing the given sequence.
    partial_trace : ndarary, shape=(m,), dtype=int
        The column of the alignment trace referring to the given
        sequence.
    gap_symbol_code : int
        The symbol code for the gap symbol.

    Returns
    -------
    new_seq_code : ndarary, shape=(m,)
        The sequence code representing a new sequence, that is the given
        sequence with inserted gap symbols.
    """
    # The gap positions index the last element, which is overwritten anyway
    return np.where(partial_trace == -1, gap_symbol_code, seq_code[partial_trace])
