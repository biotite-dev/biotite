# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for computing the TM-score between two structures and
for computing the superimposition to do so.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "tm_score",
    "superimpose_structural_homologs",
]

import itertools
import numpy as np
from biotite.sequence.align.alignment import get_codes, remove_gaps
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.align.pairwise import align_optimal
from biotite.sequence.seqtypes import PurePositionalSequence
from biotite.structure.filter import filter_amino_acids
from biotite.structure.geometry import distance
from biotite.structure.residues import get_residue_count
from biotite.structure.superimpose import superimpose
from biotite.structure.util import coord_for_atom_name_per_residue

# Minimum value for d0
# This is not part of the explanation in the paper, but it is implemented in TM-align
_D0_MIN = 0.5
# Gap open penalty for hybrid alignment
_HYBRID_PENALTY = -1
# Gap open penalty for pure TM-based alignment
_TM_GAP_PENALTY = -0.6
# Arbitrary scale factor to avoid rounding errors when converting scores to integer
_SCORE_SCALING = 100


def tm_score(
    reference, subject, reference_indices, subject_indices, reference_length="shorter"
):
    """
    Compute the *TM*-score for the given protein structures. :footcite:`Zhang2004`

    Parameters
    ----------
    reference, subject : AtomArray or ndarray, dtype=float
        The protein structures to be compared.
        The number of their atoms may differ from each other.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.
    reference_indices, subject_indices : ndarray, dtype=int, shape=(n,)
        The indices of the atoms in the reference and subject, respectively,
        that correspond to each other.
        In consequence, the length of both arrays must be equal.
    reference_length : int or {"shorter", "longer", "reference"}
        The reference length used to normalize the TM-score.
        If "shorter", the number of residues in the smaller structure is used.
        If "longer", the number of residues in the larger structure is used.
        If "reference", the number of residues in the reference structure is used.
        The length can also be provided directly as an integer.

    Returns
    -------
    tm_score : float
        The *TM*-score for the given structure.

    See Also
    --------
    superimpose_structural_homologs :
        Aims to minimize the *TM*-score between two structures.
        It also returns the corresponding atom indices that can be passed to
        :func:`tm_score()`.

    Notes
    -----
    This functions takes the coordinates as they are.
    It is recommended to use superimpose them using
    :func:`superimpose_structural_homologs()` before, as that function aims to find a
    superimposition that minimizes the *TM*-score.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> reference = atom_array_stack[0]
    >>> subject = atom_array_stack[1]
    >>> superimposed, _, ref_indices, sub_indices = superimpose_structural_homologs(
    ...     reference, subject, max_iterations=1
    ... )
    >>> print(tm_score(reference, superimposed, ref_indices, sub_indices))
    0.69...
    """
    if not np.all(filter_amino_acids(reference)):
        raise ValueError("Reference structure must be peptide only")
    if not np.all(filter_amino_acids(subject)):
        raise ValueError("Subject structure must be peptide only")
    ref_length = _get_reference_length(
        reference_length, get_residue_count(reference), get_residue_count(subject)
    )
    distances = distance(reference[reference_indices], subject[subject_indices])
    return np.sum(_tm_score(distances, ref_length)).item() / ref_length


def superimpose_structural_homologs(
    fixed,
    mobile,
    structural_alphabet="3di",
    substitution_matrix=None,
    max_iterations=float("inf"),
    reference_length="shorter",
):
    """
    Superimpose two remotely homologous protein structures.

    This method relies on structural similarity between the two given structures,
    inspired by the *TM-align algorithm*. :footcite:`Zhang2005`.
    Thus, this method is better suited for structurally homologous pairs in the
    *twilight zone*, i.e. with low amino acid sequence similarity.

    Parameters
    ----------
    fixed : AtomArray, shape(n,)
        The fixed structure.
        Must contain only peptide chains.
    mobile : AtomArray, shape(n,)
        The structure which is superimposed on the `fixed` structure.
        Must contain only peptide chains.
        Must contain the same number of chains as `fixed`.
    structural_alphabet : {"3di", "pb"}, optional
        The structural alphabet to use for finding corresponding residues using sequence
        alignment.
        Either *3Di* or *Protein Blocks*.
    substitution_matrix : SubstitutionMatrix, optional
        The substitution matrix to use for finding corresponding residues using sequence
        alignment.
    max_iterations : int, optional
        The maximum number of iterations to perform in the last step.
    reference_length : int or {"shorter", "longer", "reference"}
        The reference length used to normalize the TM-score and to compute :math:`d_0`.
        If "shorter", the number of residues in the smaller structure is used.
        If "longer", the number of residues in the larger structure is used.
        If "reference", the number of residues in the fixed structure is used.
        The length can also be provided directly as an integer.

    Returns
    -------
    fitted : AtomArray or AtomArrayStack
        A copy of the `mobile` structure, superimposed on the fixed structure.
    transform : AffineTransformation
        This object contains the affine transformation(s) that were
        applied on `mobile`.
        :meth:`AffineTransformation.apply()` can be used to transform
        another AtomArray in the same way.
    fixed_indices, mobile_indices : ndarray, shape(k,), dtype=int
        The indices of the corresponding ``CA`` atoms in the fixed and mobile structure,
        respectively.
        These atoms were used for the superimposition, if their pairwise distance is
        below the :math:`d_0` threshold :footcite:`Zhang2004`.

    See Also
    --------
    superimpose_homologs : Analogous functionality for structures with high sequence similarity.

    Notes
    -----
    The challenge of aligning two structures with different number of residues is
    finding the corresponding residues between them.
    This algorithm inspired by *TM-align* :footcite:`Zhang2005` uses a 3 step heuristic:

    1. Find corresponding residues using a structural alphabet alignment and superimpose
       the chains based on them.
    2. Refine the corresponding residues using a sequence alignment based on a hybrid
       positional substitution matrix:
       The scores are a 50/50 combination of the structural alphabet substitution score
       and the distance-based TM-score between two residues.
       The superimposition is updated based on the new corresponding residues.
    3. Refine the corresponding residues using a sequence alignment with a pure
       TM-score based positional substitution matrix.
       Update the superimposition based on the new corresponding residues.
       Repeat this step until the correspondences are stable.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> fixed = atom_array_stack[0]
    >>> mobile = atom_array_stack[1]
    >>> superimposed, _, fix_indices, mob_indices = superimpose_structural_homologs(
    ...     fixed, mobile, max_iterations=1
    ... )
    >>> print(tm_score(fixed, superimposed, fix_indices, mob_indices))
    0.69...
    >>> print(rmsd(fixed[fix_indices], superimposed[mob_indices]))
    0.83...
    """
    # Avoid circular imports
    from biotite.structure.alphabet.i3d import to_3di
    from biotite.structure.alphabet.pb import to_protein_blocks

    match structural_alphabet.lower():
        case "3di":
            conversion_function = to_3di
            if substitution_matrix is None:
                substitution_matrix = SubstitutionMatrix.std_3di_matrix()
        case "pb":
            conversion_function = to_protein_blocks
            if substitution_matrix is None:
                substitution_matrix = SubstitutionMatrix.std_protein_blocks_matrix()
        case _:
            raise ValueError(
                f"Unsupported structural alphabet: '{structural_alphabet}'"
            )

    # Concatenate the structural sequences for simplicity
    # In the the sequence alignment, this will make barely a difference compared
    # to separate alignments, as there is no gap extension penalty
    fixed_seq = _concatenate_sequences(conversion_function(fixed)[0])
    mobile_seq = _concatenate_sequences(conversion_function(mobile)[0])
    fixed_ca_coord = coord_for_atom_name_per_residue(fixed, ["CA"])[0]
    mobile_ca_coord = coord_for_atom_name_per_residue(mobile, ["CA"])[0]
    # NaN values (i.e. residues without CA atom) would let the superimposition fail
    fixed_not_nan_mask = ~np.isnan(fixed_ca_coord).any(axis=-1)
    mobile_not_nan_mask = ~np.isnan(mobile_ca_coord).any(axis=-1)
    fixed_seq = fixed_seq[fixed_not_nan_mask]
    fixed_ca_coord = fixed_ca_coord[fixed_not_nan_mask]
    mobile_seq = mobile_seq[mobile_not_nan_mask]
    mobile_ca_coord = mobile_ca_coord[mobile_not_nan_mask]
    reference_length = _get_reference_length(
        reference_length, len(fixed_seq), len(mobile_seq)
    )

    # 1. step
    anchors = _find_anchors_structure_based(fixed_seq, mobile_seq, substitution_matrix)
    _, transform = superimpose(
        *_filter_by_anchors(fixed_ca_coord, mobile_ca_coord, anchors)
    )
    superimposed_ca_coord = transform.apply(mobile_ca_coord)

    # 2. step
    anchors = _find_anchors_hybrid(
        fixed_seq,
        mobile_seq,
        fixed_ca_coord,
        superimposed_ca_coord,
        substitution_matrix,
        reference_length,
    )
    _, transform = superimpose(
        *_filter_by_anchors(
            fixed_ca_coord,
            mobile_ca_coord,
            anchors,
        )
    )
    superimposed_ca_coord = transform.apply(mobile_ca_coord)

    # 3. step
    for n_iterations in itertools.count(1):
        previous_anchors = anchors
        anchors = _find_anchors_tm_based(
            fixed_ca_coord, superimposed_ca_coord, reference_length
        )
        _, transform = superimpose(
            *_filter_by_anchors(
                fixed_ca_coord,
                mobile_ca_coord,
                anchors,
                superimposed_ca_coord,
                reference_length,
            )
        )
        superimposed_ca_coord = transform.apply(mobile_ca_coord)
        if n_iterations >= max_iterations or np.array_equal(previous_anchors, anchors):
            break

    # The anchors currently refer to the CA atoms only
    # -> map anchors back to all-atom indices
    fixed_anchors = np.where(fixed.atom_name == "CA")[0][anchors[:, 0]]
    mobile_anchors = np.where(mobile.atom_name == "CA")[0][anchors[:, 1]]
    return transform.apply(mobile), transform, fixed_anchors, mobile_anchors


def _concatenate_sequences(sequences):
    """
    Concatenate the sequences into a single sequence.

    Parameters
    ----------
    sequences : list of Sequence
        The sequences to concatenate.

    Returns
    -------
    sequence : Sequence
        The concatenated sequence.
    """
    # Start with an empty sequence of the same type
    return sum(sequences, start=type(sequences[0])())


def _filter_by_anchors(
    fixed_ca_coord,
    mobile_ca_coord,
    anchors,
    superimposed_ca_coord=None,
    reference_length=None,
):
    """
    Filter the coordinates by the anchor indices.

    Parameters
    ----------
    fixed_ca_coord, mobile_ca_coord : ndarray, shape=(n,3)
        The coordinates of the CA atoms of the fixed and mobile structure,
        respectively.
    anchors : ndarray, shape=(k,2)
        The anchor indices.
    superimposed_ca_coord : ndarray, shape=(m,3), optional
        The coordinates of the CA atoms of the superimposed structure.
        If given, the anchors are additionally filtered by the distance between the
        fixed and superimposed structure, which must be lower than :math:`d_0`.
    reference_length : int, optional
        The reference length used to compute :math:`d_0`.
        Needs to be given if `superimposed_ca_coord` is given.

    Returns
    -------
    anchor_fixed_coord, anchor_mobile_coord : ndarray, shape=(k,3)
        The anchor coordinates of the fixed and mobile structure.
    """
    anchor_fixed_coord = fixed_ca_coord[anchors[:, 0]]
    anchor_mobile_coord = mobile_ca_coord[anchors[:, 1]]
    if reference_length is not None and superimposed_ca_coord is not None:
        anchor_superimposed_coord = superimposed_ca_coord[anchors[:, 1]]
        mask = _mask_by_d0_threshold(
            anchor_fixed_coord, anchor_superimposed_coord, reference_length
        )
        anchor_fixed_coord = anchor_fixed_coord[mask]
        anchor_mobile_coord = anchor_mobile_coord[mask]
    return anchor_fixed_coord, anchor_mobile_coord


def _find_anchors_structure_based(fixed_seq, mobile_seq, substitution_matrix):
    alignment = align_optimal(
        fixed_seq,
        mobile_seq,
        substitution_matrix,
        gap_penalty=(-_get_median_match_score(substitution_matrix), 0),
        terminal_penalty=False,
        max_number=1,
    )[0]
    # Cannot anchor gaps
    alignment = remove_gaps(alignment)
    # Anchors must be structurally similar
    alignment_codes = get_codes(alignment)
    score_matrix = substitution_matrix.score_matrix()
    anchor_mask = score_matrix[alignment_codes[0], alignment_codes[1]] > 0
    anchors = alignment.trace[anchor_mask]
    return anchors


def _find_anchors_hybrid(
    fixed_seq,
    mobile_seq,
    fixed_ca_coord,
    mobile_ca_coord,
    substitution_matrix,
    reference_length,
):
    # Bring substitution scores into the range of pairwise TM scores
    scale_factor = _get_median_match_score(substitution_matrix)
    # Create positional substitution matrix to be able to add the TM-score to it:
    # The TM-score is based on the coordinates of a particular residue and not on the
    # general symbol in the structural alphabet
    # Hence, the shape of the substitution matrix must reflect the number of residues
    # instead of the number of symbols in the structural alphabet
    positional_matrix, fixed_seq, mobile_seq = substitution_matrix.as_positional(
        fixed_seq,
        mobile_seq,
    )

    tm_score_matrix = _pairwise_tm_score(
        fixed_ca_coord, mobile_ca_coord, reference_length
    )
    sa_score_matrix = positional_matrix.score_matrix()
    # Scale the score matrix and the gap penalty to avoid rounding errors
    # when the score matrix is converted to integer type
    hybrid_score_matrix = _SCORE_SCALING * (
        sa_score_matrix / scale_factor + tm_score_matrix
    )
    gap_penalty = _SCORE_SCALING * _HYBRID_PENALTY
    hybrid_matrix = SubstitutionMatrix(
        positional_matrix.get_alphabet1(),
        positional_matrix.get_alphabet2(),
        hybrid_score_matrix.astype(np.int32),
    )
    alignment = align_optimal(
        fixed_seq,
        mobile_seq,
        hybrid_matrix,
        (gap_penalty, 0),
        terminal_penalty=False,
        max_number=1,
    )[0]
    alignment = remove_gaps(alignment)
    anchors = alignment.trace
    return anchors


def _find_anchors_tm_based(fixed_ca_coord, mobile_ca_coord, reference_length):
    # The substitution matrix is positional -> Any positional sequence suffices
    fixed_seq = PurePositionalSequence(len(fixed_ca_coord))
    mobile_seq = PurePositionalSequence(len(mobile_ca_coord))
    tm_score_matrix = _SCORE_SCALING * _pairwise_tm_score(
        fixed_ca_coord, mobile_ca_coord, reference_length
    )
    gap_penalty = _SCORE_SCALING * _TM_GAP_PENALTY
    matrix = SubstitutionMatrix(
        fixed_seq.alphabet,
        mobile_seq.alphabet,
        tm_score_matrix.astype(np.int32),
    )
    alignment = align_optimal(
        fixed_seq,
        mobile_seq,
        matrix,
        (gap_penalty, 0),
        terminal_penalty=False,
        max_number=1,
    )[0]
    alignment = remove_gaps(alignment)
    anchors = alignment.trace
    return anchors


def _get_median_match_score(substitution_matrix):
    """
    Get the median score of two symbols matching.

    Parameters
    ----------
    substitution_matrix : SubstitutionMatrix
        The substitution matrix to get the median match score from.
        Must be symmetric.

    Returns
    -------
    score : int
        The median match score.

    Notes
    -----
    The median is used instead of the mean, as the score range can be quite large,
    especially when the matrix assigns an arbitrary score to the *undefined symbol*.
    Furthermore, this ensures that the return value is an integer, which is required
    for using it as gap penalty.
    """
    return np.median(np.diagonal(substitution_matrix.score_matrix()))


def _mask_by_d0_threshold(fixed_ca_coord, mobile_ca_coord, reference_length):
    """
    Mask every pairwise distance below the :math:`d_0` threshold.

    Parameters
    ----------
    fixed_ca_coord, mobile_ca_coord : ndarray, shape=(n,3)
        The coordinates of the CA atoms of the fixed and mobile structure whose distance
        is measured.
    reference_length : int
        The reference length used to compute :math:`d_0`.

    Returns
    -------
    mask : ndarray, shape=(n,), dtype=bool
        A boolean mask that indicates which distances are below the :math:`d_0`
        threshold.
    """
    mask = distance(fixed_ca_coord, mobile_ca_coord) < _d0(reference_length)
    if not np.any(mask):
        raise ValueError("No anchors found, the structures are too dissimilar")
    return mask


def _pairwise_tm_score(reference_coord, subject_coord, reference_length):
    """
    Compute the TM score for the Cartesian product of two coordinate arrays.

    Parameters
    ----------
    reference_coord, subject_coord : ndarray, shape=(p,3) or shape=(q,3), dtype=float
        The coordinates of the CA atoms to compute all pairwise distances between.
    reference_length : int
        The reference length used to compute :math:`d_0`.

    Returns
    -------
    tm_score : ndarray, shape=(p,q), dtype=float
        The TM score for the Cartesian product of the two coordinate arrays.
    """
    distance_matrix = distance(
        reference_coord[:, np.newaxis, :],
        subject_coord[np.newaxis, :, :],
    )
    return _tm_score(distance_matrix, reference_length)


def _tm_score(distance, reference_length):
    """
    Compute the TM score for the given distances.

    Parameters
    ----------
    distance : float or ndarray
        The distance(s) between the CA atoms of two residues.
    reference_length : int
        The reference length used to compute :math:`d_0`.

    Returns
    -------
    tm_score : float or ndarray
        The TM score for the given distances.
    """
    return 1 / (1 + (distance / _d0(reference_length)) ** 2)


def _d0(reference_length):
    """
    Compute the :math:`d_0` threshold.

    Parameters
    ----------
    reference_length : int
        The reference length used to compute :math:`d_0`.

    Returns
    -------
    d0 : float
        The :math:`d_0` threshold.
    """
    # Constants taken from Zhang2004
    return max(
        # Avoid complex solutions -> clip to positive values
        # For short sequence lengths _D0_MIN takes precedence anyway
        1.24 * max((reference_length - 15), 0) ** (1 / 3) - 1.8,
        _D0_MIN,
    )


def _get_reference_length(user_parameter, reference_length, subject_length):
    """
    Get the reference length to normalize the TM-score and compute :math:`d_0`.

    Parameters
    ----------
    user_parameter : int or {"shorter", "longer", "reference"}
        The value given by the user via the `reference_length` parameter.
    reference_length, subject_length : int
        The lengths of the reference and subject structure, respectively.
    """
    match user_parameter:
        case "shorter":
            return min(reference_length, subject_length)
        case "longer":
            return max(reference_length, subject_length)
        case "reference":
            return reference_length
        case int(number):
            return number
        case _:
            raise ValueError(f"Unsupported reference length: '{user_parameter}'")
