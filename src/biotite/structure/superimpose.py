# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for structure superimposition.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann, Claude J. Rogers"
__all__ = [
    "superimpose",
    "superimpose_homologs",
    "superimpose_without_outliers",
]


import numpy as np
from biotite.sequence.align.alignment import get_codes, remove_gaps
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.align.pairwise import align_optimal
from biotite.sequence.alphabet import common_alphabet
from biotite.sequence.seqtypes import ProteinSequence
from biotite.structure.atoms import coord
from biotite.structure.chains import chain_iter
from biotite.structure.filter import filter_amino_acids, filter_nucleotides
from biotite.structure.geometry import centroid, distance
from biotite.structure.sequence import to_sequence
from biotite.structure.transform import AffineTransformation


def superimpose(fixed, mobile, atom_mask=None):
    """
    Superimpose structures onto each other, minimizing the RMSD between
    them.
    :footcite:`Kabsch1976, Kabsch1978`.

    More precisely, the `mobile` structure is rotated and translated onto
    the `fixed` structure.

    Parameters
    ----------
    fixed : AtomArray, shape(n,) or AtomArrayStack, shape(m,n) or ndarray, shape(n,), dtype=float or ndarray, shape(m,n), dtype=float
        The fixed structure(s).
        Alternatively coordinates can be given.
    mobile : AtomArray, shape(n,) or AtomArrayStack, shape(m,n) or ndarray, shape(n,), dtype=float or ndarray, shape(m,n), dtype=float
        The structure(s) which is/are superimposed on the `fixed`
        structure.
        Each atom at index *i* in `mobile` must correspond the
        atom at index *i* in `fixed` to obtain correct results.
        Furthermore, if both `fixed` and `mobile` are
        :class:`AtomArrayStack` objects, they must have the same
        number of models.
        Alternatively coordinates can be given.
    atom_mask : ndarray, dtype=bool, optional
        If given, only the atoms covered by this boolean mask will be
        considered for superimposition.
        This means that the algorithm will minimize the RMSD based
        on the covered atoms instead of all atoms.
        The returned superimposed structure will contain all atoms
        of the input structure, regardless of this parameter.

    Returns
    -------
    fitted : AtomArray or AtomArrayStack or ndarray, shape(n,), dtype=float or ndarray, shape(m,n), dtype=float
        A copy of the `mobile` structure(s),
        superimposed on the fixed structure(s).
        Only coordinates are returned, if coordinates were given in
        `mobile`.
    transformation : AffineTransformation
        The affine transformation(s) that were applied on `mobile`.
        :meth:`AffineTransformation.apply()` can be used to transform
        another AtomArray in the same way.

    See Also
    --------
    superimpose_without_outliers : Superimposition with outlier removal.
    superimpose_homologs : Superimposition of homologous structures.

    Notes
    -----
    The `transformation` can come in handy, in case you want to
    superimpose two
    structures with different amount of atoms.
    Often the two structures need to be filtered in order to obtain the
    same size and annotation arrays.
    After superimposition the transformation can be applied on the
    original structure using :meth:`AffineTransformation.apply()`.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    At first two models of a structure are taken and one of them is
    randomly rotated/translated.
    Consequently the RMSD is quite large:

    >>> array1 = atom_array_stack[0]
    >>> array2 = atom_array_stack[1]
    >>> array2 = translate(array2, [1,2,3])
    >>> array2 = rotate(array2, [1,2,3])
    >>> print("{:.3f}".format(rmsd(array1, array2)))
    11.260

    RMSD decreases after superimposition of only CA atoms:

    >>> array2_fit, transformation = superimpose(
    ...     array1, array2, atom_mask=(array2.atom_name == "CA")
    ... )
    >>> print("{:.3f}".format(rmsd(array1, array2_fit)))
    1.961

    RMSD is even lower when all atoms are considered in the
    superimposition:

    >>> array2_fit, transformation = superimpose(array1, array2)
    >>> print("{:.3f}".format(rmsd(array1, array2_fit)))
    1.928
    """
    # Bring coordinates into the same dimensionality
    mob_coord = _reshape_to_3d(coord(mobile))
    fix_coord = _reshape_to_3d(coord(fixed))

    if atom_mask is not None:
        # Implicitly this creates array copies
        mob_filtered = mob_coord[:, atom_mask, :]
        fix_filtered = fix_coord[:, atom_mask, :]
    else:
        mob_filtered = np.copy(mob_coord)
        fix_filtered = np.copy(fix_coord)

    # Center coordinates at (0,0,0)
    mob_centroid = centroid(mob_filtered)
    fix_centroid = centroid(fix_filtered)
    mob_centered_filtered = mob_filtered - mob_centroid[:, np.newaxis, :]
    fix_centered_filtered = fix_filtered - fix_centroid[:, np.newaxis, :]

    rotation = _get_rotation_matrices(fix_centered_filtered, mob_centered_filtered)
    transform = AffineTransformation(-mob_centroid, rotation, fix_centroid)
    return transform.apply(mobile), transform


def superimpose_without_outliers(
    fixed,
    mobile,
    min_anchors=3,
    max_iterations=10,
    quantiles=(0.25, 0.75),
    outlier_threshold=1.5,
):
    r"""
    Superimpose structures onto a fixed structure, ignoring
    conformational outliers.

    This method iteratively superimposes the `mobile` structure onto the
    `fixed` structure, removes conformational outliers and superimposes
    the remaining atoms (called *anchors*) again until no outlier
    remains.

    Parameters
    ----------
    fixed : AtomArray, shape(n,) or AtomArrayStack, shape(m,n) or ndarray, shape(n,), dtype=float or ndarray, shape(m,n), dtype=float
        The fixed structure(s).
        Alternatively coordinates can be given.
    mobile : AtomArray, shape(n,) or AtomArrayStack, shape(m,n) or ndarray, shape(n,), dtype=float or ndarray, shape(m,n), dtype=float
        The structure(s) which is/are superimposed on the `fixed`
        structure.
        Each atom at index *i* in `mobile` must correspond the
        atom at index *i* in `fixed` to obtain correct results.
        Furthermore, if both `fixed` and `mobile` are
        :class:`AtomArrayStack` objects, they must have the same
        number of models.
        Alternatively coordinates can be given.
    min_anchors : int, optional
        The outlier removal is stopped, if less than `min_anchors`
        anchors would be left.
    max_iterations : int, optional
        The maximum number of iterations for removing conformational
        outliers.
        Setting the value to 1 means that no outlier removal is
        conducted.
    quantiles : tuple (float, float), optional
        The lower and upper quantile for the interpercentile range
        (IPR).
        By default the interquartile range is taken.
    outlier_threshold : float, optional
        The threshold for considering a conformational outlier.
        The threshold is given in units of IPR.

    Returns
    -------
    fitted : AtomArray or AtomArrayStack
        A copy of the `mobile` structure(s), superimposed on the fixed
        structure.
        Only coordinates are returned, if coordinates were given in
        `mobile`.
    transform : AffineTransformation
        This object contains the affine transformation(s) that were
        applied on `mobile`.
        :meth:`AffineTransformation.apply()` can be used to transform
        another AtomArray in the same way.
    anchor_indices : ndarray, shape(k,), dtype=int
        The indices of the anchor atoms.
        These atoms were used for the superimposition.

    See Also
    --------
    superimpose : Superimposition without outlier removal.
    superimpose_homologs : Superimposition of homologous structures.

    Notes
    -----
    This method runs the following algorithm in iterations:

    1. Superimpose anchor atoms of `mobile` onto `fixed`.
    2. Calculate the squared distance :math:`d^2` between the
       superimposed anchors.
    3. Remove conformational outliers from anchors based on the
       following criterion:

       .. math:: d^2 > P_\text{upper}(d^2) + \left( P_\text{upper}(d^2) - P_\text{lower}(d^2) \right) \cdot T

       In prose this means that an anchor is considered an outlier, if
       it is `outlier_threshold` :math:`T` times the interpercentile
       range (IPR) above the upper percentile.
       By default, this is 1.5 times the interquartile range, which is
       the usual threshold to mark outliers in box plots.

    In the beginning, all atoms are considered as anchors.

    Considering all atoms (not only the anchors), this approach does
    **not** minimize the RMSD, in contrast to :func:`superimpose()`.
    The purpose of this function is to ignore outliers to decrease the
    RMSD in the more conserved parts of the structure.
    """
    if max_iterations < 1:
        raise ValueError("Maximum number of iterations must be at least 1")

    # Ensure that the first quantile is smaller than the second one
    quantiles = sorted(quantiles)

    fixed_coord = coord(fixed)
    mobile_coord = coord(mobile)
    # Before refinement, all anchors are included
    # 'inlier' is the opposite of 'outlier'
    updated_inlier_mask = np.ones(fixed_coord.shape[-2], dtype=bool)

    for _ in range(max_iterations):
        # Run superimposition
        inlier_mask = updated_inlier_mask
        filtered_fixed_coord = fixed_coord[..., inlier_mask, :]
        filtered_mobile_coord = mobile_coord[..., inlier_mask, :]
        superimposed_coord, transform = superimpose(
            filtered_fixed_coord, filtered_mobile_coord
        )

        # Find outliers
        sq_dist = distance(filtered_fixed_coord, superimposed_coord) ** 2
        if sq_dist.ndim == 2:
            # If multiple models are superimposed,
            # use the mean squared distance to determine outliers
            sq_dist = np.mean(sq_dist, axis=0)
        lower_quantile, upper_quantile = np.quantile(sq_dist, quantiles)
        ipr = upper_quantile - lower_quantile
        updated_inlier_mask = inlier_mask.copy()
        # Squared distance was only calculated for the existing inliers
        # -> update the mask only for these atoms
        updated_inlier_mask[updated_inlier_mask] = (
            sq_dist <= upper_quantile + outlier_threshold * ipr
        )
        if np.all(updated_inlier_mask):
            # No outliers anymore -> early termination
            break
        if np.count_nonzero(updated_inlier_mask) < min_anchors:
            # Less than min_anchors anchors would be left -> early termination
            break

    anchor_indices = np.where(inlier_mask)[0]
    return transform.apply(mobile), transform, anchor_indices


def superimpose_homologs(
    fixed,
    mobile,
    substitution_matrix=None,
    gap_penalty=-10,
    min_anchors=3,
    terminal_penalty=False,
    **kwargs,
):
    r"""
    Superimpose a protein or nucleotide structure onto another one,
    considering sequence differences and conformational outliers.

    The method finds corresponding residues by sequence alignment and
    selects their :math:`C_{\alpha}` or :math:`P` atoms as
    superimposition *anchors*.
    Then iteratively the anchor atoms are superimposed and outliers are
    removed.

    Parameters
    ----------
    fixed : AtomArray, shape(n,) or AtomArrayStack, shape(m,n)
        The fixed structure(s).
    mobile : AtomArray, shape(n,) or AtomArrayStack, shape(m,n)
        The structure(s) which is/are superimposed on the `fixed` structure.
        Must contain the same number of chains as `fixed` with corresponding chains
        being in the same order.
        The specific chain IDs can be different.
    substitution_matrix : str or SubstitutionMatrix, optional
        The (name of the) substitution matrix used for sequence
        alignment.
        Must fit the chain type.
        By default, ``"BLOSUM62"`` and ``"NUC"`` are used respectively.
        Only aligned residues with a positive score are considered as
        initial anchors.
    gap_penalty : int or tuple of int, optional
        The gap penalty for sequence alignment.
        A single value indicates a linear penalty, while a tuple
        indicates an affine penalty.
    min_anchors : int, optional
        If less than `min_anchors` anchors are found by sequence
        alignment, the method ditches the alignment and matches all
        anchor atoms.
        If the number of anchor atoms is not equal in `fixed` and
        `mobile` in this fallback case, an exception is raised.
        Furthermore, the outlier removal is stopped, if less than
        `min_anchors` anchors would be left.
    terminal_penalty : bool, optional
        If set to true, gap penalties are applied to terminal gaps in the sequence
        alignment.
    **kwargs
        Additional parameters for
        :func:`superimpose_without_outliers()`.

    Returns
    -------
    fitted : AtomArray or AtomArrayStack
        A copy of the `mobile` structure(s), superimposed on the fixed
        structure(s).
    transform : AffineTransformation
        This object contains the affine transformation(s) that were
        applied on `mobile`.
        :meth:`AffineTransformation.apply()` can be used to transform
        another AtomArray in the same way.
    fixed_anchor_indices, mobile_anchor_indices : ndarray, shape(k,), dtype=int
        The indices of the anchor atoms in the fixed and mobile
        structure, respectively.
        These atoms were used for the superimposition.

    See Also
    --------
    superimpose : Superimposition without outlier removal.
    superimpose_without_outliers : Internally used for outlier removal.
    superimpose_structural_homologs : Better suited for low sequence similarity.

    Notes
    -----
    As this method relies on sequence alignment, it works only for
    proteins/nucleic acids with decent sequence homology.
    """
    fixed_anchor_indices = _get_backbone_anchor_indices(fixed)
    mobile_anchor_indices = _get_backbone_anchor_indices(mobile)
    if (
        len(fixed_anchor_indices) < min_anchors
        or len(mobile_anchor_indices) < min_anchors
    ):
        raise ValueError(
            "Structures have too few backbone atoms for required number of anchors"
        )

    anchor_indices = _find_matching_anchors(
        fixed[..., fixed_anchor_indices],
        mobile[..., mobile_anchor_indices],
        substitution_matrix,
        gap_penalty,
        terminal_penalty,
    )
    if len(anchor_indices) < min_anchors:
        # Fallback: Match all backbone anchors
        if len(fixed_anchor_indices) != len(mobile_anchor_indices):
            raise ValueError(
                "Tried fallback due to low anchor number, "
                "but number of backbone atoms does not match"
            )
        fixed_anchor_indices = fixed_anchor_indices
        mobile_anchor_indices = mobile_anchor_indices
    else:
        # The anchor indices point to the backbone atoms
        # -> get the corresponding indices for the whole structure
        fixed_anchor_indices = fixed_anchor_indices[anchor_indices[:, 0]]
        mobile_anchor_indices = mobile_anchor_indices[anchor_indices[:, 1]]

    _, transform, selected_anchor_indices = superimpose_without_outliers(
        fixed[..., fixed_anchor_indices],
        mobile[..., mobile_anchor_indices],
        min_anchors,
        **kwargs,
    )
    fixed_anchor_indices = fixed_anchor_indices[selected_anchor_indices]
    mobile_anchor_indices = mobile_anchor_indices[selected_anchor_indices]

    return (
        transform.apply(mobile),
        transform,
        fixed_anchor_indices,
        mobile_anchor_indices,
    )


def _reshape_to_3d(coord):
    """
    Reshape the coordinate array to 3D, if it is 2D.
    """
    if coord.ndim < 2:
        raise ValueError("Coordinates must be at least two-dimensional")
    if coord.ndim == 2:
        return coord[np.newaxis, ...]
    elif coord.ndim == 3:
        return coord
    else:
        raise ValueError("Coordinates must be at most three-dimensional")


def _get_rotation_matrices(fixed, mobile):
    """
    Get the rotation matrices to superimpose the given mobile
    coordinates into the given fixed coordinates, minimizing the RMSD.

    Uses the *Kabsch* algorithm.
    Both sets of coordinates must already be centered at origin.
    """
    # Calculate cross-covariance matrices
    cov = np.sum(fixed[:, :, :, np.newaxis] * mobile[:, :, np.newaxis, :], axis=1)
    v, s, w = np.linalg.svd(cov)
    # Remove possibility of reflected atom coordinates
    reflected_mask = np.linalg.det(v) * np.linalg.det(w) < 0
    v[reflected_mask, :, -1] *= -1
    matrices = np.matmul(v, w)
    return matrices


def _get_backbone_anchor_indices(atoms):
    """
    Select one representative anchor atom for each amino acid and
    nucleotide and return their indices.
    """
    return np.where(
        ((filter_amino_acids(atoms)) & (atoms.atom_name == "CA"))
        | ((filter_nucleotides(atoms)) & (atoms.atom_name == "P"))
    )[0]


def _find_matching_anchors(
    fixed_anchor_atoms,
    mobile_anchors_atoms,
    substitution_matrix,
    gap_penalty,
    terminal_penalty,
):
    """
    Find corresponding residues using pairwise sequence alignment.
    """
    anchor_list = []
    fixed_seq_offset = 0
    mobile_seq_offset = 0
    for fixed_chain, mobile_chain in zip(
        chain_iter(fixed_anchor_atoms), chain_iter(mobile_anchors_atoms), strict=True
    ):
        # The input is a single chain -> expect a single sequence
        fixed_seq = to_sequence(fixed_chain, allow_hetero=True)[0][0]
        mobile_seq = to_sequence(mobile_chain, allow_hetero=True)[0][0]

        common_alph = common_alphabet([fixed_seq.alphabet, mobile_seq.alphabet])
        if common_alph is None:
            raise ValueError("Cannot superimpose peptides with nucleic acids")
        if substitution_matrix is None:
            if isinstance(fixed_seq, ProteinSequence):
                substitution_matrix = SubstitutionMatrix.std_protein_matrix()
            else:
                substitution_matrix = SubstitutionMatrix.std_nucleotide_matrix()
        elif isinstance(substitution_matrix, str):
            substitution_matrix = SubstitutionMatrix(
                common_alph, common_alph, substitution_matrix
            )

        alignment = align_optimal(
            fixed_seq,
            mobile_seq,
            substitution_matrix,
            gap_penalty,
            terminal_penalty=terminal_penalty,
            max_number=1,
        )[0]
        # Cannot anchor gaps
        alignment = remove_gaps(alignment)
        ali_codes = get_codes(alignment)
        score_matrix = substitution_matrix.score_matrix()
        # Anchors must be similar amino acids
        anchors = alignment.trace[score_matrix[ali_codes[0], ali_codes[1]] > 0]

        anchors += fixed_seq_offset, mobile_seq_offset
        fixed_seq_offset += len(fixed_seq)
        mobile_seq_offset += len(mobile_seq)
        anchor_list.append(anchors)

    return np.concatenate(anchor_list, axis=0)
