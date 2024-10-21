"""
Superimposition of remotely homologous structures
=================================================
"""

import biotite.sequence.align as align
import biotite.structure as struc
import numpy as np
from numpy.typing import NDArray

from geotite.positional_matrix import transform_into_positional
from geotite.protein_blocks import convert_to_pb, pb_substitution_matrix

PB_GAP_PENALTY = (-10, -1)


def _tm_score(
    distance: NDArray[np.double] | float, target_length: int
) -> NDArray[np.double] | float:
    """
    Compute the TM score for given distances.

    Parameters
    ----------
    distance : float or ndarray
        The distance(s) between the CA atoms of two residues.
    target_length : int
        The number of residues in the target structure.
    """
    d_0 = 1.24 * (target_length - 15) ** (1 / 3) - 1.8
    return 1 / (1 + (distance / d_0) ** 2)


def _find_matching_anchors(
    fixed: struc.AtomArray, mobile: struc.AtomArray, include_ca_distances: bool = False
) -> NDArray[np.int_]:
    """
    Find corresponding CA atoms in two protein chains.
    They are found via a sequence alignment on a structural alphabet (Protein Blocks).
    Optionally the superimposition matrix can be include position specific CA distances
    by means of a TM-score.
    """
    fixed_seq = convert_to_pb(fixed)
    mobile_seq = convert_to_pb(mobile)

    pb_matrix = pb_substitution_matrix()
    matrix, fixed_seq, mobile_seq = transform_into_positional(
        pb_matrix,
        fixed_seq,
        mobile_seq,
    )
    if include_ca_distances:
        fixed_ca = fixed[_get_backbone_anchor_indices(fixed)]
        mobile_ca = mobile[_get_backbone_anchor_indices(mobile)]
        ca_distance_matrix = struc.distance(
            fixed_ca.coord[:, np.newaxis, :],
            mobile_ca.coord[np.newaxis, :, :],
        )
        tm_score_matrix = _tm_score(ca_distance_matrix, len(fixed_seq))
        # Bring TM scores into same order of magnitude as substitution scores
        scaling = np.max(pb_matrix.score_matrix()) - np.min(pb_matrix.score_matrix())
        tm_score_matrix *= scaling
        matrix = align.SubstitutionMatrix(
            matrix.get_alphabet1(),
            matrix.get_alphabet2(),
            matrix.score_matrix() + tm_score_matrix,
        )

    alignment = align.align_optimal(
        fixed_seq,
        mobile_seq,
        matrix,
        gap_penalty=PB_GAP_PENALTY,
        terminal_penalty=False,
        max_number=1,
    )[0]
    alignment_codes = align.get_codes(alignment)
    anchor_mask = (
        # Anchors must be similar amino acids
        (matrix.score_matrix()[alignment_codes[0], alignment_codes[1]] > 0)
        # Cannot anchor gaps
        & (alignment_codes[0] != -1)
        & (alignment_codes[1] != -1)
    )
    anchors = alignment.trace[anchor_mask]
    return anchors


def _get_backbone_anchor_indices(atoms: struc.AtomArray) -> NDArray[np.int_]:
    """
    Get the indices of the anchor atom for each residue.
    """
    return np.where(atoms.atom_name == "CA")[0]


def superimpose_homolog_complex(fixed, mobile):
    """
    Superimpose two remotely homologous protein complexes.

    This method runs a sequence alignment on the *3Di* structural alphabet
    to find corresponding residues (anchors) that can be superimposed.
    The actual anchor atoms are the `CA` atoms of each residue.
    Finally, conformational outliers are removed.

    Parameters
    ----------
    fixed : AtomArray, shape(n,)
        The fixed structure.
        Must comprise a protein complex containing two chains.
        Each residue must contain the complete backbone (`N`, `CA`, `C`).
    mobile : AtomArray, shape(n,)
        The structure which is superimposed on the `fixed` structure.
        Must comprise a protein complex containing two chains.
        Each residue must contain the complete backbone (`N`, `CA`, `C`).

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

    See also
    --------
    biotite.structure.superimpose_homologs
        Analogous functionality for structures with high sequence similarity.

    Notes
    -----
    The anchors are found via

    1. a *3Di* sequence alignment
    2. a superimposition of the chains based on the found anchors
    3. a refined alignment that includes that includes both the structural alphabet and
       a CA-distance based score (TM-score)

    """
    # Find anchors for each chain pair separately
    fixed_receptor, fixed_ligand = _get_receptor_and_ligand(fixed)
    mobile_receptor, mobile_ligand = _get_receptor_and_ligand(mobile)

    # First iteration:
    # Align sequences using PB substitution matrix only
    anchor_indices = _find_matching_anchors_for_multiple_chains(
        (fixed_receptor, fixed_ligand),
        (mobile_receptor, mobile_ligand),
    )

    # Second iteration:
    # Refined hybrid alignment that includes a substitution matrix that includes both
    # substitution scores and CA distance penalties
    _, transform, _ = struc.superimpose_without_outliers(
        fixed[..., anchor_indices[:, 0]], mobile[..., anchor_indices[:, 1]], **kwargs
    )
    anchor_indices = _find_matching_anchors_for_multiple_chains(
        (fixed_receptor, fixed_ligand),
        (transform.apply(mobile_receptor), transform.apply(mobile_ligand)),
        include_ca_distances=True,
    )

    fixed_anchor_indices = _get_backbone_anchor_indices(fixed)[anchor_indices[:, 0]]
    mobile_anchor_indices = _get_backbone_anchor_indices(mobile)[anchor_indices[:, 1]]
    if restrict_to_contacts:
        # Only keep anchors that are close to the interface in either structure
        contact_mask = np.full(len(anchor_indices), False, dtype=bool)
        for chain, indices in [
            (fixed, fixed_anchor_indices),
            (mobile, mobile_anchor_indices),
        ]:
            all_atom_contact_mask = _filter_contacts(chain, contact_threshold)
            residue_masks = struc.get_residue_masks(chain, indices)
            # If least one atom of an anchor residue is in contact,
            # that residue is retained in the mask
            anchor_residue_contact_mask = np.any(
                residue_masks & all_atom_contact_mask, axis=1
            )
            contact_mask |= anchor_residue_contact_mask
        fixed_anchor_indices = fixed_anchor_indices[contact_mask]
        mobile_anchor_indices = mobile_anchor_indices[contact_mask]

    _, transform, selected_anchor_indices = struc.superimpose_without_outliers(
        fixed[..., fixed_anchor_indices], mobile[..., mobile_anchor_indices], **kwargs
    )
    fixed_anchor_indices = fixed_anchor_indices[selected_anchor_indices]
    mobile_anchor_indices = mobile_anchor_indices[selected_anchor_indices]

    return (
        transform.apply(mobile),
        transform,
        fixed_anchor_indices,
        mobile_anchor_indices,
    )