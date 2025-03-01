# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["standardize_order"]

import warnings
import numpy as np
from biotite.structure.error import BadStructureError
from biotite.structure.info.ccd import get_from_ccd
from biotite.structure.residues import get_residue_starts


def standardize_order(atoms):
    """
    Get an index array for an input :class:`AtomArray` or
    :class:`AtomArrayStack` that reorders the atoms for each residue
    to obtain the standard *RCSB PDB* atom order.

    The standard atom order is determined from the reference residues in
    the official *Chemical Component Dictionary*.
    If a residue of the input structure contains additional atoms that
    are not present in the reference residue, these indices to these
    atoms are appended to the end of the respective residue.
    A example for this are optional hydrogen atoms, that appear due to
    protonation.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        Input structure with atoms that are potentially not in the
        *standard* order.

    Returns
    -------
    indices : ndarray, dtype=int, shape=(n,)
        When this index array is applied on the input `atoms`,
        the atoms for each residue are reordered to obtain the
        standard *RCSB PDB* atom order.

    Raises
    ------
    BadStructureError
        If the input `atoms` have duplicate atoms (same atom name)
        within a residue.

    Examples
    --------

    Use as single residue as example.

    >>> residue = atom_array[atom_array.res_id == 1]
    >>> print(residue)
        A       1  ASN N      N        -8.901    4.127   -0.555
        A       1  ASN CA     C        -8.608    3.135   -1.618
        A       1  ASN C      C        -7.117    2.964   -1.897
        A       1  ASN O      O        -6.634    1.849   -1.758
        A       1  ASN CB     C        -9.437    3.396   -2.889
        A       1  ASN CG     C       -10.915    3.130   -2.611
        A       1  ASN OD1    O       -11.269    2.700   -1.524
        A       1  ASN ND2    N       -11.806    3.406   -3.543
        A       1  ASN H1     H        -8.330    3.957    0.261
        A       1  ASN H2     H        -8.740    5.068   -0.889
        A       1  ASN H3     H        -9.877    4.041   -0.293
        A       1  ASN HA     H        -8.930    2.162   -1.239
        A       1  ASN HB2    H        -9.310    4.417   -3.193
        A       1  ASN HB3    H        -9.108    2.719   -3.679
        A       1  ASN HD21   H       -11.572    3.791   -4.444
        A       1  ASN HD22   H       -12.757    3.183   -3.294

    Reverse the atom array.
    Consequently, this also changes the atom order within the residue.

    >>> reordered = residue[np.arange(len(residue))[::-1]]
    >>> print(reordered)
        A       1  ASN HD22   H       -12.757    3.183   -3.294
        A       1  ASN HD21   H       -11.572    3.791   -4.444
        A       1  ASN HB3    H        -9.108    2.719   -3.679
        A       1  ASN HB2    H        -9.310    4.417   -3.193
        A       1  ASN HA     H        -8.930    2.162   -1.239
        A       1  ASN H3     H        -9.877    4.041   -0.293
        A       1  ASN H2     H        -8.740    5.068   -0.889
        A       1  ASN H1     H        -8.330    3.957    0.261
        A       1  ASN ND2    N       -11.806    3.406   -3.543
        A       1  ASN OD1    O       -11.269    2.700   -1.524
        A       1  ASN CG     C       -10.915    3.130   -2.611
        A       1  ASN CB     C        -9.437    3.396   -2.889
        A       1  ASN O      O        -6.634    1.849   -1.758
        A       1  ASN C      C        -7.117    2.964   -1.897
        A       1  ASN CA     C        -8.608    3.135   -1.618
        A       1  ASN N      N        -8.901    4.127   -0.555

    The order is restored with the exception of the N-terminus protonation.

    >>> restored = reordered[info.standardize_order(reordered)]
    >>> print(restored)
        A       1  ASN N      N        -8.901    4.127   -0.555
        A       1  ASN CA     C        -8.608    3.135   -1.618
        A       1  ASN C      C        -7.117    2.964   -1.897
        A       1  ASN O      O        -6.634    1.849   -1.758
        A       1  ASN CB     C        -9.437    3.396   -2.889
        A       1  ASN CG     C       -10.915    3.130   -2.611
        A       1  ASN OD1    O       -11.269    2.700   -1.524
        A       1  ASN ND2    N       -11.806    3.406   -3.543
        A       1  ASN H2     H        -8.740    5.068   -0.889
        A       1  ASN HA     H        -8.930    2.162   -1.239
        A       1  ASN HB2    H        -9.310    4.417   -3.193
        A       1  ASN HB3    H        -9.108    2.719   -3.679
        A       1  ASN HD21   H       -11.572    3.791   -4.444
        A       1  ASN HD22   H       -12.757    3.183   -3.294
        A       1  ASN H3     H        -9.877    4.041   -0.293
        A       1  ASN H1     H        -8.330    3.957    0.261
    """
    reordered_indices = np.zeros(atoms.array_length(), dtype=int)

    starts = get_residue_starts(atoms, add_exclusive_stop=True)
    for i in range(len(starts) - 1):
        start = starts[i]
        stop = starts[i + 1]

        res_name = atoms.res_name[start]
        chem_comp_atom = get_from_ccd("chem_comp_atom", res_name, "atom_id")
        if chem_comp_atom is None:
            # If the residue is not in the CCD, keep the current order
            warnings.warn(
                f"Residue '{res_name}' is not in the CCD, keeping current atom order"
            )
            reordered_indices[start:stop] = np.arange(start, stop)
            continue

        standard_atom_names = chem_comp_atom.as_array()
        reordered_indices[start:stop] = (
            _reorder(atoms.atom_name[start:stop], standard_atom_names) + start
        )

    return reordered_indices


def _reorder(origin, target):
    """
    Create indices to `origin`, that changes the order of `origin`,
    so that the order is the same as in `target`.

    Indices for elements of `target` that are not in `origin`
    are ignored.
    Indices for elements of `origin` that are not in `target`
    are appended to the end of the returned array.


    Parameters
    ----------
    origin : ndarray, dtype=str
        The atom names to reorder.
    target : ndarray, dtype=str
        The atom names in target order.

    Returns
    -------
    indices : ndarray, dtype=int
        Indices for `origin` that that changes the order of `origin`
        to the order of `target`.
    """
    target_hits, origin_hits = np.where(target[:, np.newaxis] == origin[np.newaxis, :])

    counts = np.bincount(target_hits, minlength=len(target))
    if (counts > 1).any():
        counts = np.bincount(target_hits, minlength=len(target))
        # Identify which atom is duplicate
        duplicate_i = np.where(counts > 1)[0][0]
        duplicate_name = target[duplicate_i]
        raise BadStructureError(
            f"Input structure has duplicate atom '{duplicate_name}'"
        )

    if len(origin_hits) < len(origin):
        # The origin structure has additional atoms
        # to the target structure
        # -> Identify which atoms are missing in the target structure
        # and append these to the end of the residue
        missing_atom_mask = np.bincount(origin_hits, minlength=len(origin)).astype(bool)
        return np.concatenate([origin_hits, np.where(~missing_atom_mask)[0]])
    else:
        return origin_hits
