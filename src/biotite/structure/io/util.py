# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Common functions used by a number of subpackages.
"""

__name__ = "biotite.structure.io"
__author__ = "Patrick Kunzmann"
__all__ = [
    "number_of_integer_digits",
    "convert_unicode_to_uint32",
    "convert_uint32_to_unicode",
    "merge_special_symmetry",
]

from collections.abc import Iterable
from typing import Any
import numpy as np
from biotite.structure.atoms import AtomArray, AtomArrayStack
from biotite.structure.bonds import BondList
from biotite.structure.residues import get_residue_starts
from biotite.typing import N, NDArray1, NDArray2

# Maximum separation (in Angstrom) of equivalent atoms at a special position.
SPECIAL_SYMMETRY_DISTANCE = 0.2


def number_of_integer_digits(values: NDArray1[Any, np.number]) -> int:
    """
    Get the maximum number of characters needed to represent the
    pre-decimal positions of the given numeric values.

    Parameters
    ----------
    values : ndarray, dtype=number
        The values to be checked.

    Returns
    -------
    n_digits : int
        The maximum number of characters needed to represent the
        pre-decimal positions of the given numeric values.
    """
    if len(values) == 0:
        return 0
    values = values.astype(int, copy=False)
    n_digits = 0
    n_digits = max(n_digits, len(str(np.min(values))))
    n_digits = max(n_digits, len(str(np.max(values))))
    return n_digits


def convert_unicode_to_uint32(
    array: NDArray1[N, np.str_],
) -> NDArray2[N, Any, np.uint32]:
    """
    Convert a unicode string array into a 2D uint32 array.

    The second dimension corresponds to the character position within a
    string.

    Parameters
    ----------
    array : ndarray, dtype=unicode
        The unicode string array to be converted.

    Returns
    -------
    array : ndarray, dtype=uint32
        The converted unicode string array.
        Each element along the second dimension corresponds to an UTF-32 symbol.
    """
    dtype = array.dtype
    if not np.issubdtype(dtype, np.str_):
        raise TypeError("Expected unicode string array")
    length = array.shape[0]
    n_char = dtype.itemsize // 4
    return np.frombuffer(array, dtype=np.uint32).reshape(length, n_char)  # pyright: ignore[reportReturnType]


def convert_uint32_to_unicode(
    array: NDArray2[N, Any, np.uint32],
) -> NDArray1[N, np.str_]:
    """
    Convert a 2D uint32 array into a unicode string array.

    This is the inverse of :func:`convert_unicode_to_uint32`.

    Parameters
    ----------
    array : ndarray, dtype=uint32, shape=(n, m)
        The uint32 array to be converted.
        Each element along the second dimension corresponds to an UTF-32 symbol.

    Returns
    -------
    array : ndarray, dtype=unicode
        The converted unicode string array.
    """
    if array.ndim != 2:
        raise ValueError("Expected 2D array")
    length, n_char = array.shape
    return np.frombuffer(
        array.astype(np.uint32, copy=False).tobytes(), dtype=f"<U{n_char}"
    )


def merge_special_symmetry(
    atoms: AtomArray[Any] | AtomArrayStack[Any, Any],
    residue_masks: Iterable[NDArray1[Any, np.bool_]],
) -> AtomArray[Any] | AtomArrayStack[Any, Any]:
    """
    Merge overlapping atoms and residues at special symmetry positions.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The structure containing symmetry copies.
        Must have a ``sym_id`` annotation identifying each copy.
        Alternate locations must already be filtered to one conformer.
    residue_masks : iterable of ndarray, shape=(n,), dtype=bool
        Each mask selects all symmetry copies of one residue at a special
        position. The masks refer to the original atom order in `atoms`.

    Returns
    -------
    merged : AtomArray or AtomArrayStack
        The structure with overlapping atoms merged into the copy with the
        lowest symmetry ID. Distinct atoms from overlapping residue copies
        are retained and grouped into a single residue with that symmetry ID.
        The input structure is not modified.

    Notes
    -----
    Atoms are merged only if they belong to different symmetry copies,
    have the same element, and are separated by less than 0.2 Angstrom.
    For an :class:`AtomArrayStack`, they must overlap in every model.

    The coordinates and annotations of the retained atom are preserved,
    except that occupancies are summed and capped at 1 when present.
    Bonds to removed atoms are redirected to the retained atoms, and
    self-bonds are removed. Atom names are preserved, so residues completed
    by symmetry may contain repeated atom names.
    """
    order = np.argsort(atoms.sym_id, kind="stable")
    atoms = atoms[..., order]
    atom_indices = np.arange(atoms.array_length())
    replacements = atom_indices.copy()
    residue_starts = get_residue_starts(atoms)
    residue_indices = np.searchsorted(residue_starts, atom_indices, side="right") - 1
    residue_parents = np.arange(len(residue_starts))
    for matches in residue_masks:
        indices = np.flatnonzero(matches[order])
        for kept in indices:
            if replacements[kept] != kept:
                continue
            candidates = indices[
                (atoms.sym_id[indices] > atoms.sym_id[kept])
                & (replacements[indices] == indices)
                & (atoms.element[indices] == atoms.element[kept])
            ]
            distances = np.linalg.norm(
                atoms.coord[..., candidates, :] - atoms.coord[..., kept, None, :],
                axis=-1,
            )
            overlap = distances < SPECIAL_SYMMETRY_DISTANCE
            if isinstance(atoms, AtomArrayStack):
                overlap = overlap.all(axis=0)
            for removed in candidates[overlap]:
                replacements[removed] = kept
                # Union the residues, so the remaining atoms form one residue.
                first = residue_indices[kept]
                second = residue_indices[removed]
                while residue_parents[first] != first:
                    first = residue_parents[first]
                while residue_parents[second] != second:
                    second = residue_parents[second]
                residue_parents[max(first, second)] = min(first, second)
    if np.array_equal(replacements, atom_indices):
        return atoms

    if "occupancy" in atoms.get_annotation_categories():
        occupancies = np.zeros(atoms.array_length())
        np.add.at(occupancies, replacements, atoms.occupancy)
        atoms.occupancy = np.minimum(occupancies, 1.0)
    if atoms.bonds is not None:
        bonds = atoms.bonds.as_array()
        bonds[:, :2] = replacements[bonds[:, :2]]
        bonds = bonds[bonds[:, 0] != bonds[:, 1]]
        atoms.bonds = BondList(atoms.array_length(), bonds)
    for residue in range(len(residue_parents)):
        while residue_parents[residue] != residue_parents[residue_parents[residue]]:
            residue_parents[residue] = residue_parents[residue_parents[residue]]
    owners = residue_starts[residue_parents[residue_indices]]
    atoms.sym_id = atoms.sym_id[owners]
    # Place the distinct atoms from each merged residue next to its retained copy.
    order = np.argsort(owners, kind="stable")
    return atoms[..., order[replacements[order] == order]]
