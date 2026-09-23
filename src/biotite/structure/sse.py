"""
This module allows estimation of secondary structure elements in protein
structures.
"""

from __future__ import annotations

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["SecondaryStructure", "annotate_sse"]

from enum import IntEnum
from typing import TypeVar
import numpy as np
from numpy.typing import ArrayLike
from biotite.rust.structure import CellList
from biotite.structure.atoms import AtomArray
from biotite.structure.filter import filter_amino_acids
from biotite.structure.geometry import angle, dihedral, distance
from biotite.structure.integrity import check_res_id_continuity
from biotite.structure.residues import get_residue_starts
from biotite.typing import XYZ, K, N, NDArray1, NDArray2

_r_helix = (np.deg2rad(89 - 12), np.deg2rad(89 + 12))
_a_helix = (np.deg2rad(50 - 20), np.deg2rad(50 + 20))
_d2_helix = ((5.5 - 0.5), (5.5 + 0.5))  # Not used in the algorithm description
_d3_helix = ((5.3 - 0.5), (5.3 + 0.5))
_d4_helix = ((6.4 - 0.6), (6.4 + 0.6))

_r_strand = (np.deg2rad(124 - 14), np.deg2rad(124 + 14))
_a_strand = (np.deg2rad(-180), np.deg2rad(-125), np.deg2rad(145), np.deg2rad(180))
_d2_strand = ((6.7 - 0.6), (6.7 + 0.6))
_d3_strand = ((9.9 - 0.9), (9.9 + 0.9))
_d4_strand = ((12.4 - 1.1), (12.4 + 1.1))


class SecondaryStructure(IntEnum):
    r"""
    The secondary structure elements (SSEs) a residue can be assigned to.

    Each element has a corresponding one-letter `symbol`, that is used
    in text-based representations, e.g. by the original *P-SEA*
    software.

    - ``NONE`` (``''``): The residue is not an amino acid or has no
      ``CA`` atom.
    - ``COIL`` (``'c'``): coil
    - ``HELIX`` (``'a'``): :math:`{\alpha}`-helix
    - ``STRAND`` (``'b'``): :math:`{\beta}`-strand/sheet

    Examples
    --------

    >>> sse = SecondaryStructure.from_symbols(["c", "a", "a", "b", ""])
    >>> print(sse)
    [ 0  1  1  2 -1]
    >>> print(SecondaryStructure.to_symbols(sse))
    ['c' 'a' 'a' 'b' '']
    >>> print(SecondaryStructure.HELIX.symbol)
    a
    >>> print(SecondaryStructure.from_symbol("b").name)
    STRAND
    """

    NONE = -1
    COIL = 0
    HELIX = 1
    STRAND = 2

    @property
    def symbol(self) -> str:
        """
        The one-letter symbol of the secondary structure element.

        Returns
        -------
        symbol : str
            The one-letter symbol.
        """
        return _SSE_SYMBOLS[self]

    @classmethod
    def from_symbol(cls, symbol: str) -> SecondaryStructure:
        """
        Get the secondary structure element from its one-letter symbol.

        Parameters
        ----------
        symbol : str
            The symbol.

        Returns
        -------
        sse : SecondaryStructure
            The corresponding secondary structure element.
        """
        for sse, sse_symbol in _SSE_SYMBOLS.items():
            if sse_symbol == symbol:
                return sse
        raise ValueError(f"Unknown secondary structure symbol '{symbol}'")

    @classmethod
    def from_symbols(cls, symbols: ArrayLike) -> NDArray1[K, np.int_]:
        """
        Convert an array of one-letter symbols into an array of
        secondary structure elements.

        Parameters
        ----------
        symbols : array-like, shape=(k,), dtype=str
            The symbols.

        Returns
        -------
        sse : ndarray, shape=(k,), dtype=int
            The corresponding secondary structure elements.
        """
        return _symbols_to_values(_SSE_SYMBOLS, symbols)

    @classmethod
    def to_symbols(cls, sse: ArrayLike) -> NDArray1[K, np.str_]:
        """
        Convert an array of secondary structure elements into an array
        of one-letter symbols.

        Parameters
        ----------
        sse : array-like, shape=(k,), dtype=int
            The secondary structure elements.

        Returns
        -------
        symbols : ndarray, shape=(k,), dtype=str
            The corresponding symbols.
        """
        return _values_to_symbols(_SSE_SYMBOLS, sse)


_E = TypeVar("_E", bound=IntEnum)

_SSE_SYMBOLS = {
    SecondaryStructure.NONE: "",
    SecondaryStructure.COIL: "c",
    SecondaryStructure.HELIX: "a",
    SecondaryStructure.STRAND: "b",
}


def _symbols_to_values(
    symbol_dict: dict[_E, str], symbols: ArrayLike
) -> NDArray1[K, np.int_]:
    """
    Convert an array of symbols into an array of the corresponding enum values
    using the given mapping.
    """
    symbols = np.asarray(symbols, dtype=str)
    values = np.zeros(symbols.shape, dtype=int)
    known = np.zeros(symbols.shape, dtype=bool)
    for value, symbol in symbol_dict.items():
        mask = symbols == symbol
        values[mask] = value
        known |= mask
    if not known.all():
        unknown = np.unique(symbols[~known]).tolist()
        raise ValueError(f"Unknown symbols {unknown}")
    return values  # pyright: ignore[reportReturnType]


def _values_to_symbols(
    symbol_dict: dict[_E, str], values: ArrayLike
) -> NDArray1[K, np.str_]:
    """
    Convert an array of enum values into an array of the corresponding symbols
    using the given mapping.
    """
    values = np.asarray(values, dtype=int)
    symbols = np.zeros(values.shape, dtype="U1")
    known = np.zeros(values.shape, dtype=bool)
    for value, symbol in symbol_dict.items():
        mask = values == value
        symbols[mask] = symbol
        known |= mask
    if not known.all():
        unknown = np.unique(values[~known]).tolist()
        raise ValueError(f"Unknown values {unknown}")
    return symbols  # pyright: ignore[reportReturnType]


def annotate_sse(atom_array: AtomArray[N]) -> NDArray1[K, np.int_]:
    r"""
    Calculate the secondary structure elements (SSEs) of a
    peptide chain based on the `P-SEA` algorithm.
    :footcite:`Labesse1997`

    The annotation is based CA coordinates only, specifically
    distances and dihedral angles.
    Discontinuities between chains are detected by residue ID.

    Parameters
    ----------
    atom_array : AtomArray
        The atom array to annotate for.
        Non-peptide residues are also allowed and obtain
        :attr:`SecondaryStructure.NONE`.

    Returns
    -------
    sse : ndarray, dtype=int
        An array containing the secondary structure elements as
        :class:`SecondaryStructure` values,
        where the index corresponds to a residue of  `atom_array`
        (see e.g. :func:`get_residues()`).
        :attr:`SecondaryStructure.NONE` indicates that a residue is not
        an amino acid or it comprises no ``CA`` atom.

    Notes
    -----
    Although this function is based on the original `P-SEA` algorithm,
    there are deviations compared to the official `P-SEA` software in
    some cases.
    Do not rely on getting the exact same results.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    SSE of PDB 1L2Y:

    >>> sse = annotate_sse(atom_array)
    >>> print(sse)
    [0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0]
    >>> print("".join(SecondaryStructure.to_symbols(sse)))
    caaaaaaaaccccccccccc
    """
    residue_starts = get_residue_starts(atom_array)
    # Sort CA coord into the coord array at the respective residue index
    # If a residue has no CA, e.g. because it is not an amino acid,
    # the coordinates for that residue remain NaN
    ca_coord = np.full((len(residue_starts), 3), np.nan, dtype=np.float32)
    ca_indices = np.where(
        filter_amino_acids(atom_array) & (atom_array.atom_name == "CA")
    )[0]
    ca_coord[np.searchsorted(residue_starts, ca_indices, "right") - 1] = (
        atom_array.coord[ca_indices]
    )

    if len(ca_coord) <= 5:
        # The number of atoms is too small #
        # to measure the distances/angles
        # -> Return an SSE array where each amino acid is 'coil'
        sse = np.full(len(ca_coord), SecondaryStructure.COIL, dtype=int)
        # Residues where coord are NaN do not belong to amino acids
        # (or at least they have no CA)
        sse[np.isnan(ca_coord).any(axis=-1)] = SecondaryStructure.NONE
        return sse  # pyright: ignore[reportReturnType]

    # Add virtual residues w/o CA coord at chain discontinuity indices
    # This ensures that such discontinuities are recognized for the
    # purpose of geometric measurements
    # -> the distances/angles spanning discontinuities are NaN
    discont_indices = check_res_id_continuity(atom_array)
    discont_res_indices = np.searchsorted(residue_starts, discont_indices, "right") - 1
    ca_coord = np.insert(
        ca_coord,
        discont_res_indices,
        np.full((len(discont_res_indices), 3), np.nan),
        axis=0,
    )
    # Later the SSE for virtual residues are removed again
    # via this mask
    no_virtual_mask = np.ones(len(residue_starts), dtype=bool)
    no_virtual_mask = np.insert(no_virtual_mask, discont_res_indices, False)

    length = len(ca_coord)

    # The distances and angles are not defined for the entire interval,
    # therefore the indices do not have the full range
    # Values that are not defined are NaN
    d2i = np.full(length, np.nan)
    d3i = np.full(length, np.nan)
    d4i = np.full(length, np.nan)
    ri = np.full(length, np.nan)
    ai = np.full(length, np.nan)

    d2i[1 : length - 1] = distance(ca_coord[0 : length - 2], ca_coord[2:length])
    d3i[1 : length - 2] = distance(ca_coord[0 : length - 3], ca_coord[3:length])
    d4i[1 : length - 3] = distance(ca_coord[0 : length - 4], ca_coord[4:length])
    ri[1 : length - 1] = angle(
        ca_coord[0 : length - 2], ca_coord[1 : length - 1], ca_coord[2:length]
    )
    ai[1 : length - 2] = dihedral(
        ca_coord[0 : length - 3],
        ca_coord[1 : length - 2],
        ca_coord[2 : length - 1],
        ca_coord[3 : length - 0],
    )

    # Find CA that meet criteria for potential helices and strands
    relaxed_helix = ((d3i >= _d3_helix[0]) & (d3i <= _d3_helix[1])) | (
        (ri >= _r_helix[0]) & (ri <= _r_helix[1])
    )
    strict_helix = (
        (d3i >= _d3_helix[0])
        & (d3i <= _d3_helix[1])
        & (d4i >= _d4_helix[0])
        & (d4i <= _d4_helix[1])
    ) | (
        (ri >= _r_helix[0])
        & (ri <= _r_helix[1])
        & (ai >= _a_helix[0])
        & (ai <= _a_helix[1])
    )

    relaxed_strand = (d3i >= _d3_strand[0]) & (d3i <= _d3_strand[1])
    strict_strand = (
        (d2i >= _d2_strand[0])
        & (d2i <= _d2_strand[1])
        & (d3i >= _d3_strand[0])
        & (d3i <= _d3_strand[1])
        & (d4i >= _d4_strand[0])
        & (d4i <= _d4_strand[1])
    ) | (
        (ri >= _r_strand[0])
        & (ri <= _r_strand[1])
        & (
            # Account for periodic boundary of dihedral angle
            ((ai >= _a_strand[0]) & (ai <= _a_strand[1]))
            | ((ai >= _a_strand[2]) & (ai <= _a_strand[3]))
        )
    )

    helix_mask = _mask_consecutive(strict_helix, 5)
    helix_mask = _extend_region(helix_mask, relaxed_helix)

    strand_mask = _mask_consecutive(strict_strand, 4)
    short_strand_mask = _mask_regions_with_contacts(
        ca_coord,
        _mask_consecutive(strict_strand, 3),
        min_contacts=5,
        min_distance=4.2,
        max_distance=5.2,
    )
    strand_mask = _extend_region(strand_mask | short_strand_mask, relaxed_strand)

    sse = np.full(length, SecondaryStructure.COIL, dtype=int)
    sse[helix_mask] = SecondaryStructure.HELIX
    sse[strand_mask] = SecondaryStructure.STRAND
    # Residues where coord are NaN do not belong to amino acids
    # (or at least they have no CA)
    sse[np.isnan(ca_coord).any(axis=-1)] = SecondaryStructure.NONE
    # Remove SSE for virtual atoms and return
    return sse[no_virtual_mask]  # pyright: ignore[reportReturnType]


def _mask_consecutive(
    mask: NDArray1[N, np.bool_], number: int
) -> NDArray1[N, np.bool_]:
    """
    Find all regions in a mask with `number` consecutive ``True``
    values.
    Return a mask that is ``True`` for all indices in such a region and
    ``False`` otherwise.
    """
    # An element is in a consecutive region,
    # if it and the following `number-1` elements are True
    # The elements `mask[-(number-1):]` cannot have the sufficient count
    # by this definition, as they are at the end of the array
    counts = np.zeros(len(mask) - (number - 1), dtype=int)
    for i in range(number):
        counts[mask[i : i + len(counts)]] += 1
    consecutive_seed = counts == number

    # Not only that element, but also the
    # following `number-1` elements are in a consecutive region
    consecutive_mask = np.zeros(len(mask), dtype=bool)
    for i in range(number):
        consecutive_mask[i : i + len(consecutive_seed)] |= consecutive_seed

    return consecutive_mask  # pyright: ignore[reportReturnType]


def _extend_region(
    base_condition_mask: NDArray1[N, np.bool_],
    extension_condition_mask: NDArray1[N, np.bool_],
) -> NDArray1[N, np.bool_]:
    """
    Extend a ``True`` region in `base_condition_mask` by at maximum of
    one element at each side, if such element fulfills
    `extension_condition_mask.`
    """
    # This mask always marks the start
    # of either a 'True' or 'False' region
    # Prepend absent region to the start to capture the event,
    # that the first element is already the start of a region
    region_change_mask = np.diff(np.append([False], base_condition_mask))

    # These masks point to the first `False` element
    # left and right of a 'True' region
    # The left end is the element before the first element of a 'True' region
    left_end_mask = region_change_mask & base_condition_mask
    # Therefore the mask needs to be shifted to the left
    left_end_mask = np.append(left_end_mask[1:], [False])
    # The right end is first element of a 'False' region
    right_end_mask = region_change_mask & ~base_condition_mask

    # The 'base_condition_mask' gets additional 'True' elements
    # at left or right ends, which meet the extension criterion
    return base_condition_mask | (
        (left_end_mask | right_end_mask) & extension_condition_mask
    )


def _mask_regions_with_contacts(
    coord: NDArray2[N, XYZ, np.floating],
    candidate_mask: NDArray1[N, np.bool_],
    min_contacts: int,
    min_distance: float,
    max_distance: float,
) -> NDArray1[N, np.bool_]:
    """
    Mask regions of `candidate_mask` that have at least `min_contacts`
    contacts with `coord` in the range `min_distance` to `max_distance`.
    """
    potential_contact_coord = coord[~np.isnan(coord).any(axis=-1)]
    if len(potential_contact_coord) == 0:
        # No potential contacts -> no contacts
        # -> no residue can satisfy 'min_contacts'
        return np.zeros(len(candidate_mask), dtype=bool)  # pyright: ignore[reportReturnType]

    cell_list = CellList(potential_contact_coord, max_distance)
    # For each candidate position,
    # get all contacts within maximum distance
    all_within_max_dist_indices = cell_list.get_atoms(
        coord[candidate_mask], max_distance
    )

    contacts = np.zeros(len(coord), dtype=int)
    for i, atom_index in enumerate(np.where(candidate_mask)[0]):
        within_max_dist_indices = all_within_max_dist_indices[i]
        # Remove padding values
        within_max_dist_indices = within_max_dist_indices[within_max_dist_indices != -1]
        # Now count all contacts within maximum distance
        # that also satisfy the minimum distance
        contacts[atom_index] = np.count_nonzero(
            distance(
                coord[atom_index], potential_contact_coord[within_max_dist_indices]
            )
            > min_distance
        )

    # Count the number of contacts per region
    # These indices mark the start of either a 'True' or 'False' region
    # Prepend absent region to the start to capture the event,
    # that the first element is already the start of a region
    region_change_indices = np.where(np.diff(np.append([False], candidate_mask)))[0]
    # Add exclusive stop
    region_change_indices = np.append(region_change_indices, [len(coord)])
    output_mask = np.zeros(len(candidate_mask), dtype=bool)
    for i in range(len(region_change_indices) - 1):
        start = region_change_indices[i]
        stop = region_change_indices[i + 1]
        total_contacts = np.sum(contacts[start:stop])
        if total_contacts >= min_contacts:
            output_mask[start:stop] = True

    return output_mask  # pyright: ignore[reportReturnType]
