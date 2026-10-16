"""
This module allows estimation of secondary structure elements in protein
structures.
"""

from __future__ import annotations

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["PSeaElement", "DsspElement", "annotate_sse", "annotate_dssp"]

import functools
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, TypeVar
import numpy as np
from numpy.typing import ArrayLike
from biotite.structure.atoms import AtomArray
from biotite.structure.celllist import CellList
from biotite.structure.filter import filter_amino_acids
from biotite.structure.geometry import angle, dihedral, distance
from biotite.structure.integrity import check_res_id_continuity
from biotite.structure.residues import get_residue_starts
from biotite.typing import XYZ, K, N, NDArray1, NDArray2

_R_HELIX = (np.deg2rad(89 - 12), np.deg2rad(89 + 12))
_A_HELIX = (np.deg2rad(50 - 20), np.deg2rad(50 + 20))
_D2_HELIX = ((5.5 - 0.5), (5.5 + 0.5))  # Not used in the algorithm description
_D3_HELIX = ((5.3 - 0.5), (5.3 + 0.5))
_D4_HELIX = ((6.4 - 0.6), (6.4 + 0.6))

_R_STRAND = (np.deg2rad(124 - 14), np.deg2rad(124 + 14))
_A_STRAND = (np.deg2rad(-180), np.deg2rad(-125), np.deg2rad(145), np.deg2rad(180))
_D2_STRAND = ((6.7 - 0.6), (6.7 + 0.6))
_D3_STRAND = ((9.9 - 0.9), (9.9 + 0.9))
_D4_STRAND = ((12.4 - 1.1), (12.4 + 1.1))

# Parameters of the DSSP algorithm, taken from the reference implementation
# H-bonds are only considered between residues with closer CA atoms
_DSSP_CA_CUTOFF = 9.0
# Electrostatic coupling constant (-332 * 0.42 * 0.2) in kcal/mol*Å
_DSSP_COUPLING = -27.888
_DSSP_MIN_DISTANCE = 0.5
_DSSP_MIN_ENERGY = -9.9
_DSSP_MAX_ENERGY = -0.5
# Larger C-N distances between consecutive residues indicate a chain break
_DSSP_PEPTIDE_BOND_CUTOFF = 2.5
_DSSP_BEND_ANGLE = np.deg2rad(70)
_DSSP_PPII_PHI = (np.deg2rad(-75 - 29), np.deg2rad(-75 + 29))
_DSSP_PPII_PSI = (np.deg2rad(145 - 29), np.deg2rad(145 + 29))
_DSSP_PPII_LENGTH = 3


_E = TypeVar("_E", bound=IntEnum)


class PSeaElement(IntEnum):
    r"""
    The secondary structure elements (SSEs) assigned by the *P-SEA*
    algorithm in :func:`annotate_sse()`.

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

    >>> sse = PSeaElement.from_symbols(["c", "a", "a", "b", ""])
    >>> print(sse)
    [ 0  1  1  2 -1]
    >>> print(PSeaElement.to_symbols(sse))
    ['c' 'a' 'a' 'b' '']
    >>> print(PSeaElement.HELIX.symbol)
    a
    >>> print(PSeaElement.from_symbol("b").name)
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
        return _PSEA_SYMBOLS[self]

    @classmethod
    def from_symbol(cls, symbol: str) -> PSeaElement:
        """
        Get the secondary structure element from its one-letter symbol.

        Parameters
        ----------
        symbol : str
            The symbol.

        Returns
        -------
        sse : PSeaElement
            The corresponding secondary structure element.
        """
        for sse, sse_symbol in _PSEA_SYMBOLS.items():
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
        return _symbols_to_values(_PSEA_SYMBOLS, symbols)

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
        return _values_to_symbols(_PSEA_SYMBOLS, sse)


_PSEA_SYMBOLS = {
    PSeaElement.NONE: "",
    PSeaElement.COIL: "c",
    PSeaElement.HELIX: "a",
    PSeaElement.STRAND: "b",
}


class DsspElement(IntEnum):
    r"""
    The secondary structure elements assigned by *DSSP*, i.e.
    :func:`annotate_dssp()` and :class:`biotite.application.dssp.DsspApp`.

    Each element has a corresponding one-letter `symbol`, that is used
    in the *DSSP* output.

    - ``NONE`` (``''``): The residue is not an amino acid or has an
      incomplete backbone.
    - ``COIL`` (``'C'``): loop, coil or irregular
    - ``ALPHA_HELIX`` (``'H'``): :math:`{\alpha}`-helix
    - ``BRIDGE`` (``'B'``): isolated :math:`{\beta}`-bridge
    - ``STRAND`` (``'E'``): extended strand, participation in
      :math:`{\beta}`-ladder
    - ``HELIX_3_10`` (``'G'``): 3 :sub:`10`-helix
    - ``PI_HELIX`` (``'I'``): :math:`{\pi}`-helix
    - ``TURN`` (``'T'``): hydrogen bonded turn
    - ``BEND`` (``'S'``): bend
    - ``POLYPROLINE_HELIX`` (``'P'``): polyproline II helix
      (only assigned by *DSSP* 4 and later)

    Examples
    --------

    >>> sse = DsspElement.from_symbols(["C", "H", "H", "E", "P"])
    >>> print(sse)
    [0 1 1 3 8]
    >>> print(DsspElement.to_symbols(sse))
    ['C' 'H' 'H' 'E' 'P']
    >>> print(DsspElement.ALPHA_HELIX.symbol)
    H
    >>> print(DsspElement.from_symbol("E").name)
    STRAND
    """

    NONE = -1
    COIL = 0
    ALPHA_HELIX = 1
    BRIDGE = 2
    STRAND = 3
    HELIX_3_10 = 4
    PI_HELIX = 5
    TURN = 6
    BEND = 7
    POLYPROLINE_HELIX = 8

    @property
    def symbol(self) -> str:
        """
        The one-letter symbol of the secondary structure element.

        Returns
        -------
        symbol : str
            The one-letter symbol.
        """
        return _DSSP_SYMBOLS[self]

    @classmethod
    def from_symbol(cls, symbol: str) -> DsspElement:
        """
        Get the secondary structure element from its one-letter symbol.

        Parameters
        ----------
        symbol : str
            The symbol.

        Returns
        -------
        sse : DsspElement
            The corresponding secondary structure element.
        """
        for sse, sse_symbol in _DSSP_SYMBOLS.items():
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
        return _symbols_to_values(_DSSP_SYMBOLS, symbols)

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
        return _values_to_symbols(_DSSP_SYMBOLS, sse)


_DSSP_SYMBOLS = {
    DsspElement.NONE: "",
    DsspElement.COIL: "C",
    DsspElement.ALPHA_HELIX: "H",
    DsspElement.BRIDGE: "B",
    DsspElement.STRAND: "E",
    DsspElement.HELIX_3_10: "G",
    DsspElement.PI_HELIX: "I",
    DsspElement.TURN: "T",
    DsspElement.BEND: "S",
    DsspElement.POLYPROLINE_HELIX: "P",
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
        :attr:`PSeaElement.NONE`.

    Returns
    -------
    sse : ndarray, dtype=int
        An array containing the secondary structure elements as
        :class:`PSeaElement` values,
        where the index corresponds to a residue of  `atom_array`
        (see e.g. :func:`get_residues()`).
        :attr:`PSeaElement.NONE` indicates that a residue is not
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
    >>> print("".join(PSeaElement.to_symbols(sse)))
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
        sse = np.full(len(ca_coord), PSeaElement.COIL, dtype=int)
        # Residues where coord are NaN do not belong to amino acids
        # (or at least they have no CA)
        sse[np.isnan(ca_coord).any(axis=-1)] = PSeaElement.NONE
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
    relaxed_helix = ((d3i >= _D3_HELIX[0]) & (d3i <= _D3_HELIX[1])) | (
        (ri >= _R_HELIX[0]) & (ri <= _R_HELIX[1])
    )
    strict_helix = (
        (d3i >= _D3_HELIX[0])
        & (d3i <= _D3_HELIX[1])
        & (d4i >= _D4_HELIX[0])
        & (d4i <= _D4_HELIX[1])
    ) | (
        (ri >= _R_HELIX[0])
        & (ri <= _R_HELIX[1])
        & (ai >= _A_HELIX[0])
        & (ai <= _A_HELIX[1])
    )

    relaxed_strand = (d3i >= _D3_STRAND[0]) & (d3i <= _D3_STRAND[1])
    strict_strand = (
        (d2i >= _D2_STRAND[0])
        & (d2i <= _D2_STRAND[1])
        & (d3i >= _D3_STRAND[0])
        & (d3i <= _D3_STRAND[1])
        & (d4i >= _D4_STRAND[0])
        & (d4i <= _D4_STRAND[1])
    ) | (
        (ri >= _R_STRAND[0])
        & (ri <= _R_STRAND[1])
        & (
            # Account for periodic boundary of dihedral angle
            ((ai >= _A_STRAND[0]) & (ai <= _A_STRAND[1]))
            | ((ai >= _A_STRAND[2]) & (ai <= _A_STRAND[3]))
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

    sse = np.full(length, PSeaElement.COIL, dtype=int)
    sse[helix_mask] = PSeaElement.HELIX
    sse[strand_mask] = PSeaElement.STRAND
    # Residues where coord are NaN do not belong to amino acids
    # (or at least they have no CA)
    sse[np.isnan(ca_coord).any(axis=-1)] = PSeaElement.NONE
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


def annotate_dssp(atom_array: AtomArray[N]) -> NDArray1[K, np.int_]:
    r"""
    Calculate the secondary structure elements (SSEs) of a protein
    based on the `DSSP` algorithm.
    :footcite:`Kabsch1983, Hekkelman2025`

    The assignment is based on the backbone hydrogen bond pattern,
    measured via an electrostatic energy criterion.
    Chain breaks are detected by chain ID changes and by too large
    distances between the ``C`` and ``N`` atom of consecutive residues.

    Parameters
    ----------
    atom_array : AtomArray
        The atom array to annotate for.
        Non-peptide residues are also allowed and obtain
        :attr:`DsspElement.NONE`.

    Returns
    -------
    sse : ndarray, dtype=int
        An array containing the secondary structure elements as
        :class:`DsspElement` values,
        where the index corresponds to a residue of `atom_array`
        (see e.g. :func:`get_residues()`).
        :attr:`DsspElement.NONE` indicates that a residue is not
        an amino acid or it lacks one of the backbone atoms
        ``N``, ``CA``, ``C`` or ``O``.

    See Also
    --------
    annotate_sse : Faster, but less accurate, assignment based on ``CA`` atoms only.
    biotite.application.dssp.DsspApp : Interface to the original *DSSP* software.

    Notes
    -----
    This function implements the algorithm of *DSSP* 4, including
    the assignment of polyproline II helices and the preference of
    :math:`{\pi}`-helices over :math:`{\alpha}`-helices.
    Hydrogen atoms in the input structure are ignored, instead the
    amide hydrogen position is derived from the backbone, as in the
    original *DSSP*.

    In contrast to the original *DSSP* software, the first residue
    after a chain break cannot act as hydrogen bond donor, as the
    position of its amide hydrogen atom is undefined.
    Hence, the assigned SSEs may differ in rare cases.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    SSE of PDB 1L2Y:

    >>> sse = annotate_dssp(atom_array)
    >>> print(sse)
    [0 1 1 1 1 1 1 1 6 6 4 4 4 4 6 0 8 8 8 0]
    >>> print("".join(DsspElement.to_symbols(sse)))
    CHHHHHHHTTGGGGTCPPPC
    """
    residue_starts = get_residue_starts(atom_array)
    backbone_coord = _get_backbone_coord(atom_array, residue_starts)
    # Only residues with complete backbone are considered by DSSP
    is_complete = ~np.isnan(backbone_coord).any(axis=(1, 2))
    sse = np.full(len(residue_starts), DsspElement.NONE, dtype=int)
    if not is_complete.any():
        return sse  # pyright: ignore[reportReturnType]

    n, ca, c, o = backbone_coord[is_complete].transpose(1, 0, 2)
    is_proline = atom_array.res_name[residue_starts][is_complete] == "PRO"
    # Consecutive residues with the same segment ID are part of a continuous chain
    chain_id = atom_array.chain_id[residue_starts][is_complete]
    is_break = (chain_id[1:] != chain_id[:-1]) | (
        distance(c[:-1], n[1:]) > _DSSP_PEPTIDE_BOND_CUTOFF
    )
    segment = np.concatenate([[0], np.cumsum(is_break)])

    acceptors = _find_dssp_hbonds(n, ca, c, o, segment, is_proline)
    sse[is_complete] = _assign_dssp_elements(acceptors, n, ca, c, segment)
    return sse  # pyright: ignore[reportReturnType]


def _get_backbone_coord(
    atom_array: AtomArray, residue_starts: NDArray1[Any, np.integer]
) -> NDArray2[Any, Any, np.floating]:
    """
    Get the coordinates of the ``N``, ``CA``, ``C`` and ``O`` atom
    (in this order) for each residue.
    Coordinates of missing atoms are NaN.
    """
    coord = np.full((len(residue_starts), 4, 3), np.nan, dtype=np.float32)
    is_amino_acid = filter_amino_acids(atom_array)
    for i, atom_name in enumerate(("N", "CA", "C", "O")):
        indices = np.where(is_amino_acid & (atom_array.atom_name == atom_name))[0]
        residue_indices = np.searchsorted(residue_starts, indices, "right") - 1
        coord[residue_indices, i] = atom_array.coord[indices]
    return coord  # pyright: ignore[reportReturnType]


def _find_dssp_hbonds(
    n: NDArray2[N, XYZ, np.floating],
    ca: NDArray2[N, XYZ, np.floating],
    c: NDArray2[N, XYZ, np.floating],
    o: NDArray2[N, XYZ, np.floating],
    segment: NDArray1[N, np.integer],
    is_proline: NDArray1[N, np.bool_],
) -> NDArray2[N, Any, np.integer]:
    """
    Find the backbone hydrogen bonds according to the DSSP energy criterion.

    Returns
    -------
    acceptors : ndarray, shape=(n,2), dtype=int
        For each residue as donor (N-H), the indices of the up to two
        residues, whose C=O accepts the hydrogen bond.
        As in DSSP, only the two energetically most favorable bonds are kept.
        Missing bonds are indicated by ``-1``.
    """
    # The amide hydrogen is placed 1 Å from N, in the direction opposite to
    # the C=O of the preceding residue
    co = c[:-1] - o[:-1]
    h = np.full_like(n, np.nan)
    h[1:] = n[1:] + co / np.linalg.norm(co, axis=-1, keepdims=True)
    # Proline has no amide hydrogen and at chain starts its position is undefined
    can_donate = ~is_proline & ~np.isnan(h).any(axis=-1)
    can_donate[1:] &= segment[1:] == segment[:-1]

    donor, acceptor = _get_ca_pairs(ca).T
    # Like the reference implementation, omit bonds from a residue to its predecessor
    mask = can_donate[donor] & (donor != acceptor + 1)
    donor, acceptor = donor[mask], acceptor[mask]
    energy = _hbond_energy(n[donor], h[donor], c[acceptor], o[acceptor])

    # Keep only the two lowest energy bonds for each donor
    order = np.lexsort((acceptor, energy, donor))
    donor, acceptor, energy = donor[order], acceptor[order], energy[order]
    rank = np.arange(len(donor)) - np.searchsorted(donor, donor)
    mask = (rank < 2) & (energy < _DSSP_MAX_ENERGY)
    acceptors = np.full((len(n), 2), -1, dtype=int)
    acceptors[donor[mask], rank[mask]] = acceptor[mask]
    return acceptors  # pyright: ignore[reportReturnType]


def _get_ca_pairs(ca: NDArray2[N, XYZ, np.floating]) -> NDArray2[Any, Any, np.integer]:
    """
    Get all pairs of distinct residue indices, whose ``CA`` atoms are within
    the DSSP cutoff distance.
    Each pair appears in both orders.
    """
    cell_list = CellList(ca, _DSSP_CA_CUTOFF)
    pairs = cell_list.get_atoms(
        ca, _DSSP_CA_CUTOFF, result_format=CellList.Result.PAIRS
    )
    return pairs[pairs[:, 0] != pairs[:, 1]]


def _hbond_energy(
    n: NDArray2[K, XYZ, np.floating],
    h: NDArray2[K, XYZ, np.floating],
    c: NDArray2[K, XYZ, np.floating],
    o: NDArray2[K, XYZ, np.floating],
) -> NDArray1[K, np.floating]:
    """
    Calculate the electrostatic energy of hydrogen bonds between
    N-H donor and C=O acceptor groups.
    """
    distances = np.stack(
        [distance(h, o), distance(h, c), distance(n, c), distance(n, o)]
    ).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_ho, d_hc, d_nc, d_no = distances
        energy = _DSSP_COUPLING * (1 / d_ho - 1 / d_hc + 1 / d_nc - 1 / d_no)
    energy[(distances < _DSSP_MIN_DISTANCE).any(axis=0)] = _DSSP_MIN_ENERGY
    # Rounding is part of the reference implementation
    return np.maximum(np.round(energy, 3), _DSSP_MIN_ENERGY)


def _assign_dssp_elements(
    acceptors: NDArray2[N, Any, np.integer],
    n: NDArray2[N, XYZ, np.floating],
    ca: NDArray2[N, XYZ, np.floating],
    c: NDArray2[N, XYZ, np.floating],
    segment: NDArray1[N, np.integer],
) -> NDArray1[N, np.int_]:
    """
    Assign the DSSP secondary structure elements based on the hydrogen bonds
    and the backbone geometry.
    The order of assignment establishes the priority of the elements.
    """
    length = len(segment)
    sse = np.full(length, DsspElement.COIL, dtype=int)

    strand_mask, bridge_mask = _find_beta_ladders(acceptors, ca, segment)
    sse[strand_mask] = DsspElement.STRAND
    sse[bridge_mask] = DsspElement.BRIDGE

    # `n_turns[k][i]` is true, if residue `i` starts an n-turn of length `k`,
    # i.e. the C=O of residue `i` is bonded to the N-H of residue `i + k`
    n_turns = {}
    for k in (3, 4, 5):
        i = np.arange(length - k)
        n_turns[k] = np.zeros(length, dtype=bool)
        n_turns[k][i] = (segment[i] == segment[i + k]) & _has_hbond(acceptors, i + k, i)
    # A minimal helix is defined by two consecutive n-turns
    helix_starts = {k: n_turns[k] & _shift(n_turns[k], 1) for k in n_turns}

    sse[_shift(helix_starts[4], range(4)).any(axis=0)] = DsspElement.ALPHA_HELIX
    # 3-10 helices must not overwrite other elements
    is_occupied = np.isin(
        sse, [DsspElement.ALPHA_HELIX, DsspElement.STRAND, DsspElement.BRIDGE]
    )
    starts = helix_starts[3] & ~_shift(is_occupied, range(0, -3, -1)).any(axis=0)
    sse[_shift(starts, range(3)).any(axis=0)] = DsspElement.HELIX_3_10
    # Pi helices take precedence over alpha helices
    is_occupied = np.isin(
        sse, [DsspElement.HELIX_3_10, DsspElement.STRAND, DsspElement.BRIDGE]
    )
    starts = helix_starts[5] & ~_shift(is_occupied, range(0, -5, -1)).any(axis=0)
    sse[_shift(starts, range(5)).any(axis=0)] = DsspElement.PI_HELIX

    # Residues enclosed by any n-turn are part of a turn
    turn_mask = np.zeros(length, dtype=bool)
    for k, n_turn in n_turns.items():
        turn_mask |= _shift(n_turn, range(1, k)).any(axis=0)
    sse[turn_mask & (sse == DsspElement.COIL)] = DsspElement.TURN

    # The bend angle is the angle between the CA(i-2)->CA(i) and CA(i)->CA(i+2)
    bend_angle = np.full(length, np.nan)
    i = np.arange(2, length - 2)
    bend_angle[i] = np.pi - angle(ca[i - 2], ca[i], ca[i + 2])
    bend_angle[i[segment[i - 2] != segment[i + 2]]] = np.nan
    sse[(bend_angle > _DSSP_BEND_ANGLE) & (sse == DsspElement.COIL)] = DsspElement.BEND

    # Polyproline II helices are defined by consecutive residues with
    # characteristic backbone dihedral angles
    phi = np.full(length, np.nan)
    psi = np.full(length, np.nan)
    phi[1:] = dihedral(c[:-1], n[1:], ca[1:], c[1:])
    psi[:-1] = dihedral(n[:-1], ca[:-1], c[:-1], n[1:])
    is_continuous = segment[1:] == segment[:-1]
    phi[1:][~is_continuous] = np.nan
    psi[:-1][~is_continuous] = np.nan
    is_ppii = (
        (phi >= _DSSP_PPII_PHI[0])
        & (phi <= _DSSP_PPII_PHI[1])
        & (psi >= _DSSP_PPII_PSI[0])
        & (psi <= _DSSP_PPII_PSI[1])
    )
    window = range(_DSSP_PPII_LENGTH)
    starts = _shift(is_ppii, [-k for k in window]).all(axis=0)
    ppii_mask = _shift(starts, window).any(axis=0)  # pyright: ignore[reportArgumentType]
    sse[ppii_mask & (sse == DsspElement.COIL)] = DsspElement.POLYPROLINE_HELIX

    return sse  # pyright: ignore[reportReturnType]


@dataclass
class _Ladder:
    """
    A ladder of consecutive bridges between the residues `i` and `j`.
    `i` and `j` are kept in ascending order, so for antiparallel ladders
    ``i[k]`` is paired with ``j[-k-1]``.
    """

    is_parallel: bool
    i: list[int]
    j: list[int]


def _find_beta_ladders(
    acceptors: NDArray2[N, Any, np.integer],
    ca: NDArray2[N, XYZ, np.floating],
    segment: NDArray1[N, np.integer],
) -> tuple[NDArray1[N, np.bool_], NDArray1[N, np.bool_]]:
    """
    Find the residues that are part of a ladder of multiple bridges (strand)
    or of a ladder of only a single bridge (isolated bridge).
    """
    length = len(segment)
    pairs = _get_ca_pairs(ca)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    # Process pairs in the same order as the reference implementation
    i, j = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))].T
    # A bridge requires the adjacent residues to exist without a chain break
    mask = (i > 0) & (j < length - 1)
    i, j = i[mask], j[mask]
    mask = (segment[i - 1] == segment[i + 1]) & (segment[j - 1] == segment[j + 1])
    i, j = i[mask], j[mask]

    hbond = functools.partial(_has_hbond, acceptors)
    is_parallel = (hbond(i + 1, j) & hbond(j, i - 1)) | (
        hbond(j + 1, i) & hbond(i, j - 1)
    )
    is_antiparallel = ~is_parallel & (
        (hbond(i + 1, j - 1) & hbond(j + 1, i - 1)) | (hbond(j, i) & hbond(i, j))
    )
    is_bridge = is_parallel | is_antiparallel

    # Join consecutive bridges into ladders
    ladders = []
    # Maps the next bridge, that would continue a ladder, to that ladder
    ladder_ends = {}
    for bridge_i, bridge_j, parallel in zip(
        i[is_bridge].tolist(), j[is_bridge].tolist(), is_parallel[is_bridge].tolist()
    ):
        ladder = ladder_ends.pop((bridge_i, bridge_j, parallel), None)
        if ladder is None:
            ladder = _Ladder(parallel, [bridge_i], [bridge_j])
            ladders.append(ladder)
        else:
            ladder.i.append(bridge_i)
            if parallel:
                ladder.j.append(bridge_j)
            else:
                ladder.j.insert(0, bridge_j)
        next_j = bridge_j + 1 if parallel else bridge_j - 1
        ladder_ends[bridge_i + 1, next_j, parallel] = ladder

    # Join ladders separated by a beta-bulge
    ladders.sort(key=lambda ladder: ladder.i[0])
    k = 0
    while k < len(ladders):
        ladder = ladders[k]
        m = k + 1
        while m < len(ladders):
            if ladders[m].i[0] - ladder.i[-1] >= 6:
                # As ladders are sorted, all following ladders are also too far
                # away to be connected via a bulge (see `_is_bulge()`)
                break
            if _is_bulge(ladder, ladders[m], segment):
                other = ladders.pop(m)
                ladder.i = ladder.i + other.i
                if ladder.is_parallel:
                    ladder.j = ladder.j + other.j
                else:
                    ladder.j = other.j + ladder.j
            else:
                m += 1
        k += 1

    strand_mask = np.zeros(length, dtype=bool)
    bridge_mask = np.zeros(length, dtype=bool)
    for ladder in ladders:
        mask = strand_mask if len(ladder.i) > 1 else bridge_mask
        mask[ladder.i[0] : ladder.i[-1] + 1] = True
        mask[ladder.j[0] : ladder.j[-1] + 1] = True
    # Strands take precedence over isolated bridges
    return strand_mask, bridge_mask & ~strand_mask  # pyright: ignore[reportReturnType]


def _is_bulge(
    ladder: _Ladder, other: _Ladder, segment: NDArray1[Any, np.integer]
) -> bool:
    """
    Check whether the `other` ladder continues `ladder` after a beta-bulge,
    i.e. a gap of at most 4 residues on one strand and at most 1 residue
    on the other strand.
    """
    if ladder.is_parallel != other.is_parallel:
        return False
    # Both ladders must reside in the same continuous chains
    if (
        segment[min(ladder.i[0], other.i[0])] != segment[max(ladder.i[-1], other.i[-1])]
        or segment[min(ladder.j[0], other.j[0])]
        != segment[max(ladder.j[-1], other.j[-1])]
    ):
        return False
    gap_i = other.i[0] - ladder.i[-1]
    if ladder.is_parallel:
        gap_j = other.j[0] - ladder.j[-1]
    else:
        gap_j = ladder.j[0] - other.j[-1]
    if not 0 < gap_i < 6 or gap_j < 0:
        return False
    return (gap_j < 6 and gap_i < 3) or gap_j < 3


def _has_hbond(
    acceptors: NDArray2[Any, Any, np.integer],
    donor: NDArray1[K, np.integer],
    acceptor: NDArray1[K, np.integer],
) -> NDArray1[K, np.bool_]:
    """
    Check for each given donor/acceptor residue pair, whether a hydrogen bond
    exists.
    """
    return (acceptors[donor] == acceptor[:, np.newaxis]).any(axis=-1)


def _shift(
    mask: NDArray1[N, np.bool_], offsets: int | range | list[int]
) -> NDArray2[Any, N, np.bool_]:
    """
    Shift the mask by the given offsets, i.e. ``shifted[k, i] = mask[i - offsets[k]]``.
    Positions shifted in from outside the mask are ``False``.
    If a single offset is given, a 1D array is returned.
    """
    if isinstance(offsets, int):
        return _shift(mask, [offsets])[0]
    length = len(mask)
    shifted = np.zeros((len(offsets), length), dtype=bool)
    for k, offset in enumerate(offsets):
        if offset >= 0:
            shifted[k, offset:] = mask[: max(length - offset, 0)]
        else:
            shifted[k, :offset] = mask[-offset:]
    return shifted  # pyright: ignore[reportReturnType]
