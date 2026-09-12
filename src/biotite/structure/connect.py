# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "connect_via_distances",
    "connect_via_residue_names",
    "find_connected",
    "find_rotatable_bonds",
    "infer_bond_types",
]

import itertools
import warnings
from collections.abc import Mapping
from typing import Any
import networkx as nx
import numpy as np
from biotite.rust.structure import (
    connect_inter_residue as rust_connect_inter_residue,
)
from biotite.rust.structure import (
    connect_via_distances as rust_connect_via_distances,
)
from biotite.rust.structure import (
    connect_via_residue_names as rust_connect_via_residue_names,
)
from biotite.rust.structure import (
    find_connected,
)
from biotite.rust.structure import (
    infer_bond_types as rust_infer_bond_types,
)
from biotite.structure.atoms import AtomArray, AtomArrayStack
from biotite.structure.bonds import BondList, BondType
from biotite.structure.error import BadStructureError, InconsistentBondTypeWarning
from biotite.typing import K, N, NDArray1

# Covalent radii (in Å) taken from Cordero et al.
# For carbon the sp3 radius is used, for manganese, iron and cobalt the low-spin radius
_COVALENT_RADII = {
    "H": 0.31,
    "HE": 0.28,
    "LI": 1.28,
    "BE": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "NE": 0.58,
    "NA": 1.66,
    "MG": 1.41,
    "AL": 1.21,
    "SI": 1.11,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "AR": 1.06,
    "K": 2.03,
    "CA": 1.76,
    "SC": 1.70,
    "TI": 1.60,
    "V": 1.53,
    "CR": 1.39,
    "MN": 1.39,
    "FE": 1.32,
    "CO": 1.26,
    "NI": 1.24,
    "CU": 1.32,
    "ZN": 1.22,
    "GA": 1.22,
    "GE": 1.20,
    "AS": 1.19,
    "SE": 1.20,
    "BR": 1.20,
    "KR": 1.16,
    "RB": 2.20,
    "SR": 1.95,
    "Y": 1.90,
    "ZR": 1.75,
    "NB": 1.64,
    "MO": 1.54,
    "TC": 1.47,
    "RU": 1.46,
    "RH": 1.42,
    "PD": 1.39,
    "AG": 1.45,
    "CD": 1.44,
    "IN": 1.42,
    "SN": 1.39,
    "SB": 1.39,
    "TE": 1.38,
    "I": 1.39,
    "XE": 1.40,
    "CS": 2.44,
    "BA": 2.15,
    "LA": 2.07,
    "CE": 2.04,
    "PR": 2.03,
    "ND": 2.01,
    "PM": 1.99,
    "SM": 1.98,
    "EU": 1.98,
    "GD": 1.96,
    "TB": 1.94,
    "DY": 1.92,
    "HO": 1.92,
    "ER": 1.89,
    "TM": 1.90,
    "YB": 1.87,
    "LU": 1.87,
    "HF": 1.75,
    "TA": 1.70,
    "W": 1.62,
    "RE": 1.51,
    "OS": 1.44,
    "IR": 1.41,
    "PT": 1.36,
    "AU": 1.36,
    "HG": 1.32,
    "TL": 1.45,
    "PB": 1.46,
    "BI": 1.48,
    "PO": 1.40,
    "AT": 1.50,
    "RN": 1.50,
    "FR": 2.60,
    "RA": 2.21,
    "AC": 2.15,
    "TH": 2.06,
    "PA": 2.00,
    "U": 1.96,
    "NP": 1.90,
    "PU": 1.87,
    "AM": 1.80,
    "CM": 1.69,
}


def connect_via_distances(
    atoms: AtomArray[N],
    distance_range: Mapping[tuple[str, str], tuple[float, float]] | None = None,
    tolerance: float = 0.4,
    inter_residue: bool = True,
    default_bond_type: BondType = BondType.ANY,
    periodic: bool = False,
) -> BondList[N]:
    """
    Create a :class:`BondList` for a given atom array, based on pairwise atom distances.

    A :attr:`BondType.ANY`, bond is created for two atoms within the same residue,
    if the distance between them does not exceed the sum of their covalent radii
    :footcite:`Cordero2008` plus the given `tolerance`.
    Bonds between two adjacent residues are created for the atoms expected to connect
    these residues, i.e. ``'C'`` and ``'N'`` for peptides and ``"O3'"`` and ``'P'`` for
    nucleotides.

    Parameters
    ----------
    atoms : AtomArray
        The structure to create the :class:`BondList` for.
    distance_range : dict of tuple(str, str) -> tuple(float, float), optional
        Deprecated, has no effect anymore:
        The distance range is now computed from the covalent radii and `tolerance`.
    tolerance : float, optional
        The tolerance added to the sum of the covalent radii of two atoms to obtain the
        maximum bond distance.
    inter_residue : bool, optional
        If true, connections between consecutive amino acids and
        nucleotides are also added.
    default_bond_type : BondType or int, optional
        By default, all created bonds have :attr:`BondType.ANY`.
        An alternative :class:`BondType` can be given in this parameter.
    periodic : bool, optional
        If set to true, bonds can also be detected in periodic
        boundary conditions.
        The `box` attribute of `atoms` is required in this case.

    Returns
    -------
    BondList
        The created bond list.
        It only contains connectivity information, i.e. each bond has the
        :attr:`BondType.ANY` type.

    See Also
    --------
    connect_via_residue_names
        Connect atoms based on their residue names, which is more accurate when the
        molecule is part of the *Chemical Component Dictionary* (CCD).
    infer_bond_types
        Infer proper bond types for the created bonds afterwards.

    Notes
    -----
    A bond is created between two atoms, if their distance :math:`D` fulfills

    .. math::

        D \\leq r_1 + r_2 + t,

    where :math:`r_1` and :math:`r_2` are the covalent radii of the two atoms and
    :math:`t` is the given `tolerance`.
    Covalent radii are taken from :footcite:`Cordero2008`, where the
    radius for :math:`sp^3` carbon is used for carbon atoms and the
    low-spin radius is used for manganese, iron and cobalt.

    This method might miss bonds, if the bond distance is unexpectedly high, or it might
    create false bonds, if two atoms within a residue are accidentally in the right
    distance.
    The latter is typically the case for metal atoms in clusters, e.g. iron-sulfur
    clusters, as their distance to each other is within the covalent range.
    A more accurate method for determining bonds is :func:`connect_via_residue_names()`,
    if residue names from the *Chemical Component Dictionary* are used.

    References
    ----------

    .. footbibliography::
    """
    from biotite.structure.atoms import AtomArray
    from biotite.structure.residues import get_residue_starts

    if distance_range is not None:
        warnings.warn(
            "'distance_range' is deprecated and has no effect anymore, "
            "use 'tolerance' to adjust the maximum bond distance",
            DeprecationWarning,
        )
    if not isinstance(atoms, AtomArray):
        raise TypeError(f"Expected 'AtomArray', not '{type(atoms).__name__}'")
    if periodic:
        if atoms.box is None:
            raise BadStructureError("Atom array has no box")

    # NaN for unknown elements ensures that they never form bonds,
    # as any comparison with NaN is false
    radii = np.array(
        [_COVALENT_RADII.get(element.upper(), np.nan) for element in atoms.element],
        dtype=np.float32,
    )

    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)

    if periodic:
        bond_list = _connect_via_distances_periodic(
            atoms, residue_starts, radii, tolerance, default_bond_type
        )
    else:
        bond_list = rust_connect_via_distances(
            atoms.coord,
            radii,
            residue_starts,
            tolerance,
            default_bond_type,
        )

    if inter_residue:
        inter_bonds = _connect_inter_residue(atoms, residue_starts)
        if default_bond_type == BondType.ANY:
            # As all bonds should be of type ANY, convert also
            # inter-residue bonds to ANY
            inter_bonds.remove_bond_order()
        return bond_list.merge(inter_bonds)
    else:
        return bond_list


def _connect_via_distances_periodic(
    atoms: AtomArray[N],
    residue_starts: NDArray1[K, np.integer],
    radii: NDArray1[N, np.floating],
    tolerance: float,
    default_bond_type: BondType,
) -> BondList[N]:
    """
    Fallback for :func:`connect_via_distances` with periodic boundary conditions.
    Uses :func:`biotite.structure.geometry.distance` for periodic distance computation.
    """
    from biotite.structure.geometry import distance

    box = atoms.box
    coord = atoms.coord
    bt_int = int(default_bond_type)

    bonds = []
    for start, stop in itertools.pairwise(residue_starts):
        radii_in_res = radii[start:stop]
        coord_in_res = coord[start:stop]
        distances = distance(
            coord_in_res[:, np.newaxis, :], coord_in_res[np.newaxis, :, :], box
        )
        for atom_index1 in range(len(radii_in_res)):
            for atom_index2 in range(atom_index1):
                max_distance = (
                    radii_in_res[atom_index1] + radii_in_res[atom_index2] + tolerance
                )
                dist = distances[atom_index1, atom_index2]  # pyright: ignore[reportIndexIssue]
                # The comparison is false for atoms with unknown radius (NaN)
                if dist <= max_distance:
                    bonds.append((start + atom_index1, start + atom_index2, bt_int))

    if bonds:
        return BondList(atoms.array_length(), np.stack(bonds, axis=0, dtype=np.int64))
    return BondList(atoms.array_length())


def connect_via_residue_names(
    atoms: AtomArray[N] | AtomArrayStack[Any, N],
    inter_residue: bool = True,
    custom_bond_dict: Mapping[str, Mapping[tuple[str, str], BondType]] | None = None,
) -> BondList[N]:
    """
    Create a :class:`BondList` for a given atom array (stack), based on
    the deposited bonds for each residue in the *Chemical Component Dictionary* (CCD)
    dataset.

    Bonds between two adjacent residues are created for the atoms
    expected to connect these residues, i.e. ``'C'`` and ``'N'`` for
    peptides and ``"O3'"`` and ``'P'`` for nucleotides.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The structure to create the :class:`BondList` for.
    inter_residue : bool, optional
        If true, connections between consecutive amino acids and
        nucleotides are also added.
    custom_bond_dict : dict (str -> dict ((str, str) -> int)), optional
        A dictionary of dictionaries:
        The outer dictionary maps residue names to inner dictionaries.
        The inner dictionary maps tuples of two atom names to their
        respective :class:`BondType` (represented as integer).
        If given, these bonds are used instead of the bonds read from the CCD.

    Returns
    -------
    BondList
        The created bond list.
        No bonds are added for residues that are not found in the CCD.

    See Also
    --------
    connect_via_distances
        Connect atoms based on their pairwise distances in case the
        molecule is not part of the *Chemical Component Dictionary* (CCD).

    Notes
    -----
    This method can only find bonds for residues in the RCSB
    *Chemical Component Dictionary*, unless `custom_bond_dict` is set.
    Although this includes most molecules one encounters, this will fail
    for exotic molecules, e.g. specialized inhibitors.

    .. currentmodule:: biotite.structure.info

    To supplement `custom_bond_dict` with bonds for residues from the
    *Chemical Component Dictionary*  you can use
    :meth:`bonds_in_residue()`.

    >>> import pprint
    >>> custom_bond_dict = {
    ...     "XYZ": {
    ...         ("A", "B"): BondType.SINGLE,
    ...         ("B", "C"): BondType.SINGLE
    ...     }
    ... }
    >>> # Supplement with bonds for common residues
    >>> custom_bond_dict["ALA"] = bonds_in_residue("ALA")
    >>> pp = pprint.PrettyPrinter(width=40)
    >>> pp.pprint(custom_bond_dict)
    {'ALA': {('C', 'O'): <BondType.DOUBLE: 2>,
             ('C', 'OXT'): <BondType.SINGLE: 1>,
             ('CA', 'C'): <BondType.SINGLE: 1>,
             ('CA', 'CB'): <BondType.SINGLE: 1>,
             ('CA', 'HA'): <BondType.SINGLE: 1>,
             ('CB', 'HB1'): <BondType.SINGLE: 1>,
             ('CB', 'HB2'): <BondType.SINGLE: 1>,
             ('CB', 'HB3'): <BondType.SINGLE: 1>,
             ('N', 'CA'): <BondType.SINGLE: 1>,
             ('N', 'H'): <BondType.SINGLE: 1>,
             ('N', 'H2'): <BondType.SINGLE: 1>,
             ('OXT', 'HXT'): <BondType.SINGLE: 1>},
     'XYZ': {('A', 'B'): <BondType.SINGLE: 1>,
             ('B', 'C'): <BondType.SINGLE: 1>}}
    """
    # Avoid circular import
    from biotite.structure.info.bonds import bonds_in_residue
    from biotite.structure.residues import get_residue_starts

    bond_info = {
        res_name: _bond_dict_to_list(bonds_in_residue(res_name)) or []
        for res_name in np.unique(atoms.res_name)
    }
    if custom_bond_dict is not None:
        bond_info = {
            res_name: _bond_dict_to_list(bond_info)
            for res_name, bond_info in custom_bond_dict.items()
        }

    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    bond_list = rust_connect_via_residue_names(
        atoms.res_name.tolist(), atoms.atom_name.tolist(), residue_starts, bond_info
    )

    if inter_residue:
        inter_bonds = _connect_inter_residue(atoms, residue_starts)
        return bond_list.merge(inter_bonds)
    else:
        return bond_list


def infer_bond_types(
    atoms: AtomArray[N],
    total_charge: int = 0,
    max_iterations: int | None = None,
) -> tuple[BondList[N], NDArray1[N, np.integer]]:
    """
    Infer the bond types for the bonds in the given structure based on the connectivity
    between the atoms, their elements and the total charge.
    :footcite:`Kim2015`

    Combinations of allowed valences for each atom are enumerated in decreasing order
    of total unsaturation and multiple bonds are greedily assigned between adjacent
    unsaturated atoms.
    The first assignment, in which each atom satisfies its valence and the formal
    charges sum up to `total_charge`, is returned.

    Parameters
    ----------
    atoms : AtomArray
        The structure to infer the bond types for.
        The ``bonds`` attribute is required as source of the connectivity.
        Note that hydrogen atoms must be present, as otherwise the unsaturation of
        their bonded heavy atoms cannot be determined correctly.
        If the ``charge`` annotation is present, the formal charge of each atom is
        constrained to the given value, which narrows the search considerably.
    total_charge : int, optional
        The total formal charge of the structure, used as constraint during the
        assignment.
    max_iterations : int, optional
        The maximum number of bond order assignments to try.
        By default, all valence combinations are tried before the search is given up.

    Returns
    -------
    bonds : BondList
        The bonds from ``atoms.bonds`` with assigned bond types.
        Only :attr:`BondType.SINGLE`, :attr:`BondType.DOUBLE` and
        :attr:`BondType.TRIPLE` are assigned.
        Hence, bonds in aromatic systems are represented by alternating single and
        double bonds, i.e. one kekulized resonance structure.
    charges : ndarray, shape=(n,), dtype=int
        The formal charge of each atom, as implied by the assigned bond types.
        If the ``charge`` annotation is present in `atoms`, it is returned unchanged.
        Otherwise atoms with an unsupported element are assigned a charge of ``0``.

    Warns
    -----
    InconsistentBondTypeWarning
        If no assignment exists, where each atom satisfies its valence and the formal
        charges sum up to `total_charge`, or `max_iterations` is exceeded,
        before such an assignment is found.
        In both cases the assignment with the highest sum of bond orders found so far
        is returned.

    Warnings
    --------
    This function is recommended for small molecules only:
    The number of valence combinations grows exponentially with the number of atoms
    with multiple allowed valences (e.g. nitrogen, oxygen, sulfur and phosphorus).
    Hence, for large molecules long run times can be expected.
    Use `max_iterations` to abort the search prematurely in such cases.

    Furthermore, the following molecules are not covered by the underlying valence
    model and hence cannot be assigned properly:

    - Radicals and other odd-electron species, e.g. superoxide.
    - Hypervalent sulfur or selenium with only two bond partners, e.g. in sulfur
      dioxide, as their allowed valences are restricted based on their number of bond
      partners.
    - Fused aromatic systems, where the multiple bonds may be placed in a way that
      satisfies all valences, but does not reproduce the aromatic system.

    In the first two cases an :class:`InconsistentBondTypeWarning` is raised.
    The search may also fail for other exotic molecules, especially if the ``charge``
    annotation is missing.

    See Also
    --------
    connect_via_distances
        Can be used to create the required connectivity beforehand, if it is unknown.

    Notes
    -----
    Supported elements are *H*, *B*, *C*, *N*, *O*, *F*, *Si*, *P*, *S*, *Cl*, *Se*,
    *Br* and *I*.
    Bonds of atoms with other elements (e.g. metals) keep :attr:`BondType.SINGLE`.
    Their formal charge is taken from the ``charge`` annotation, if it is present, and
    is assumed to be zero otherwise.

    In deviation from the reference, which allows only the valences of the neutral
    element, each atom may also adopt the valences of the anion and cation, i.e. it may
    carry a formal charge of ``-1`` or ``1``.
    As valence states with fewer formal charges are preferred, the neutral form is
    still chosen, if it satisfies the constraints.
    Consequently, if the ``charge`` annotation is present, the possible valence states
    of each atom are narrowed down to those with the respective formal charge, which
    reduces the number of combinations to try substantially.
    If the given charge does not fit any valence state of the respective element, the
    charge is still trusted, as it may represent a bonding situation that the valence
    model cannot express, e.g. a dative bond to a four-bonded neutral boron atom.
    In this rare case the atom is treated as saturated, i.e. all its bonds remain
    single bonds, and its valence does not imply its formal charge anymore.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> molecule = residue("URA")
    >>> # Remove bond type information, but keep connectivity
    >>> molecule.bonds.remove_bond_order()
    >>> # Infer bond types
    >>> molecule.bonds, molecule.charge = infer_bond_types(molecule)
    >>> for i, j, bond_type in molecule.bonds.as_array():
    ...     print(
    ...         molecule.atom_name[i], molecule.atom_name[j], BondType(bond_type).name
    ...     )
    N1 C2 SINGLE
    N1 C6 SINGLE
    N1 HN1 SINGLE
    C2 O2 DOUBLE
    C2 N3 SINGLE
    N3 C4 SINGLE
    N3 HN3 SINGLE
    C4 O4 DOUBLE
    C4 C5 SINGLE
    C5 C6 DOUBLE
    C5 H5 SINGLE
    C6 H6 SINGLE
    """
    if atoms.bonds is None:
        raise BadStructureError("The input structure must have an associated BondList")
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("At least one iteration is required")

    # Known formal charges constrain the valence state of each atom
    if "charge" in atoms.get_annotation_categories():
        known_charges = atoms.charge.astype(np.int64, copy=False)
    else:
        known_charges = None
    bond_list, charges, converged = rust_infer_bond_types(
        [element.upper() for element in atoms.element],
        atoms.bonds,
        total_charge,
        known_charges,
        max_iterations,
    )
    if not converged:
        warnings.warn(
            "No bond type assignment satisfying valences and total charge was found, "
            "returning the assignment with the highest sum of bond orders found so far",
            InconsistentBondTypeWarning,
        )
    return bond_list, charges


def find_rotatable_bonds(bonds: BondList[N]) -> BondList[N]:
    """
    Find all rotatable bonds in a given :class:`BondList`.

    The following conditions must be true for a bond to be counted as
    rotatable:

        1. The bond must be a single bond (``BondType.SINGLE``)
        2. The connected atoms must not be within the same cycle/ring
        3. Both connected atoms must not be terminal, e.g. not a *C-H*
           bond, as rotation about such bonds would not change any
           coordinates

    Parameters
    ----------
    bonds : BondList
        The bonds to find the rotatable bonds in.

    Returns
    -------
    rotatable_bonds : BondList
        The subset of the input `bonds` that contains only rotatable
        bonds.

    Examples
    --------

    >>> molecule = residue("TYR")
    >>> for i, j, _ in find_rotatable_bonds(molecule.bonds).as_array():
    ...     print(molecule.atom_name[i], molecule.atom_name[j])
    N CA
    CA C
    CA CB
    C OXT
    CB CG
    CZ OH
    """
    bond_graph = bonds.as_graph()
    cycles = nx.algorithms.cycles.cycle_basis(bond_graph)  # pyright: ignore[reportAttributeAccessIssue]

    number_of_partners = np.count_nonzero(bonds.get_all_bonds()[0] != -1, axis=1)

    rotatable_bonds = []
    bonds_array = bonds.as_array()
    for i, j, bond_type in bonds_array:
        # Can only rotate about single bonds
        # Furthermore, it makes no sense to rotate about a bond,
        # that leads to a single atom
        if (
            bond_type == BondType.SINGLE
            and number_of_partners[i] > 1
            and number_of_partners[j] > 1
        ):
            # Cannot rotate about a bond, if the two connected atoms
            # are in a cycle
            in_same_cycle = False
            for cycle in cycles:
                if i in cycle and j in cycle:
                    in_same_cycle = True
                    break
            if not in_same_cycle:
                rotatable_bonds.append((i, j, int(bond_type)))
    if rotatable_bonds:
        return BondList(
            bonds.get_atom_count(), np.array(rotatable_bonds, dtype=np.int64)
        )
    else:
        return BondList(bonds.get_atom_count())


def _connect_inter_residue(
    atoms: AtomArray[N] | AtomArrayStack[Any, N],
    residue_starts: NDArray1[Any, np.integer],
) -> BondList[N]:
    """
    Create a :class:`BondList` containing the bonds between adjacent
    amino acid or nucleotide residues.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The structure to create the :class:`BondList` for.
    residue_starts : ndarray, dtype=int
        Return value of
        ``get_residue_starts(atoms, add_exclusive_stop=True)``.

    Returns
    -------
    BondList
        A bond list containing all inter residue bonds.
    """
    # Avoid circular import
    from biotite.structure.info.misc import link_type

    link_types = [
        link_type(res_name) for res_name in atoms.res_name[residue_starts[:-1]]
    ]
    is_disconnected = (
        # Residues are not inside the same chain
        (atoms.chain_id[residue_starts[1:-1]] != atoms.chain_id[residue_starts[:-2]])
        # There is at least one residue missing in between
        | (atoms.res_id[residue_starts[1:-1]] - atoms.res_id[residue_starts[:-2]] > 1)
    )
    return rust_connect_inter_residue(
        atoms.atom_name.tolist(), residue_starts, link_types, is_disconnected
    )


def _bond_dict_to_list(
    bond_dict: Mapping[tuple[str, str], BondType] | None,
) -> list[tuple[str, str, BondType]] | None:
    """
    Convert the input bond dictionary in the form ``(name1, name2) -> bond_type`` into a
    list of tuples ``(index1, index2, bond_type)``.
    """
    if bond_dict is None:
        return None
    else:
        return [
            (atom_name1, atom_name2, bond_type)
            for (atom_name1, atom_name2), bond_type in bond_dict.items()
        ]
