# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module allows efficient search of atoms in a defined radius around
a location.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "BondList",
    "BondType",
    "connect_via_distances",
    "connect_via_residue_names",
    "find_connected",
    "find_rotatable_bonds",
]

import itertools
from enum import IntEnum
import networkx as nx
import numpy as np
from biotite.rust.structure import BondList, bond_type_members
from biotite.structure.error import BadStructureError


def _without_aromaticity(self):
    """
    Get the non-aromatic counterpart of this bond type.

    If this bond type is already non-aromatic, it is returned unchanged.

    Returns
    -------
    BondType
        The non-aromatic counterpart of this bond type.

    Examples
    --------
    >>> BondType.AROMATIC_DOUBLE.without_aromaticity()
    <BondType.DOUBLE: 2>
    >>> BondType.SINGLE.without_aromaticity()
    <BondType.SINGLE: 1>
    """
    match self:
        case BondType.AROMATIC_SINGLE:
            return BondType.SINGLE
        case BondType.AROMATIC_DOUBLE:
            return BondType.DOUBLE
        case BondType.AROMATIC_TRIPLE:
            return BondType.TRIPLE
        case BondType.AROMATIC:
            return BondType.ANY
        case _:
            return self


# Create BondType IntEnum dynamically from Rust enum members
BondType = IntEnum(
    "BondType",
    {name: value for name, value in bond_type_members().items()},
    module=__name__,
)
BondType.__doc__ = """
This enum type represents the type of a chemical bond.

- ``ANY`` - Used if the actual type is unknown
- ``SINGLE`` - Single bond
- ``DOUBLE`` - Double bond
- ``TRIPLE`` - Triple bond
- ``QUADRUPLE`` - A quadruple bond
- ``AROMATIC_SINGLE`` - Aromatic bond with a single formal bond
- ``AROMATIC_DOUBLE`` - Aromatic bond with a double formal bond
- ``AROMATIC_TRIPLE`` - Aromatic bond with a triple formal bond
- ``COORDINATION`` - Coordination complex involving a metal atom
- ``AROMATIC`` - Aromatic bond without specification of the formal bond
"""
BondType.without_aromaticity = _without_aromaticity


# fmt: off
_DEFAULT_DISTANCE_RANGE = {
    # Taken from Allen et al.
    #               min   - 2*std     max   + 2*std
    ("B",  "C" ) : (1.556 - 2*0.015,  1.556 + 2*0.015),
    ("BR", "C" ) : (1.875 - 2*0.029,  1.966 + 2*0.029),
    ("BR", "O" ) : (1.581 - 2*0.007,  1.581 + 2*0.007),
    ("C",  "C" ) : (1.174 - 2*0.011,  1.588 + 2*0.025),
    ("C",  "CL") : (1.713 - 2*0.011,  1.849 + 2*0.011),
    ("C",  "F" ) : (1.320 - 2*0.009,  1.428 + 2*0.009),
    ("C",  "H" ) : (1.059 - 2*0.030,  1.099 + 2*0.007),
    ("C",  "I" ) : (2.095 - 2*0.015,  2.162 + 2*0.015),
    ("C",  "N" ) : (1.325 - 2*0.009,  1.552 + 2*0.023),
    ("C",  "O" ) : (1.187 - 2*0.011,  1.477 + 2*0.008),
    ("C",  "P" ) : (1.791 - 2*0.006,  1.855 + 2*0.019),
    ("C",  "S" ) : (1.630 - 2*0.014,  1.863 + 2*0.015),
    ("C",  "SE") : (1.893 - 2*0.013,  1.970 + 2*0.032),
    ("C",  "SI") : (1.837 - 2*0.012,  1.888 + 2*0.023),
    ("CL", "O" ) : (1.414 - 2*0.026,  1.414 + 2*0.026),
    ("CL", "P" ) : (1.997 - 2*0.035,  2.008 + 2*0.035),
    ("CL", "S" ) : (2.072 - 2*0.023,  2.072 + 2*0.023),
    ("CL", "SI") : (2.072 - 2*0.009,  2.072 + 2*0.009),
    ("F",  "N" ) : (1.406 - 2*0.016,  1.406 + 2*0.016),
    ("F",  "P" ) : (1.495 - 2*0.016,  1.579 + 2*0.025),
    ("F",  "S" ) : (1.640 - 2*0.011,  1.640 + 2*0.011),
    ("F",  "SI") : (1.588 - 2*0.014,  1.694 + 2*0.013),
    ("H",  "N" ) : (1.009 - 2*0.022,  1.033 + 2*0.022),
    ("H",  "O" ) : (0.967 - 2*0.010,  1.015 + 2*0.017),
    ("I",  "O" ) : (2.144 - 2*0.028,  2.144 + 2*0.028),
    ("N",  "N" ) : (1.124 - 2*0.015,  1.454 + 2*0.021),
    ("N",  "O" ) : (1.210 - 2*0.011,  1.463 + 2*0.012),
    ("N",  "P" ) : (1.571 - 2*0.013,  1.697 + 2*0.015),
    ("N",  "S" ) : (1.541 - 2*0.022,  1.710 + 2*0.019),
    ("N",  "SI") : (1.711 - 2*0.019,  1.748 + 2*0.022),
    ("O",  "P" ) : (1.449 - 2*0.007,  1.689 + 2*0.024),
    ("O",  "S" ) : (1.423 - 2*0.008,  1.580 + 2*0.015),
    ("O",  "SI") : (1.622 - 2*0.014,  1.680 + 2*0.008),
    ("P",  "P" ) : (2.214 - 2*0.022,  2.214 + 2*0.022),
    ("P",  "S" ) : (1.913 - 2*0.014,  1.954 + 2*0.005),
    ("P",  "SE") : (2.093 - 2*0.019,  2.093 + 2*0.019),
    ("P",  "SI") : (2.264 - 2*0.019,  2.264 + 2*0.019),
    ("S",  "S" ) : (1.897 - 2*0.012,  2.070 + 2*0.022),
    ("S",  "SE") : (2.193 - 2*0.015,  2.193 + 2*0.015),
    ("S",  "SI") : (2.145 - 2*0.020,  2.145 + 2*0.020),
    ("SE", "SE") : (2.340 - 2*0.024,  2.340 + 2*0.024),
    ("SI", "SE") : (2.359 - 2*0.012,  2.359 + 2*0.012),
}
# fmt: on


def connect_via_distances(
    atoms,
    distance_range=None,
    inter_residue=True,
    default_bond_type=BondType.ANY,
    periodic=False,
):
    """
    connect_via_distances(atoms, distance_range=None, inter_residue=True,
                          default_bond_type=BondType.ANY, periodic=False)

    Create a :class:`BondList` for a given atom array, based on
    pairwise atom distances.

    A :attr:`BondType.ANY`, bond is created for two atoms within the
    same residue, if the distance between them is within the expected
    bond distance range.
    Bonds between two adjacent residues are created for the atoms
    expected to connect these residues, i.e. ``'C'`` and ``'N'`` for
    peptides and ``"O3'"`` and ``'P'`` for nucleotides.

    Parameters
    ----------
    atoms : AtomArray
        The structure to create the :class:`BondList` for.
    distance_range : dict of tuple(str, str) -> tuple(float, float), optional
        Custom minimum and maximum bond distances.
        The dictionary keys are tuples of chemical elements representing
        the atoms to be potentially bonded.
        The order of elements within each tuple does not matter.
        The dictionary values are the minimum and maximum bond distance,
        respectively, for the given combination of elements.
        This parameter updates the default dictionary.
        Hence, the default bond distances for missing element pairs are
        still taken from the default dictionary.
        The default bond distances are taken from :footcite:`Allen1987`.
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

    See Also
    --------
    connect_via_residue_names

    Notes
    -----
    This method might miss bonds, if the bond distance is unexpectedly
    high or low, or it might create false bonds, if two atoms within a
    residue are accidentally in the right distance.
    A more accurate method for determining bonds is
    :func:`connect_via_residue_names()`.

    References
    ----------

    .. footbibliography::
    """
    from biotite.structure.atoms import AtomArray
    from biotite.structure.geometry import distance
    from biotite.structure.residues import get_residue_starts

    if not isinstance(atoms, AtomArray):
        raise TypeError(f"Expected 'AtomArray', not '{type(atoms).__name__}'")
    if periodic:
        if atoms.box is None:
            raise BadStructureError("Atom array has no box")
        box = atoms.box
    else:
        box = None

    # Prepare distance dictionary...
    if distance_range is None:
        distance_range = {}
    # Merge default and custom entries
    dist_ranges = {}
    for key, val in itertools.chain(
        _DEFAULT_DISTANCE_RANGE.items(), distance_range.items()
    ):
        element1, element2 = key
        # Add entries for both element orders
        dist_ranges[(element1.upper(), element2.upper())] = val
        dist_ranges[(element2.upper(), element1.upper())] = val

    bonds = []
    coord = atoms.coord
    elements = atoms.element

    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    # Omit exclusive stop in 'residue_starts'
    for i in range(len(residue_starts) - 1):
        curr_start_i = residue_starts[i]
        next_start_i = residue_starts[i + 1]

        elements_in_res = elements[curr_start_i:next_start_i]
        coord_in_res = coord[curr_start_i:next_start_i]
        # Matrix containing all pairwise atom distances in the residue
        distances = distance(
            coord_in_res[:, np.newaxis, :], coord_in_res[np.newaxis, :, :], box
        )
        for atom_index1 in range(len(elements_in_res)):
            for atom_index2 in range(atom_index1):
                dist_range = dist_ranges.get(
                    (elements_in_res[atom_index1], elements_in_res[atom_index2])
                )
                if dist_range is None:
                    # No bond distance entry for this element
                    # combination -> skip
                    continue
                else:
                    min_dist, max_dist = dist_range
                dist = distances[atom_index1, atom_index2]
                if dist >= min_dist and dist <= max_dist:
                    # Convert BondType to int if necessary
                    bt_int = (
                        int(default_bond_type)
                        if hasattr(default_bond_type, "__int__")
                        else default_bond_type
                    )
                    bonds.append(
                        (
                            curr_start_i + atom_index1,
                            curr_start_i + atom_index2,
                            bt_int,
                        )
                    )

    if bonds:
        bond_list = BondList(atoms.array_length(), np.array(bonds, dtype=np.int64))
    else:
        bond_list = BondList(atoms.array_length())

    if inter_residue:
        inter_bonds = _connect_inter_residue(atoms, residue_starts)
        if default_bond_type == BondType.ANY:
            # As all bonds should be of type ANY, convert also
            # inter-residue bonds to ANY
            inter_bonds.remove_bond_order()
        return bond_list.merge(inter_bonds)
    else:
        return bond_list


def connect_via_residue_names(atoms, inter_residue=True, custom_bond_dict=None):
    """
    connect_via_residue_names(atoms, inter_residue=True, custom_bond_dict=None)

    Create a :class:`BondList` for a given atom array (stack), based on
    the deposited bonds for each residue in the RCSB ``components.cif``
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
        If given, these bonds are used instead of the bonds read from
        ``components.cif``.

    Returns
    -------
    BondList
        The created bond list.
        No bonds are added for residues that are not found in
        ``components.cif``.

    See Also
    --------
    connect_via_distances

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
    from biotite.structure.info.bonds import bonds_in_residue
    from biotite.structure.residues import get_residue_starts

    bonds = []
    atom_names = atoms.atom_name
    res_names = atoms.res_name

    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    # Omit exclusive stop in 'residue_starts'
    for res_i in range(len(residue_starts) - 1):
        curr_start_i = residue_starts[res_i]
        next_start_i = residue_starts[res_i + 1]

        if custom_bond_dict is None:
            bond_dict_for_res = bonds_in_residue(res_names[curr_start_i])
        else:
            bond_dict_for_res = custom_bond_dict.get(res_names[curr_start_i], {})

        atom_names_in_res = atom_names[curr_start_i:next_start_i]
        for (atom_name1, atom_name2), bond_type in bond_dict_for_res.items():
            atom_indices1 = np.where(atom_names_in_res == atom_name1)[0].astype(
                np.int64, copy=False
            )
            atom_indices2 = np.where(atom_names_in_res == atom_name2)[0].astype(
                np.int64, copy=False
            )
            # In rare cases the same atom name may appear multiple times
            # (e.g. in altlocs)
            # -> create all possible bond combinations
            # Convert BondType to int if necessary
            bt_int = int(bond_type) if hasattr(bond_type, "__int__") else bond_type
            for i in range(len(atom_indices1)):
                for j in range(len(atom_indices2)):
                    bonds.append(
                        (
                            curr_start_i + atom_indices1[i],
                            curr_start_i + atom_indices2[j],
                            bt_int,
                        )
                    )

    if bonds:
        bond_list = BondList(atoms.array_length(), np.array(bonds, dtype=np.int64))
    else:
        bond_list = BondList(atoms.array_length())

    if inter_residue:
        inter_bonds = _connect_inter_residue(atoms, residue_starts)
        return bond_list.merge(inter_bonds)
    else:
        return bond_list


_PEPTIDE_LINKS = ["PEPTIDE LINKING", "L-PEPTIDE LINKING", "D-PEPTIDE LINKING"]
_NUCLEIC_LINKS = ["RNA LINKING", "DNA LINKING"]


def _connect_inter_residue(atoms, residue_starts):
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
    from biotite.structure.info.misc import link_type

    bonds = []
    atom_names = atoms.atom_name
    res_names = atoms.res_name
    res_ids = atoms.res_id
    chain_ids = atoms.chain_id

    # Iterate over all starts excluding:
    #   - the last residue and
    #   - exclusive end index of 'atoms'
    for i in range(len(residue_starts) - 2):
        curr_start_i = residue_starts[i]
        next_start_i = residue_starts[i + 1]
        after_next_start_i = residue_starts[i + 2]

        # Check if the current and next residue is in the same chain
        if chain_ids[next_start_i] != chain_ids[curr_start_i]:
            continue
        # Check if the current and next residue
        # have consecutive residue IDs
        # (Same residue ID is also possible if insertion code is used)
        if res_ids[next_start_i] - res_ids[curr_start_i] > 1:
            continue

        # Get link type for this residue from RCSB components.cif
        curr_link = link_type(res_names[curr_start_i])
        next_link = link_type(res_names[next_start_i])

        if curr_link in _PEPTIDE_LINKS and next_link in _PEPTIDE_LINKS:
            curr_connect_atom_name = "C"
            next_connect_atom_name = "N"
        elif curr_link in _NUCLEIC_LINKS and next_link in _NUCLEIC_LINKS:
            curr_connect_atom_name = "O3'"
            next_connect_atom_name = "P"
        else:
            # Create no bond if the connection types of consecutive
            # residues are not compatible
            continue

        # Index in atom array for atom name in current residue
        # Addition of 'curr_start_i' is necessary, as only a slice of
        # 'atom_names' is taken, beginning at 'curr_start_i'
        curr_connect_indices = (
            curr_start_i
            + np.where(atom_names[curr_start_i:next_start_i] == curr_connect_atom_name)[
                0
            ]
        )
        # Index in atom array for atom name in next residue
        next_connect_indices = (
            next_start_i
            + np.where(
                atom_names[next_start_i:after_next_start_i] == next_connect_atom_name
            )[0]
        )
        if len(curr_connect_indices) == 0 or len(next_connect_indices) == 0:
            # The connector atoms are not found in the adjacent residues
            # -> skip this bond
            continue

        bonds.append(
            (curr_connect_indices[0], next_connect_indices[0], int(BondType.SINGLE))
        )

    if bonds:
        return BondList(atoms.array_length(), np.array(bonds, dtype=np.int64))
    else:
        return BondList(atoms.array_length())


def find_connected(bond_list, root, as_mask=False):
    """
    find_connected(bond_list, root, as_mask=False)

    Get indices to all atoms that are directly or indirectly connected
    to the root atom indicated by the given index.

    An atom is *connected* to the `root` atom, if that atom is reachable
    by traversing an arbitrary number of bonds, starting from the
    `root`.
    Effectively, this means that all atoms are *connected* to `root`,
    that are in the same molecule as `root`.
    Per definition `root` is also *connected* to itself.

    Parameters
    ----------
    bond_list : BondList
        The reference bond list.
    root : int
        The index of the root atom.
    as_mask : bool, optional
        If true, the connected atom indices are returned as boolean
        mask.
        By default, the connected atom indices are returned as integer
        array.

    Returns
    -------
    connected : ndarray, dtype=int or ndarray, dtype=bool
        Either a boolean mask or an integer array, representing the
        connected atoms.
        In case of a boolean mask: ``connected[i] == True``, if the atom
        with index ``i`` is connected.

    Examples
    --------
    Consider a system with 4 atoms, where only the last atom is not
    bonded with the other ones (``0-1-2 3``):

    >>> bonds = BondList(4)
    >>> bonds.add_bond(0, 1)
    >>> bonds.add_bond(1, 2)
    >>> print(find_connected(bonds, 0))
    [0 1 2]
    >>> print(find_connected(bonds, 1))
    [0 1 2]
    >>> print(find_connected(bonds, 2))
    [0 1 2]
    >>> print(find_connected(bonds, 3))
    [3]
    """
    all_bonds, _ = bond_list.get_all_bonds()

    if root >= bond_list.get_atom_count():
        raise ValueError(
            f"Root atom index {root} is out of bounds for bond list "
            f"representing {bond_list.get_atom_count()} atoms"
        )

    is_connected_mask = np.zeros(bond_list.get_atom_count(), dtype=np.uint8)
    # Find connections in a recursive way,
    # by visiting all atoms that are reachable by a bond
    _find_connected_recursive(root, is_connected_mask, all_bonds)
    if as_mask:
        return is_connected_mask.astype(bool)
    else:
        return np.where(is_connected_mask)[0]


def _find_connected_recursive(index, is_connected_mask, all_bonds):
    """Recursive helper for find_connected."""
    if is_connected_mask[index]:
        # This atom has already been visited
        # -> exit condition
        return
    is_connected_mask[index] = 1

    for j in range(all_bonds.shape[1]):
        connected_index = all_bonds[index, j]
        if connected_index == -1:
            # Ignore padding values
            continue
        _find_connected_recursive(connected_index, is_connected_mask, all_bonds)


def find_rotatable_bonds(bonds):
    """
    find_rotatable_bonds(bonds)

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
    cycles = nx.algorithms.cycles.cycle_basis(bond_graph)

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
