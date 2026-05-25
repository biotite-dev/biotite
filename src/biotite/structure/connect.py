# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "connect_via_distances",
    "connect_via_residue_names",
    "find_connected",
    "find_rotatable_bonds",
]

import itertools
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
from biotite.structure.bonds import BondList, BondType
from biotite.structure.error import BadStructureError

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
        Connect atoms based on their residue names, which is more accurate when the
        molecule is part of the *Chemical Component Dictionary* (CCD).

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
    from biotite.structure.residues import get_residue_starts

    if not isinstance(atoms, AtomArray):
        raise TypeError(f"Expected 'AtomArray', not '{type(atoms).__name__}'")
    if periodic:
        if atoms.box is None:
            raise BadStructureError("Atom array has no box")

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

    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)

    if periodic:
        bond_list = _connect_via_distances_periodic(
            atoms, residue_starts, dist_ranges, default_bond_type
        )
    else:
        bond_list = rust_connect_via_distances(
            atoms.coord,
            atoms.element.tolist(),
            residue_starts,
            dist_ranges,
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
    atoms, residue_starts, dist_ranges, default_bond_type
):
    """
    Fallback for :func:`connect_via_distances` with periodic boundary conditions.
    Uses :func:`biotite.structure.geometry.distance` for periodic distance computation.
    """
    from biotite.structure.geometry import distance

    box = atoms.box
    coord = atoms.coord
    elements = atoms.element
    bt_int = int(default_bond_type)

    bonds = []
    for start, stop in itertools.pairwise(residue_starts):
        elements_in_res = elements[start:stop]
        coord_in_res = coord[start:stop]
        distances = distance(
            coord_in_res[:, np.newaxis, :], coord_in_res[np.newaxis, :, :], box
        )
        for atom_index1 in range(len(elements_in_res)):
            for atom_index2 in range(atom_index1):
                dist_range = dist_ranges.get(
                    (elements_in_res[atom_index1], elements_in_res[atom_index2])
                )
                if dist_range is None:
                    continue
                dist = distances[atom_index1, atom_index2]
                if dist_range[0] <= dist <= dist_range[1]:
                    bonds.append((start + atom_index1, start + atom_index2, bt_int))

    if bonds:
        return BondList(atoms.array_length(), np.stack(bonds, axis=0, dtype=np.int64))


def connect_via_residue_names(atoms, inter_residue=True, custom_bond_dict=None):
    """
    connect_via_residue_names(atoms, inter_residue=True, custom_bond_dict=None)

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


def _bond_dict_to_list(bond_dict):
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
