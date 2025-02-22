# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module allows efficient search of atoms in a defined radius around
a location.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["BondList", "BondType",
           "connect_via_distances", "connect_via_residue_names",
           "find_connected", "find_rotatable_bonds"]

cimport cython
cimport numpy as np
from libc.stdlib cimport free, realloc

from collections.abc import Sequence
import itertools
import numbers
from enum import IntEnum
import networkx as nx
import numpy as np
from .error import BadStructureError
from ..copyable import Copyable

ctypedef np.uint64_t ptr
ctypedef np.uint8_t  uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64
ctypedef np.int8_t   int8
ctypedef np.int16_t  int16
ctypedef np.int32_t  int32
ctypedef np.int64_t  int64


ctypedef fused IndexType:
    uint8
    uint16
    uint32
    uint64
    int8
    int16
    int32
    int64


class BondType(IntEnum):
    """
    This enum type represents the type of a chemical bond.

        - `ANY` - Used if the actual type is unknown
        - `SINGLE` - Single bond
        - `DOUBLE` - Double bond
        - `TRIPLE` - Triple bond
        - `QUADRUPLE` - A quadruple bond
        - `AROMATIC_SINGLE` - Aromatic bond with a single formal bond
        - `AROMATIC_DOUBLE` - Aromatic bond with a double formal bond
        - `AROMATIC_TRIPLE` - Aromatic bond with a triple formal bond
        - `AROMATIC` - Aromatic bond without specification of the formal bond
        - `COORDINATION` - Coordination complex involving a metal atom
    """
    ANY = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    QUADRUPLE = 4
    AROMATIC_SINGLE = 5
    AROMATIC_DOUBLE = 6
    AROMATIC_TRIPLE = 7
    COORDINATION = 8
    AROMATIC = 9


    def without_aromaticity(self):
        """
        Remove aromaticity from the bond type.

        :attr:`BondType.AROMATIC_{ORDER}` is converted into
        :attr:`BondType.{ORDER}`.

        Returns
        -------
        new_bond_type : BondType
            The :class:`BondType` without aromaticity.

        Examples
        --------

        >>> print(BondType.AROMATIC_DOUBLE.without_aromaticity().name)
        DOUBLE
        """
        if self == BondType.AROMATIC_SINGLE:
            return BondType.SINGLE
        elif self == BondType.AROMATIC_DOUBLE:
            return BondType.DOUBLE
        elif self == BondType.AROMATIC_TRIPLE:
            return BondType.TRIPLE
        elif self == BondType.AROMATIC:
            return BondType.ANY
        else:
            return self


@cython.boundscheck(False)
@cython.wraparound(False)
class BondList(Copyable):
    """
    __init__(atom_count, bonds=None)

    A bond list stores indices of atoms
    (usually of an :class:`AtomArray` or :class:`AtomArrayStack`)
    that form chemical bonds together with the type (or order) of the
    bond.

    Internally the bonds are stored as *n x 3* :class:`ndarray`.
    For each row, the first column specifies the index of the first
    atom, the second column the index of the second atom involved in the
    bond.
    The third column stores an integer that is interpreted as member
    of the the :class:`BondType` enum, that specifies the order of the
    bond.

    When indexing a :class:`BondList`, the index is not forwarded to the
    internal :class:`ndarray`. Instead the indexing behavior is
    consistent with indexing an :class:`AtomArray` or
    :class:`AtomArrayStack`:
    Bonds with at least one atom index that is not covered by the index
    are removed, atom indices that occur after an uncovered atom index
    move up.
    Effectively, this means that after indexing an :class:`AtomArray`
    and a :class:`BondList` with the same index, the atom indices in the
    :class:`BondList` will still point to the same atoms in the
    :class:`AtomArray`.
    Indexing a :class:`BondList` with a single integer is equivalent
    to calling :func:`get_bonds()`.

    The same consistency applies to adding :class:`BondList` instances
    via the '+' operator:
    The atom indices of the second :class:`BondList` are increased by
    the atom count of the first :class:`BondList` and then both
    :class:`BondList` objects are merged.

    Parameters
    ----------
    atom_count : int
        A positive integer, that specifies the number of atoms the
        :class:`BondList` refers to
        (usually the length of an atom array (stack)).
        Effectively, this value is the exclusive maximum for the indices
        stored in the :class:`BondList`.
    bonds : ndarray, shape=(n,2) or shape=(n,3), dtype=int, optional
        This array contains the indices of atoms which are bonded:
        For each row, the first column specifies the first atom,
        the second row the second atom involved in a chemical bond.
        If an *n x 3* array is provided, the additional column
        specifies a :class:`BondType` instead of :attr:`BondType.ANY`.
        By default, the created :class:`BondList` is empty.

    Notes
    -----
    When initially providing the bonds as :class:`ndarray`, the input is
    sanitized: Redundant bonds are removed, and each bond entry is
    sorted so that the lower one of the two atom indices is in the first
    column.
    If a bond appears multiple times with different bond types, the
    first bond takes precedence.

    Examples
    --------

    Construct a :class:`BondList`, where a central atom (index 1) is
    connected to three other atoms (index 0, 3 and 4):

    >>> bond_list = BondList(5, np.array([(1,0),(1,3),(1,4)]))
    >>> print(bond_list)
    [[0 1 0]
     [1 3 0]
     [1 4 0]]

    Remove the first atom (index 0) via indexing:
    The bond containing index 0 is removed, since the corresponding atom
    does not exist anymore. Since all other atoms move up in their
    position, the indices in the bond list are decreased by one:

    >>> bond_list = bond_list[1:]
    >>> print(bond_list)
    [[0 2 0]
     [0 3 0]]

    :class:`BondList` objects can be associated to an :class:`AtomArray`
    or :class:`AtomArrayStack`.
    The following snippet shows this for a benzene molecule:

    >>> benzene = AtomArray(12)
    >>> # Omit filling most required annotation categories for brevity
    >>> benzene.atom_name = np.array(
    ...     ["C1", "C2", "C3", "C4", "C5", "C6", "H1", "H2", "H3", "H4", "H5", "H6"]
    ... )
    >>> benzene.bonds = BondList(
    ...     benzene.array_length(),
    ...     np.array([
    ...         # Bonds between carbon atoms in the ring
    ...         (0,  1, BondType.AROMATIC_SINGLE),
    ...         (1,  2, BondType.AROMATIC_DOUBLE),
    ...         (2,  3, BondType.AROMATIC_SINGLE),
    ...         (3,  4, BondType.AROMATIC_DOUBLE),
    ...         (4,  5, BondType.AROMATIC_SINGLE),
    ...         (5,  0, BondType.AROMATIC_DOUBLE),
    ...         # Bonds between carbon and hydrogen
    ...         (0,  6, BondType.SINGLE),
    ...         (1,  7, BondType.SINGLE),
    ...         (2,  8, BondType.SINGLE),
    ...         (3,  9, BondType.SINGLE),
    ...         (4, 10, BondType.SINGLE),
    ...         (5, 11, BondType.SINGLE),
    ...     ])
    ... )
    >>> for i, j, bond_type in benzene.bonds.as_array():
    ...     print(
    ...         f"{BondType(bond_type).name} bond between "
    ...         f"{benzene.atom_name[i]} and {benzene.atom_name[j]}"
    ...     )
    AROMATIC_SINGLE bond between C1 and C2
    AROMATIC_DOUBLE bond between C2 and C3
    AROMATIC_SINGLE bond between C3 and C4
    AROMATIC_DOUBLE bond between C4 and C5
    AROMATIC_SINGLE bond between C5 and C6
    AROMATIC_DOUBLE bond between C1 and C6
    SINGLE bond between C1 and H1
    SINGLE bond between C2 and H2
    SINGLE bond between C3 and H3
    SINGLE bond between C4 and H4
    SINGLE bond between C5 and H5
    SINGLE bond between C6 and H6

    Obtain the bonded atoms for the :math:`C_1`:

    >>> bonds, types = benzene.bonds.get_bonds(0)
    >>> print(bonds)
    [1 5 6]
    >>> print(types)
    [5 6 1]
    >>> print(f"C1 is bonded to {', '.join(benzene.atom_name[bonds])}")
    C1 is bonded to C2, C6, H1

    Cut the benzene molecule in half.
    Although the first half of the atoms are missing the indices of
    the cropped :class:`BondList` still represents the bonds of the
    remaining atoms:

    >>> half_benzene = benzene[
    ...     np.isin(benzene.atom_name, ["C4", "C5", "C6", "H4", "H5", "H6"])
    ... ]
    >>> for i, j, bond_type in half_benzene.bonds.as_array():
    ...     print(
    ...         f"{BondType(bond_type).name} bond between "
    ...         f"{half_benzene.atom_name[i]} and {half_benzene.atom_name[j]}"
    ...     )
    AROMATIC_DOUBLE bond between C4 and C5
    AROMATIC_SINGLE bond between C5 and C6
    SINGLE bond between C4 and H4
    SINGLE bond between C5 and H5
    SINGLE bond between C6 and H6
    """

    def __init__(self, uint32 atom_count, np.ndarray bonds=None):
        self._atom_count = atom_count

        if bonds is not None and len(bonds) > 0:
            if bonds.ndim != 2:
                raise ValueError("Expected a 2D-ndarray for input bonds")

            self._bonds = np.zeros((bonds.shape[0], 3), dtype=np.uint32)
            if bonds.shape[1] == 3:
                # Input contains bonds (index 0 and 1)
                # including the bond type value (index 2)
                # Bond indices:
                self._bonds[:,:2] = np.sort(
                    # Indices are sorted per bond
                    # so that the lower index is at the first position
                    _to_positive_index_array(bonds[:,:2], atom_count), axis=1
                )
                # Bond type:
                if (bonds[:, 2] >= len(BondType)).any():
                    raise ValueError(
                        f"BondType {np.max(bonds[:, 2])} is invalid"
                    )
                self._bonds[:,2] = bonds[:, 2]

                # Indices are sorted per bond
                # so that the lower index is at the first position
            elif bonds.shape[1] == 2:
                # Input contains the bonds without bond type
                # -> Default: Set bond type ANY (0)
                self._bonds[:,:2] = np.sort(
                    # Indices are sorted per bond
                    # so that the lower index is at the first position
                    _to_positive_index_array(bonds[:,:2], atom_count), axis=1
                )
            else:
                raise ValueError(
                    "Input array containing bonds must be either of shape "
                    "(n,2) or (n,3)"
                )
            self._remove_redundant_bonds()
            self._max_bonds_per_atom = self._get_max_bonds_per_atom()

        else:
            # Create empty bond list
            self._bonds = np.zeros((0, 3), dtype=np.uint32)
            self._max_bonds_per_atom = 0

    @staticmethod
    def concatenate(bonds_lists):
        """
        Concatenate multiple :class:`BondList` objects into a single
        :class:`BondList`, respectively.

        Parameters
        ----------
        bonds_lists : iterable object of BondList
            The bond lists to be concatenated.

        Returns
        -------
        concatenated_bonds : BondList
            The concatenated bond lists.

        Examples
        --------

        >>> bonds1 = BondList(2, np.array([(0, 1)]))
        >>> bonds2 = BondList(3, np.array([(0, 1), (0, 2)]))
        >>> merged_bonds = BondList.concatenate([bonds1, bonds2])
        >>> print(merged_bonds.get_atom_count())
        5
        >>> print(merged_bonds.as_array()[:, :2])
        [[0 1]
         [2 3]
         [2 4]]
        """
        # Ensure that the bonds_lists can be iterated over multiple times
        if not isinstance(bonds_lists, Sequence):
            bonds_lists = list(bonds_lists)

        cdef np.ndarray merged_bonds = np.concatenate(
            [bond_list._bonds for bond_list in bonds_lists]
        )
        # Offset the indices of appended bonds list
        # (consistent with addition of AtomArray)
        cdef int start = 0, stop = 0
        cdef int cum_atom_count = 0
        for bond_list in bonds_lists:
            stop = start + bond_list._bonds.shape[0]
            merged_bonds[start : stop, :2] += cum_atom_count
            cum_atom_count += bond_list._atom_count
            start = stop

        cdef merged_bond_list = BondList(cum_atom_count)
        # Array is not used in constructor to prevent unnecessary
        # maximum and redundant bond calculation
        merged_bond_list._bonds = merged_bonds
        merged_bond_list._max_bonds_per_atom = max(
            [bond_list._max_bonds_per_atom for bond_list in bonds_lists]
        )
        return merged_bond_list

    def __copy_create__(self):
        # Create empty bond list to prevent
        # unnecessary removal of redundant atoms
        # and calculation of maximum bonds per atom
        return BondList(self._atom_count)

    def __copy_fill__(self, clone):
        # The bonds are added here
        clone._bonds = self._bonds.copy()
        clone._max_bonds_per_atom = self._max_bonds_per_atom

    def offset_indices(self, int offset):
        """
        offset_indices(offset)

        Increase all atom indices in the :class:`BondList` by the given
        offset.

        Implicitly this increases the atom count.

        Parameters
        ----------
        offset : int
            The atom indices are increased by this value.
            Must be positive.

        Examples
        --------

        >>> bond_list = BondList(5, np.array([(1,0),(1,3),(1,4)]))
        >>> print(bond_list)
        [[0 1 0]
         [1 3 0]
         [1 4 0]]
        >>> bond_list.offset_indices(2)
        >>> print(bond_list)
        [[2 3 0]
         [3 5 0]
         [3 6 0]]
        """
        if offset < 0:
            raise ValueError("Offest must be positive")
        self._bonds[:,:2] += offset
        self._atom_count += offset

    def as_array(self):
        """
        as_array()

        Obtain a copy of the internal :class:`ndarray`.

        Returns
        -------
        array : ndarray, shape=(n,3), dtype=np.uint32
            Copy of the internal :class:`ndarray`.
            For each row, the first column specifies the index of the
            first atom, the second column the index of the second atom
            involved in the bond.
            The third column stores the :class:`BondType`.
        """
        return self._bonds.copy()

    def as_set(self):
        """
        as_set()

        Obtain a set representation of the :class:`BondList`.

        Returns
        -------
        bond_set : set of tuple(int, int, int)
            A set of tuples.
            Each tuple represents one bond:
            The first integer represents the first atom,
            the second integer represents the second atom,
            the third integer represents the :class:`BondType`.
        """
        cdef uint32[:,:] all_bonds_v = self._bonds
        cdef int i
        cdef set bond_set = set()
        for i in range(all_bonds_v.shape[0]):
            bond_set.add(
                (all_bonds_v[i,0], all_bonds_v[i,1], all_bonds_v[i,2])
            )
        return bond_set

    def as_graph(self):
        """
        as_graph()

        Obtain a graph representation of the :class:`BondList`.

        Returns
        -------
        bond_set : Graph
            A *NetworkX* :class:`Graph`.
            The atom indices are nodes, the bonds are edges.
            Each edge has a ``"bond_type"`` attribute containing the
            :class:`BondType`.

        Examples
        --------

        >>> bond_list = BondList(5, np.array([(1,0,2), (1,3,1), (1,4,1)]))
        >>> graph = bond_list.as_graph()
        >>> print(graph.nodes)
        [0, 1, 3, 4]
        >>> print(graph.edges)
        [(0, 1), (1, 3), (1, 4)]
        >>> for i, j in graph.edges:
        ...     print(i, j, graph.get_edge_data(i, j))
        0 1 {'bond_type': <BondType.DOUBLE: 2>}
        1 3 {'bond_type': <BondType.SINGLE: 1>}
        1 4 {'bond_type': <BondType.SINGLE: 1>}
        """
        cdef int i

        cdef uint32[:,:] all_bonds_v = self._bonds

        g = nx.Graph()
        cdef list edges = [None] * all_bonds_v.shape[0]
        for i in range(all_bonds_v.shape[0]):
            edges[i] = (
                all_bonds_v[i,0], all_bonds_v[i,1],
                {"bond_type": BondType(all_bonds_v[i,2])}
            )
        g.add_edges_from(edges)
        return g

    def remove_aromaticity(self):
        """
        Remove aromaticity from the bond types.

        :attr:`BondType.AROMATIC_{ORDER}` is converted into
        :attr:`BondType.{ORDER}`.

        Examples
        --------

        >>> bond_list = BondList(3)
        >>> bond_list.add_bond(0, 1, BondType.AROMATIC_SINGLE)
        >>> bond_list.add_bond(1, 2, BondType.AROMATIC_DOUBLE)
        >>> bond_list.remove_aromaticity()
        >>> for i, j, bond_type in bond_list.as_array():
        ...     print(i, j, BondType(bond_type).name)
        0 1 SINGLE
        1 2 DOUBLE
        """
        bond_types = self._bonds[:,2]
        for aromatic_type, non_aromatic_type in [
            (BondType.AROMATIC_SINGLE, BondType.SINGLE),
            (BondType.AROMATIC_DOUBLE, BondType.DOUBLE),
            (BondType.AROMATIC_TRIPLE, BondType.TRIPLE),
            (BondType.AROMATIC, BondType.ANY),
        ]:
            bond_types[bond_types == aromatic_type] = non_aromatic_type

    def remove_bond_order(self):
        """
        Convert all bonds to :attr:`BondType.ANY`.
        """
        self._bonds[:,2] = BondType.ANY

    def get_atom_count(self):
        """
        get_atom_count()

        Get the atom count.

        Returns
        -------
        atom_count : int
            The atom count.
        """
        return self._atom_count

    def get_bond_count(self):
        """
        get_bond_count()

        Get the amount of bonds.

        Returns
        -------
        bond_count : int
            The amount of bonds. This is equal to the length of the
            internal :class:`ndarray` containing the bonds.
        """
        return len(self._bonds)

    def get_bonds(self, int32 atom_index):
        """
        get_bonds(atom_index)

        Obtain the indices of the atoms bonded to the atom with the
        given index as well as the corresponding bond types.

        Parameters
        ----------
        atom_index : int
            The index of the atom to get the bonds for.

        Returns
        -------
        bonds : np.ndarray, dtype=np.uint32, shape=(k,)
            The indices of connected atoms.
        bond_types : np.ndarray, dtype=np.uint8, shape=(k,)
            Array of integers, interpreted as :class:`BondType`
            instances.
            This array specifies the type (or order) of the bonds to
            the connected atoms.

        Examples
        --------

        >>> bond_list = BondList(5, np.array([(1,0),(1,3),(1,4)]))
        >>> bonds, types = bond_list.get_bonds(1)
        >>> print(bonds)
        [0 3 4]
        """
        cdef int i=0, j=0

        cdef uint32 index = _to_positive_index(atom_index, self._atom_count)

        cdef uint32[:,:] all_bonds_v = self._bonds
        # Pessimistic array allocation:
        # assume size is equal to the atom with most bonds
        cdef np.ndarray bonds = np.zeros(self._max_bonds_per_atom,
                                         dtype=np.uint32)
        cdef uint32[:] bonds_v = bonds
        cdef np.ndarray bond_types = np.zeros(self._max_bonds_per_atom,
                                              dtype=np.uint8)
        cdef uint8[:] bond_types_v = bond_types

        for i in range(all_bonds_v.shape[0]):
            # If a bond is found for the desired atom index
            # at the first or second position of the bond,
            # then append the index of the respective other position
            if all_bonds_v[i,0] == index:
                bonds_v[j] = all_bonds_v[i,1]
                bond_types_v[j] = all_bonds_v[i,2]
                j += 1
            elif all_bonds_v[i,1] == index:
                bonds_v[j] = all_bonds_v[i,0]
                bond_types_v[j] = all_bonds_v[i,2]
                j += 1

        # Trim to correct size
        bonds = bonds[:j]
        bond_types = bond_types[:j]

        return bonds, bond_types


    def get_all_bonds(self):
        """
        get_all_bonds()

        For each atom index, give the indices of the atoms bonded to
        this atom as well as the corresponding bond types.

        Returns
        -------
        bonds : np.ndarray, dtype=np.uint32, shape=(n,k)
            The indices of connected atoms.
            The first dimension represents the atoms,
            the second dimension represents the indices of atoms bonded
            to the respective atom.
            Atoms can have have different numbers of atoms bonded to
            them.
            Therefore, the length of the second dimension *k* is equal
            to the maximum number of bonds for an atom in this
            :class:`BondList`.
            For atoms with less bonds, the corresponding entry in the
            array is padded with ``-1`` values.
        bond_types : np.ndarray, dtype=np.uint32, shape=(n,k)
            Array of integers, interpreted as :class:`BondType`
            instances.
            This array specifies the bond type (or order) corresponding
            to the returned `bonds`.
            It uses the same ``-1``-padding.

        Examples
        --------

        >>> # BondList for benzene
        >>> bond_list = BondList(
        ...     12,
        ...     np.array([
        ...         # Bonds between the carbon atoms in the ring
        ...         (0,  1, BondType.AROMATIC_SINGLE),
        ...         (1,  2, BondType.AROMATIC_DOUBLE),
        ...         (2,  3, BondType.AROMATIC_SINGLE),
        ...         (3,  4, BondType.AROMATIC_DOUBLE),
        ...         (4,  5, BondType.AROMATIC_SINGLE),
        ...         (5,  0, BondType.AROMATIC_DOUBLE),
        ...         # Bonds between carbon and hydrogen
        ...         (0,  6, BondType.SINGLE),
        ...         (1,  7, BondType.SINGLE),
        ...         (2,  8, BondType.SINGLE),
        ...         (3,  9, BondType.SINGLE),
        ...         (4, 10, BondType.SINGLE),
        ...         (5, 11, BondType.SINGLE),
        ...     ])
        ... )
        >>> bonds, types = bond_list.get_all_bonds()
        >>> print(bonds)
        [[ 1  5  6]
         [ 0  2  7]
         [ 1  3  8]
         [ 2  4  9]
         [ 3  5 10]
         [ 4  0 11]
         [ 0 -1 -1]
         [ 1 -1 -1]
         [ 2 -1 -1]
         [ 3 -1 -1]
         [ 4 -1 -1]
         [ 5 -1 -1]]
        >>> print(types)
        [[ 5  6  1]
         [ 5  6  1]
         [ 6  5  1]
         [ 5  6  1]
         [ 6  5  1]
         [ 5  6  1]
         [ 1 -1 -1]
         [ 1 -1 -1]
         [ 1 -1 -1]
         [ 1 -1 -1]
         [ 1 -1 -1]
         [ 1 -1 -1]]
        >>> for i in range(bond_list.get_atom_count()):
        ...     bonds_for_atom = bonds[i]
        ...     # Remove trailing '-1' values
        ...     bonds_for_atom = bonds_for_atom[bonds_for_atom != -1]
        ...     print(f"{i}: {bonds_for_atom}")
        0: [1 5 6]
        1: [0 2 7]
        2: [1 3 8]
        3: [2 4 9]
        4: [ 3  5 10]
        5: [ 4  0 11]
        6: [0]
        7: [1]
        8: [2]
        9: [3]
        10: [4]
        11: [5]
        """
        cdef int i=0
        cdef uint32 atom_index_i, atom_index_j, bond_type

        cdef uint32[:,:] all_bonds_v = self._bonds
        # The size of 2nd dimension is equal to the atom with most bonds
        # Since each atom can have an individual number of bonded atoms,
        # The arrays are padded with '-1'
        cdef np.ndarray bonds = np.full(
            (self._atom_count, self._max_bonds_per_atom), -1, dtype=np.int32
        )
        cdef int32[:,:] bonds_v = bonds
        cdef np.ndarray bond_types = np.full(
            (self._atom_count, self._max_bonds_per_atom), -1, dtype=np.int8
        )
        cdef int8[:,:] bond_types_v = bond_types
        # Track the number of already found bonds for each given index
        cdef np.ndarray lengths = np.zeros(self._atom_count, dtype=np.uint32)
        cdef uint32[:] lengths_v = lengths

        for i in range(all_bonds_v.shape[0]):
            atom_index_i = all_bonds_v[i,0]
            atom_index_j = all_bonds_v[i,1]
            bond_type = all_bonds_v[i,2]
            # Add second bonded atom for the first bonded atom
            # and vice versa
            # Use 'lengths' variable to append the value
            bonds_v[atom_index_i, lengths_v[atom_index_i]] = atom_index_j
            bonds_v[atom_index_j, lengths_v[atom_index_j]] = atom_index_i
            bond_types_v[atom_index_i, lengths_v[atom_index_i]] = bond_type
            bond_types_v[atom_index_j, lengths_v[atom_index_j]] = bond_type
            # Increment lengths
            lengths_v[atom_index_i] += 1
            lengths_v[atom_index_j] += 1

        return bonds, bond_types


    def adjacency_matrix(self):
        r"""
        adjacency_matrix(bond_list)

        Represent this :class:`BondList` as adjacency matrix.

        The adjacency matrix is a quadratic matrix with boolean values
        according to

        .. math::

            M_{i,j} =
            \begin{cases}
                \text{True},  & \text{if } \text{Atom}_i \text{ and } \text{Atom}_j \text{ form a bond} \\
                \text{False}, & \text{otherwise}
            \end{cases}.

        Returns
        -------
        matrix : ndarray, dtype=bool, shape=(n,n)
            The created adjacency matrix.

        Examples
        --------

        >>> # BondList for formaldehyde
        >>> bond_list = BondList(
        ...     4,
        ...     np.array([
        ...         # Bond between carbon and oxygen
        ...         (0,  1, BondType.DOUBLE),
        ...         # Bonds between carbon and hydrogen
        ...         (0,  2, BondType.SINGLE),
        ...         (0,  3, BondType.SINGLE),
        ...     ])
        ... )
        >>> print(bond_list.adjacency_matrix())
        [[False  True  True  True]
         [ True False False False]
         [ True False False False]
         [ True False False False]]
        """
        matrix = np.zeros(
            (self._atom_count, self._atom_count), dtype=bool
        )
        matrix[self._bonds[:,0], self._bonds[:,1]] = True
        matrix[self._bonds[:,1], self._bonds[:,0]] = True
        return matrix


    def bond_type_matrix(self):
        r"""
        adjacency_matrix(bond_list)

        Represent this :class:`BondList` as a matrix depicting the bond
        type.

        The matrix is a quadratic matrix:

        .. math::

            M_{i,j} =
            \begin{cases}
                \text{BondType}_{ij},  & \text{if } \text{Atom}_i \text{ and } \text{Atom}_j \text{ form a bond} \\
                -1,                    & \text{otherwise}
            \end{cases}.

        Returns
        -------
        matrix : ndarray, dtype=bool, shape=(n,n)
            The created bond type matrix.

        Examples
        --------

        >>> # BondList for formaldehyde
        >>> bond_list = BondList(
        ...     4,
        ...     np.array([
        ...         # Bond between carbon and oxygen
        ...         (0,  1, BondType.DOUBLE),
        ...         # Bonds between carbon and hydrogen
        ...         (0,  2, BondType.SINGLE),
        ...         (0,  3, BondType.SINGLE),
        ...     ])
        ... )
        >>> print(bond_list.bond_type_matrix())
        [[-1  2  1  1]
         [ 2 -1 -1 -1]
         [ 1 -1 -1 -1]
         [ 1 -1 -1 -1]]
        """
        matrix = np.full(
            (self._atom_count, self._atom_count), -1, dtype=np.int8
        )
        matrix[self._bonds[:,0], self._bonds[:,1]] = self._bonds[:,2]
        matrix[self._bonds[:,1], self._bonds[:,0]] = self._bonds[:,2]
        return matrix


    def add_bond(self, int32 atom_index1, int32 atom_index2,
                 bond_type=BondType.ANY):
        """
        add_bond(atom_index1, atom_index2, bond_type=BondType.ANY)

        Add a bond to the :class:`BondList`.

        If the bond is already existent, only the bond type is updated.

        Parameters
        ----------
        atom_index1, atom_index2 : int
            The indices of the atoms to create a bond for.
        bond_type : BondType or int, optional
            The type of the bond. Default is :attr:`BondType.ANY`.
        """
        if bond_type >= len(BondType):
            raise ValueError(f"BondType {bond_type} is invalid")

        cdef uint32 index1 = _to_positive_index(atom_index1, self._atom_count)
        cdef uint32 index2 = _to_positive_index(atom_index2, self._atom_count)
        _sort(&index1, &index2)

        cdef int i
        cdef uint32[:,:] all_bonds_v = self._bonds
        # Check if bond is already existent in list
        cdef bint in_list = False
        for i in range(all_bonds_v.shape[0]):
            # Since the bonds have the atom indices sorted
            # the reverse check is omitted
            if (all_bonds_v[i,0] == index1 and all_bonds_v[i,1] == index2):
                in_list = True
                # If in list, update bond type
                all_bonds_v[i,2] = int(bond_type)
                break
        if not in_list:
            self._bonds = np.append(
                self._bonds,
                np.array(
                    [(index1, index2, int(bond_type))], dtype=np.uint32
                ),
                axis=0
            )
            self._max_bonds_per_atom = self._get_max_bonds_per_atom()

    def remove_bond(self, int32 atom_index1, int32 atom_index2):
        """
        remove_bond(atom_index1, atom_index2)

        Remove a bond from the :class:`BondList`.

        If the bond is not existent in the :class:`BondList`, nothing happens.

        Parameters
        ----------
        atom_index1, atom_index2 : int
            The indices of the atoms whose bond should be removed.
        """
        cdef uint32 index1 = _to_positive_index(atom_index1, self._atom_count)
        cdef uint32 index2 = _to_positive_index(atom_index2, self._atom_count)
        _sort(&index1, &index2)

        # Find the bond in bond list
        cdef int i
        cdef uint32[:,:] all_bonds_v = self._bonds
        for i in range(all_bonds_v.shape[0]):
            # Since the bonds have the atom indices sorted
            # the reverse check is omitted
            if (all_bonds_v[i,0] == index1 and all_bonds_v[i,1] == index2):
                self._bonds = np.delete(self._bonds, i, axis=0)
        # The maximum bonds per atom is not recalculated,
        # as the value can only be decreased on bond removal
        # Since this value is only used for pessimistic array allocation
        # in 'get_bonds()', the slightly larger memory usage is a better
        # option than the repetitive call of _get_max_bonds_per_atom()

    def remove_bonds_to(self, int32 atom_index):
        """
        remove_bonds_to(self, atom_index)

        Remove all bonds from the :class:`BondList` where the given atom
        is involved.

        Parameters
        ----------
        atom_index : int
            The index of the atom whose bonds should be removed.
        """
        cdef uint32 index = _to_positive_index(atom_index, self._atom_count)

        cdef np.ndarray mask = np.ones(len(self._bonds), dtype=np.uint8)
        cdef uint8[:] mask_v = mask

        # Find the bond in bond list
        cdef int i
        cdef uint32[:,:] all_bonds_v = self._bonds
        for i in range(all_bonds_v.shape[0]):
            if (all_bonds_v[i,0] == index or all_bonds_v[i,1] == index):
                mask_v[i] = False
        # Remove the bonds
        self._bonds = self._bonds[mask.astype(bool, copy=False)]
        # The maximum bonds per atom is not recalculated
        # (see 'remove_bond()')

    def remove_bonds(self, bond_list):
        """
        remove_bonds(bond_list)

        Remove multiple bonds from the :class:`BondList`.

        All bonds present in `bond_list` are removed from this instance.
        If a bond is not existent in this instance, nothing happens.
        Only the bond indices, not the bond types, are relevant for
        this.

        Parameters
        ----------
        bond_list : BondList
            The bonds in `bond_list` are removed from this instance.
        """
        cdef int i=0, j=0

        # All bonds in the own BondList
        cdef uint32[:,:] all_bonds_v = self._bonds
        # The bonds that should be removed
        cdef uint32[:,:] rem_bonds_v = bond_list._bonds
        cdef np.ndarray mask = np.ones(all_bonds_v.shape[0], dtype=np.uint8)
        cdef uint8[:] mask_v = mask
        for i in range(all_bonds_v.shape[0]):
            for j in range(rem_bonds_v.shape[0]):
                if      all_bonds_v[i,0] == rem_bonds_v[j,0] \
                    and all_bonds_v[i,1] == rem_bonds_v[j,1]:
                        mask_v[i] = False

        # Remove the bonds
        self._bonds = self._bonds[mask.astype(bool, copy=False)]
        # The maximum bonds per atom is not recalculated
        # (see 'remove_bond()')

    def merge(self, bond_list):
        """
        merge(bond_list)

        Merge another :class:`BondList` with this instance into a new
        object.
        If a bond appears in both :class:`BondList`'s, the
        :class:`BondType` from the given `bond_list` takes precedence.

        The internal :class:`ndarray` instances containg the bonds are
        simply concatenated and the new atom count is the maximum of
        both bond lists.

        Parameters
        ----------
        bond_list : BondList
            This bond list is merged with this instance.

        Returns
        -------
        bond_list : BondList
            The merged :class:`BondList`.

        Notes
        -----
        This is not equal to using the `+` operator.

        Examples
        --------

        >>> bond_list1 = BondList(3, np.array([(0,1),(1,2)]))
        >>> bond_list2 = BondList(5, np.array([(2,3),(3,4)]))
        >>> merged_list = bond_list2.merge(bond_list1)
        >>> print(merged_list.get_atom_count())
        5
        >>> print(merged_list)
        [[0 1 0]
         [1 2 0]
         [2 3 0]
         [3 4 0]]

        The BondList given as parameter takes precedence:

        >>> # Specifiy bond type to see where a bond is taken from
        >>> bond_list1 = BondList(4, np.array([
        ...     (0, 1, BondType.SINGLE),
        ...     (1, 2, BondType.SINGLE)
        ... ]))
        >>> bond_list2 = BondList(4, np.array([
        ...     (1, 2, BondType.DOUBLE),    # This one is a duplicate
        ...     (2, 3, BondType.DOUBLE)
        ... ]))
        >>> merged_list = bond_list2.merge(bond_list1)
        >>> print(merged_list)
        [[0 1 1]
         [1 2 1]
         [2 3 2]]
        """
        return BondList(
            max(self._atom_count, bond_list._atom_count),
            np.concatenate(
                [bond_list.as_array(), self.as_array()],
                axis=0
            )
        )

    def __add__(self, bond_list):
        return BondList.concatenate([self, bond_list])

    def __getitem__(self, index):
        ## Variables for both, integer and boolean index arrays
        cdef uint32[:,:] all_bonds_v
        cdef int i
        cdef uint32* index1_ptr
        cdef uint32* index2_ptr
        cdef np.ndarray removal_filter
        cdef uint8[:] removal_filter_v

        ## Variables for integer arrays
        cdef int32[:] inverse_index_v
        cdef int32 new_index1, new_index2

        ## Variables for boolean mask
        # Boolean mask representation of the index
        cdef np.ndarray mask
        cdef uint8[:] mask_v
        # Boolean mask for removal of bonds
        cdef np.ndarray offsets
        cdef uint32[:] offsets_v

        if isinstance(index, numbers.Integral):
            ## Handle single index
            return self.get_bonds(index)

        elif isinstance(index, np.ndarray) and index.dtype == bool:
            ## Handle boolean masks
            copy = self.copy()
            all_bonds_v = copy._bonds
            # Use 'uint8' instead of 'bool' for memory view
            mask = np.frombuffer(index, dtype=np.uint8)

            # Each time an atom is missing in the mask,
            # the offset is increased by one
            offsets = np.cumsum(
                ~mask.astype(bool, copy=False), dtype=np.uint32
            )
            removal_filter = np.ones(all_bonds_v.shape[0], dtype=np.uint8)
            removal_filter_v = removal_filter
            mask_v = mask
            offsets_v = offsets
            # If an atom in a bond is not masked,
            # the bond is removed from the list
            # If an atom is masked,
            # its index value is decreased by the respective offset
            # The offset is neccessary, removing atoms in an AtomArray
            # decreases the index of the following atoms
            for i in range(all_bonds_v.shape[0]):
                # Usage of pointer to increase performance
                # as redundant indexing is avoided
                index1_ptr = &all_bonds_v[i,0]
                index2_ptr = &all_bonds_v[i,1]
                if mask_v[index1_ptr[0]] and mask_v[index2_ptr[0]]:
                    # Both atoms involved in bond are masked
                    # -> decrease atom index by offset
                    index1_ptr[0] -= offsets_v[index1_ptr[0]]
                    index2_ptr[0] -= offsets_v[index2_ptr[0]]
                else:
                    # At least one atom involved in bond is not masked
                    # -> remove bond
                    removal_filter_v[i] = False
            # Apply the bond removal filter
            copy._bonds = copy._bonds[removal_filter.astype(bool, copy=False)]
            copy._atom_count = len(np.nonzero(mask)[0])
            copy._max_bonds_per_atom = copy._get_max_bonds_per_atom()
            return copy

        else:
            ## Convert any other type of index into index array, as it preserves order
            copy = self.copy()
            all_bonds_v = copy._bonds
            index = _to_index_array(index, self._atom_count)
            index = _to_positive_index_array(index, self._atom_count)

            # The inverse index is required to efficiently obtain
            # the new index of an atom in case of an unsorted index
            # array
            inverse_index_v = _invert_index(index, self._atom_count)
            removal_filter = np.ones(all_bonds_v.shape[0], dtype=np.uint8)
            removal_filter_v = removal_filter
            for i in range(all_bonds_v.shape[0]):
                # Usage of pointer to increase performance
                # as redundant indexing is avoided
                index1_ptr = &all_bonds_v[i,0]
                index2_ptr = &all_bonds_v[i,1]
                new_index1 = inverse_index_v[index1_ptr[0]]
                new_index2 = inverse_index_v[index2_ptr[0]]
                if new_index1 != -1 and new_index2 != -1:
                    # Both atoms involved in bond are included
                    # by index array
                    # -> assign new atom indices
                    index1_ptr[0] = <int32>new_index1
                    index2_ptr[0] = <int32>new_index2
                else:
                    # At least one atom in bond is not included
                    # -> remove bond
                    removal_filter_v[i] = False

            copy._bonds = copy._bonds[
                removal_filter.astype(bool, copy=False)
            ]
            # Again, sort indices per bond
            # as the correct order is not guaranteed anymore
            # for unsorted index arrays
            copy._bonds[:,:2] = np.sort(copy._bonds[:,:2], axis=1)
            copy._atom_count = len(index)
            copy._max_bonds_per_atom = copy._get_max_bonds_per_atom()
            return copy

    def __iter__(self):
        raise TypeError("'BondList' object is not iterable")

    def __str__(self):
        return str(self.as_array())

    def __eq__(self, item):
        if not isinstance(item, BondList):
            return False
        return (self._atom_count == item._atom_count and
                self.as_set() == item.as_set())

    def __contains__(self, item):
        if not isinstance(item, tuple) and len(tuple) != 2:
            raise TypeError("Expected a tuple of atom indices")

        cdef int i=0

        cdef uint32 match_index1, match_index2
        # Sort indices for faster search in loop
        cdef uint32 atom_index1 = min(item)
        cdef uint32 atom_index2 = max(item)

        cdef uint32[:,:] all_bonds_v = self._bonds
        for i in range(all_bonds_v.shape[0]):
            match_index1 = all_bonds_v[i,0]
            match_index2 = all_bonds_v[i,1]
            if atom_index1 == match_index1 and atom_index2 == match_index2:
                return True

        return False


    def _get_max_bonds_per_atom(self):
        if self._atom_count == 0:
            return 0

        cdef int i
        cdef uint32[:,:] all_bonds_v = self._bonds
        # Create an array that counts number of occurences of each index
        cdef np.ndarray index_count = np.zeros(self._atom_count,
                                               dtype=np.uint32)
        cdef uint32[:] index_count_v = index_count
        for i in range(all_bonds_v.shape[0]):
            # Increment count of both indices found in bond list at i
            index_count_v[all_bonds_v[i,0]] += 1
            index_count_v[all_bonds_v[i,1]] += 1
        return np.max(index_count_v)

    def _remove_redundant_bonds(self):
        cdef int j
        cdef uint32[:,:] all_bonds_v = self._bonds
        # Boolean mask for final removal of redundant atoms
        # Unfortunately views of boolean ndarrays are not supported
        # -> use uint8 array
        cdef np.ndarray redundancy_filter = np.ones(all_bonds_v.shape[0],
                                                    dtype=np.uint8)
        cdef uint8[:] redundancy_filter_v = redundancy_filter
        # Array of pointers to C-arrays
        # The array is indexed with the atom indices in the bond list
        # The respective C-array contains the indices of bonded atoms
        cdef ptr[:] ptrs_v = np.zeros(self._atom_count, dtype=np.uint64)
        # Stores the length of the C-arrays
        cdef int[:] array_len_v = np.zeros(self._atom_count, dtype=np.int32)
        # Iterate over bond list:
        # If bond is already listed in the array of pointers,
        # set filter to false at that position
        # Else add bond to array of pointers
        cdef uint32 i1, i2
        cdef uint32* array_ptr
        cdef int length

        try:
            for j in range(all_bonds_v.shape[0]):
                i1 = all_bonds_v[j,0]
                i2 = all_bonds_v[j,1]
                # Since the bonds have the atom indices sorted
                # the reverse check is omitted
                if _in_array(<uint32*>ptrs_v[i1], i2, array_len_v[i1]):
                        redundancy_filter_v[j] = False
                else:
                    # Append bond in respective C-array
                    # and update C-array length
                    length = array_len_v[i1] +1
                    array_ptr = <uint32*>ptrs_v[i1]
                    array_ptr = <uint32*>realloc(
                        array_ptr, length * sizeof(uint32)
                    )
                    if not array_ptr:
                        raise MemoryError()
                    array_ptr[length-1] = i2
                    ptrs_v[i1] = <ptr>array_ptr
                    array_len_v[i1] = length

        finally:
            # Free pointers
            for i in range(ptrs_v.shape[0]):
                free(<int*>ptrs_v[i])

        # Eventually remove redundant bonds
        self._bonds = self._bonds[redundancy_filter.astype(bool, copy=False)]


cdef uint32 _to_positive_index(int32 index, uint32 array_length) except -1:
    """
    Convert a potentially negative index into a positive index.
    """
    cdef uint32 pos_index
    if index < 0:
        pos_index = <uint32> (array_length + index)
        if pos_index < 0:
            raise IndexError(
                f"Index {index} is out of range "
                f"for an atom count of {array_length}"
            )
        return pos_index
    else:
        if <uint32> index >= array_length:
            raise IndexError(
                f"Index {index} is out of range "
                f"for an atom count of {array_length}"
            )
        return <uint32> index


def _to_positive_index_array(index_array, length):
    """
    Convert potentially negative values in an array into positive
    values and check for out-of-bounds values.
    """
    index_array = index_array.copy()
    orig_shape = index_array.shape
    index_array = index_array.flatten()
    negatives = index_array < 0
    index_array[negatives] = length + index_array[negatives]
    if (index_array < 0).any():
        raise IndexError(
            f"Index {np.min(index_array)} is out of range "
            f"for an atom count of {length}"
        )
    if (index_array >= length).any():
        raise IndexError(
            f"Index {np.max(index_array)} is out of range "
            f"for an atom count of {length}"
        )
    return index_array.reshape(orig_shape)


def _to_index_array(object index, uint32 length):
    """
    Convert an index of arbitrary type into an index array.
    """
    if isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.integer):
        return index
    else:
        # Convert into index array
        all_indices = np.arange(length, dtype=np.uint32)
        return all_indices[index]


cdef inline bint _in_array(uint32* array, uint32 atom_index, int array_length):
    """
    Test whether a value (`atom_index`) is in a C-array `array`.
    """
    cdef int i = 0
    if array == NULL:
        return False
    for i in range(array_length):
        if array[i] == atom_index:
            return True
    return False


cdef inline void _sort(uint32* index1_ptr, uint32* index2_ptr):
    cdef uint32 swap
    if index1_ptr[0] > index2_ptr[0]:
        # Swap indices
        swap = index1_ptr[0]
        index1_ptr[0] = index2_ptr[0]
        index2_ptr[0] = swap


@cython.wraparound(False)
# Do bounds check, as the input indices may be out of bounds
def _invert_index(IndexType[:] index_v, uint32 length):
    """
    Invert an input index array, so that
    if *input[i] = j*, *output[j] = i*.
    For all elements *j*, that are not in *input*, *output[j]* = -1.
    """
    cdef int32 i
    cdef IndexType index_val
    inverse_index = np.full(length, -1, dtype=np.int32)
    cdef int32[:] inverse_index_v = inverse_index

    for i in range(index_v.shape[0]):
        index_val = index_v[i]
        if inverse_index_v[index_val] != -1:
            # One index can theoretically appear multiple times
            # This is currently not supported
            raise NotImplementedError(
                f"Duplicate indices are not supported, "
                f"but index {index_val} appeared multiple times"
            )
        inverse_index_v[index_val] = i


    return inverse_index




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

def connect_via_distances(atoms, dict distance_range=None, bint inter_residue=True,
                          default_bond_type=BondType.ANY, bint periodic=False):
    """
    connect_via_distances(atoms, distance_range=None, atom_mask=None,
                          inter_residue=True, default_bond_type=BondType.ANY,
                          periodic=False)

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
    from .atoms import AtomArray
    from .geometry import distance
    from .residues import get_residue_starts

    cdef list bonds = []
    cdef int i
    cdef int curr_start_i, next_start_i
    cdef np.ndarray coord = atoms.coord
    cdef np.ndarray coord_in_res
    cdef np.ndarray distances
    cdef float dist
    cdef np.ndarray elements = atoms.element
    cdef np.ndarray elements_in_res
    cdef int atom_index1, atom_index2
    cdef dict dist_ranges = {}
    cdef tuple dist_range
    cdef float min_dist, max_dist

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
    for key, val in itertools.chain(
        _DEFAULT_DISTANCE_RANGE.items(), distance_range.items()
    ):
        element1, element2 = key
        # Add entries for both element orders
        dist_ranges[(element1.upper(), element2.upper())] = val
        dist_ranges[(element2.upper(), element1.upper())] = val

    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    # Omit exclsive stop in 'residue_starts'
    for i in range(len(residue_starts)-1):
        curr_start_i = residue_starts[i]
        next_start_i = residue_starts[i+1]

        elements_in_res = elements[curr_start_i : next_start_i]
        coord_in_res = coord[curr_start_i : next_start_i]
        # Matrix containing all pairwise atom distances in the residue
        distances = distance(
            coord_in_res[:, np.newaxis, :],
            coord_in_res[np.newaxis, :, :],
            box
        )
        for atom_index1 in range(len(elements_in_res)):
            for atom_index2 in range(atom_index1):
                dist_range = dist_ranges.get((
                    elements_in_res[atom_index1],
                    elements_in_res[atom_index2]
                ))
                if dist_range is None:
                    # No bond distance entry for this element
                    # combination -> skip
                    continue
                else:
                    min_dist, max_dist = dist_range
                dist = distances[atom_index1, atom_index2]
                if dist >= min_dist and dist <= max_dist:
                    bonds.append((
                        curr_start_i + atom_index1,
                        curr_start_i + atom_index2,
                        default_bond_type
                    ))

    bond_list = BondList(atoms.array_length(), np.array(bonds))

    if inter_residue:
        inter_bonds = _connect_inter_residue(atoms, residue_starts)
        if default_bond_type == BondType.ANY:
            # As all bonds should be of type ANY, convert also
            # inter-residue bonds to ANY
            inter_bonds.remove_bond_order()
        return bond_list.merge(inter_bonds)
    else:
        return bond_list



def connect_via_residue_names(atoms, bint inter_residue=True,
                              dict custom_bond_dict=None):
    """
    connect_via_residue_names(atoms, atom_mask=None, inter_residue=True)

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
    from .info.bonds import bonds_in_residue
    from .residues import get_residue_starts

    cdef list bonds = []
    cdef int res_i
    cdef int i, j
    cdef int curr_start_i, next_start_i
    cdef np.ndarray atom_names = atoms.atom_name
    cdef np.ndarray atom_names_in_res
    cdef np.ndarray res_names = atoms.res_name
    cdef str atom_name1, atom_name2
    cdef int64[:] atom_indices1, atom_indices2
    cdef dict bond_dict_for_res

    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    # Omit exclsive stop in 'residue_starts'
    for res_i in range(len(residue_starts)-1):
        curr_start_i = residue_starts[res_i]
        next_start_i = residue_starts[res_i+1]

        if custom_bond_dict is None:
            bond_dict_for_res = bonds_in_residue(res_names[curr_start_i])
        else:
            bond_dict_for_res = custom_bond_dict.get(
                res_names[curr_start_i], {}
            )

        atom_names_in_res = atom_names[curr_start_i : next_start_i]
        for (atom_name1, atom_name2), bond_type in bond_dict_for_res.items():
            atom_indices1 = np.where(atom_names_in_res == atom_name1)[0] \
                            .astype(np.int64, copy=False)
            atom_indices2 = np.where(atom_names_in_res == atom_name2)[0] \
                            .astype(np.int64, copy=False)
            # In rare cases the same atom name may appear multiple times
            # (e.g. in altlocs)
            # -> create all possible bond combinations
            for i in range(atom_indices1.shape[0]):
                for j in range(atom_indices2.shape[0]):
                    bonds.append((
                        curr_start_i + atom_indices1[i],
                        curr_start_i + atom_indices2[j],
                        bond_type
                    ))

    bond_list = BondList(atoms.array_length(), np.array(bonds))

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
    from .info.misc import link_type

    cdef list bonds = []
    cdef int i
    cdef np.ndarray atom_names = atoms.atom_name
    cdef np.ndarray res_names = atoms.res_name
    cdef np.ndarray res_ids = atoms.res_id
    cdef np.ndarray chain_ids = atoms.chain_id
    cdef int curr_start_i, next_start_i, after_next_start_i
    cdef str curr_connect_atom_name, next_connect_atom_name
    cdef np.ndarray curr_connect_indices, next_connect_indices

    # Iterate over all starts excluding:
    #   - the last residue and
    #   - exclusive end index of 'atoms'
    for i in range(len(residue_starts)-2):
        curr_start_i = residue_starts[i]
        next_start_i = residue_starts[i+1]
        after_next_start_i = residue_starts[i+2]

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
        curr_connect_indices = curr_start_i + np.where(
            atom_names[curr_start_i : next_start_i]
            == curr_connect_atom_name
        )[0]
        # Index in atom array for atom name in next residue
        next_connect_indices = next_start_i + np.where(
            atom_names[next_start_i : after_next_start_i]
            == next_connect_atom_name
        )[0]
        if len(curr_connect_indices) == 0 or len(next_connect_indices) == 0:
            # The connector atoms are not found in the adjacent residues
            # -> skip this bond
            continue

        bonds.append((
            curr_connect_indices[0],
            next_connect_indices[0],
            BondType.SINGLE
        ))

    return BondList(atoms.array_length(), np.array(bonds, dtype=np.uint32))



def find_connected(bond_list, uint32 root, bint as_mask=False):
    """
    find_connected(bond_list, root, as_mask=False)

    Get indices to all atoms that are directly or inderectly connected
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

    cdef uint8[:] is_connected_mask = np.zeros(
        bond_list.get_atom_count(), dtype=np.uint8
    )
    # Find connections in a recursive way,
    # by visiting all atoms that are reachable by a bond
    _find_connected(bond_list, root, is_connected_mask, all_bonds)
    if as_mask:
        return is_connected_mask
    else:
        return np.where(np.asarray(is_connected_mask))[0]


cdef _find_connected(bond_list,
                     int32 index,
                     uint8[:] is_connected_mask,
                     int32[:,:] all_bonds):
    if is_connected_mask[index]:
        # This atom has already been visited
        # -> exit condition
        return
    is_connected_mask[index] = True

    cdef int32 j
    cdef int32 connected_index
    for j in range(all_bonds.shape[1]):
        connected_index = all_bonds[index, j]
        if connected_index == -1:
            # Ignore padding values
            continue
        _find_connected(
            bond_list, connected_index, is_connected_mask, all_bonds
        )


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
    cdef uint32 i, j
    cdef uint32 bond_type
    cdef uint32 SINGLE = int(BondType.SINGLE)
    cdef bint in_same_cycle

    bond_graph = bonds.as_graph()
    cycles = nx.algorithms.cycles.cycle_basis(bond_graph)

    cdef int64[:] number_of_partners_v = np.count_nonzero(
        bonds.get_all_bonds()[0] != -1,
        axis=1
    ).astype(np.int64, copy=False)

    rotatable_bonds = []
    cdef uint32[:,:] bonds_v = bonds.as_array()
    for i, j, bond_type in bonds_v:
        # Can only rotate about single bonds
        # Furthermore, it makes no sense to rotate about a bond,
        # that leads to a single atom
        if bond_type == BondType.SINGLE \
            and number_of_partners_v[i] > 1 \
            and number_of_partners_v[j] > 1:
                # Cannot rotate about a bond, if the two connected atoms
                # are in a cycle
                in_same_cycle = False
                for cycle in cycles:
                    if i in cycle and j in cycle:
                        in_same_cycle = True
                if not in_same_cycle:
                    rotatable_bonds.append((i,j, bond_type))
    return BondList(bonds.get_atom_count(), np.array(rotatable_bonds))
