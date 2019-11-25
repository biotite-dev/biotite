# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module allows efficient search of atoms in a defined radius around
a location.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["BondList", "BondType"]

cimport cython
cimport numpy as np
from libc.stdlib cimport realloc, malloc, free

import numbers
from enum import IntEnum
import numpy as np
from ..copyable import Copyable

ctypedef np.uint64_t ptr
ctypedef np.uint32_t uint32
ctypedef np.uint8_t uint8
ctypedef np.int64_t int64


class BondType(IntEnum):
    """
    This enum type represents the type of a chemical bond. 
    """
    ANY = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    QUADRUPLE = 4
    AROMATIC = 5


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
    :class:`AtomArray` before and after indexing.
    If a :class:`BondList` is indexed with single integer as index,
    :func:`get_bonds()` will be called with the index as parameter.

    The same consistency applies to adding :class:`BondList` instances
    via the '+' operator:
    The atom indices of the second :class:`BondList` are increased by
    the atom count of the first :class:`BondList`.
    
    Parameters
    ----------
    atom_count : int
        A positive integer, that specifies the amount of atoms the
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
    """

    def __init__(self, uint32 atom_count, np.ndarray bonds=None):
        self._atom_count = atom_count
        if bonds is not None:
            if (bonds[:,:2] >= atom_count).any():
                raise ValueError(
                    f"Index {np.max(bonds[:,:2])} in bonds is too large "
                    f"for atom count of {atom_count}"
                )
            if bonds.shape[1] == 3:
                # input contains bonds (index 0 and 1)
                # including the bond type value (index 3)
                # -> Simply copy input
                self._bonds = bonds.astype(np.uint32)
                # Indices are sorted per bond
                # so that the lower index is at the first position
                self._bonds[:,:2] = np.sort(self._bonds[:,:2], axis=1)
            elif bonds.shape[1] == 2:
                # input contains the bonds without bond type
                # -> Default: Set bond type ANY (0)
                self._bonds = np.zeros((bonds.shape[0], 3), dtype=np.uint32)
                # Set and sort atom indices per bond
                self._bonds[:,:2] = np.sort(bonds[:,:2], axis=1)
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
        self._bonds[:,0] += offset
        self._bonds[:,1] += offset
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
    
    def get_bonds(self, uint32 atom_index):
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
        bonds : np.ndarray, dtype=np.uint32
            The indices of connected atoms.
        bond_types : np.ndarray, dtype=np.uint8
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
            # then append the the index of the respective other position
            if all_bonds_v[i,0] == atom_index:
                bonds_v[j] = all_bonds_v[i,1]
                bond_types_v[j] = all_bonds_v[i,2]
                j += 1
            elif all_bonds_v[i,1] == atom_index:
                bonds_v[j] = all_bonds_v[i,0]
                bond_types_v[j] = all_bonds_v[i,2]
                j += 1
        # Trim to correct size
        bonds = bonds[:j]
        bond_types = bond_types[:j]
        return bonds, bond_types
    
    def add_bond(self, uint32 index1, uint32 index2, bond_type=BondType.ANY):
        """
        add_bond(index1, index2, bond_type=BondType.ANY)
        
        Add a bond to the :class:`BondList`.

        If the bond is already existent, only the bond type is updated.

        Parameters
        ----------
        index1, index2 : int
            The indices of the atoms to create a bond for.
        bond_type : BondType or int, optional
            The type of the bond. Default is :attr:`BondType.ANY`.
        """
        if index1 >= self._atom_count or index2 >= self._atom_count:
            raise ValueError(
                f"Index {max(index1, index2)} in new bond is too large "
                f"for atom count of {self._atom_count}"
            )
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
                np.array([(index1, index2, int(bond_type))], dtype=np.uint32),
                axis=0
            )
            self._max_bonds_per_atom = self._get_max_bonds_per_atom()

    def remove_bond(self, uint32 index1, uint32 index2):
        """
        remove_bond(index1, index2)
        
        Remove a bond from the :class:`BondList`.

        If the bond is not existent in the :class:`BondList`, nothing happens.

        Parameters
        ----------
        index1, index2 : int
            The indices of the atoms whose bond should be removed.
        """
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
        # since the value can only be decreased on bond removal
        # Since this value is only used for pessimistic array allocation
        # in 'get_bonds()', the slightly larger memory usage is a better
        # option than the repetitive call of _get_max_bonds_per_atom()

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
        self._bonds = self._bonds[mask.astype(np.bool, copy=False)]
        # The maximum bonds per atom is not recalculated
        # (see 'remove_bond())
        #for bond in bond_list._bonds:
        #    self.remove_bond(bond[0], bond[1])

    def merge(self, bond_list):
        """
        merge(bond_list)
        
        Merge the this instance with another :class:`BondList` in a new
        object.

        The internal :class:`ndarray` instances containg the bonds are
        simply concatenated and the new atom count is the maximum atom
        count of the merged bond lists.

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

        This is not equal to using '+' operator.
        
        Examples
        --------

        >>> bond_list1 = BondList(3, np.array([(0,1),(1,2)]))
        >>> bond_list2 = BondList(5, np.array([(2,3),(3,4)]))
        >>> merged_list = bond_list1.merge(bond_list2)
        >>> print(merged_list.get_atom_count())
        5
        >>> print(merged_list)
        [[0 1 0]
         [1 2 0]
         [2 3 0]
         [3 4 0]]
        """
        return BondList( 
            max(self._atom_count, bond_list._atom_count), 
            np.concatenate([self.as_array(), 
                            bond_list.as_array()], 
                            axis=0) 
        ) 

    def __add__(self, bond_list):
        cdef np.ndarray merged_bonds \
            = np.concatenate([self._bonds, bond_list._bonds])
        # Offset the indices of appended bonds list
        # (consistent with addition of AtomArray)
        merged_bonds[len(self._bonds):, :2] += self._atom_count
        cdef uint32 merged_count = self._atom_count + bond_list._atom_count
        cdef merged_bond_list = BondList(merged_count)
        # Array is not used in constructor to prevent unnecessary
        # maximum and redundant bond calculation
        merged_bond_list._bonds = merged_bonds
        merged_bond_list._max_bonds_per_atom = max(
            self._max_bonds_per_atom, merged_bond_list._max_bonds_per_atom
        )
        return merged_bond_list

    def __getitem__(self, index):
        cdef copy = self.copy()
        cdef uint32[:,:] all_bonds_v = copy._bonds
        # Boolean mask representation of the index
        cdef np.ndarray mask
        cdef uint8[:] mask_v
        # Boolean mask for removal of bonds
        cdef np.ndarray removal_filter
        cdef uint8[:] removal_filter_v
        cdef np.ndarray offsets
        cdef uint32[:] offsets_v
        cdef int i
        cdef uint32* index1_ptr
        cdef uint32* index2_ptr
        if isinstance(index, numbers.Integral):
            return copy.get_bonds(index)
        else:
            mask = _to_bool_mask(index, length=copy._atom_count)
            offsets = np.cumsum(~mask.astype(bool, copy=False),
                                dtype=np.uint32)
            removal_filter = np.ones(all_bonds_v.shape[0], dtype=np.uint8)
            mask_v = mask
            offsets_v = offsets
            removal_filter_v = removal_filter
            # If an atom in a bond is not masked,
            # the bond is removed from the list
            # If an atom is masked,
            # its index value is decreased by the respective offset
            # The offset is neccessary, removing atoms in an AtomArray
            # decreases the index of the following atoms
            for i in range(all_bonds_v.shape[0]):
                index1_ptr = &all_bonds_v[i,0]
                index2_ptr = &all_bonds_v[i,1]
                if mask_v[index1_ptr[0]] and mask_v[index2_ptr[0]]:
                    index1_ptr[0] -= offsets_v[index1_ptr[0]]
                    index2_ptr[0] -= offsets_v[index2_ptr[0]]
                else:
                    removal_filter_v[i] = False
            # Remove bonds, where at least one atom was removed
            # by the input index
            copy._bonds = copy._bonds[removal_filter.astype(bool, copy=False)]
            copy._atom_count = len(np.nonzero(mask)[0])
            copy._max_bonds_per_atom = copy._get_max_bonds_per_atom()
            return copy
    
    def __str__(self):
        return str(self.as_array())
    
    def __eq__(self, item):
        if not isinstance(item, BondList):
            return False
        return (self._atom_count == item._atom_count and
                np.array_equal(self._bonds, item._bonds))

    def _get_max_bonds_per_atom(self):
        cdef int i
        cdef uint32[:,:] all_bonds_v = self._bonds
        # Create array that counts number of occurences of each index
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
                if     _in_array(<uint32*>ptrs_v[i1], i2, array_len_v[i1]):
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
        self._bonds = self._bonds[redundancy_filter.astype(np.bool,copy=False)]


cdef inline bint _in_array(uint32* array, uint32 atom_index, int array_length):
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

def _to_bool_mask(object index, int length):
    """
    Convert an index of arbitrary type into a boolean mask
    with given length
    """
    cdef int i=0, j=0
    cdef int64[:] index_array
    cdef uint8[:] bool_mask
    if isinstance(index, np.ndarray):
        if index.dtype == np.bool:
            # Index is already boolean mask -> simply return as uint8
            return index.astype(np.uint8, copy=False)
        elif np.issubdtype(index.dtype, np.integer):
            # Index is an index array
            # -> construct a boolean mask from it
            index_array = index.astype(np.int64, copy=False)
            bool_mask = np.zeros(length, dtype=np.uint8)
            # Flip mask to true for every index in index array
            for i in range(index_array.shape[0]):
                j = index_array[i]
                bool_mask[j] = True
            return np.asarray(bool_mask)
        else:
            raise TypeError(
                f"Arrays of type '{str(index.dtype)}' are not supported")

    else:
        # Any other index type -> construct an intermediate index array
        array = np.arange(length, dtype=np.int64)
        array = array[index]
        if not isinstance(array, np.ndarray):
            raise TypeError("A single integer is not a valid index "
                            "for this method")
        return _to_bool_mask(array, length)