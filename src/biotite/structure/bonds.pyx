# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

"""
This module allows efficient search of atoms in a defined radius around
a location.
"""

cimport cython
cimport numpy as np
from libc.stdlib cimport realloc, malloc, free

import numpy as np
from enum import IntEnum
from ..copyable import Copyable

ctypedef np.uint64_t ptr
ctypedef np.uint32_t uint32
ctypedef np.uint8_t uint8

__all__ = ["BondList"]


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


class BondList(Copyable):
    """
    Input array may contain redundant bonds, which are automatically
    sanitized.
    """

    def __init__(self, np.ndarray bonds=None):
        if bonds is not None:
            if bonds.shape[1] == 3:
                # input contains bonds (index 0 and 1)
                # including the bond type value (index 3)
                # -> Simply copy input
                self._bonds = bonds.astype(np.uint32)
            elif bonds.shape[1] == 3:
                # input contains the bonds without bond type
                # -> Default: Set bond type ANY (0)
                self._bonds = np.zeros((bonds.shape[0], 3), dtype=np.uint32)
                self._bonds[:,0] = bonds[:,0]
                self._bonds[:,1] = bonds[:,1]
            else:
                raise ValueError("Input array containing bonds must have a "
                                 "length of either 2 or 3 in the second "
                                 "dimension")
            self._remove_redundant_bonds()
            self._max_bonds_per_atom = self._get_max_bonds_per_atom()
        else:
            # Create empty bond list
            self._bonds = np.zeros((0, 3), dtype=np.uint32)

    def __copy_create__(self):
        # Create empty bond list to prevent
        # unnecessary removal of redundant atoms
        # and calculation of maximum bonds per atom
        return BondList()
    
    def __copy_fill__(self, clone):
        clone._bonds = self._bonds.copy()
        clone._max_bonds_per_atom = self._max_bonds_per_atom
    
    def offset_indices(self, int offset):
        self._bonds[:,0] += offset
        self._bonds[:,1] += offset
    
    def as_array(self):
        return self._bonds.copy()
    
    def get_bonds(self, int atom_index):
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
    
    def add_bond(self, int index1, int index2, bond_type=BondType.ANY):
        cdef int i
        cdef uint32[:,:] all_bonds_v = self._bonds
        # Check if bond is already existent in list
        cdef bint in_list = False
        for i in range(all_bonds_v.shape[0]):
            if (all_bonds_v[i,0] == index1 and all_bonds_v[i,1] == index2) \
                or (all_bonds_v[i,1] == index1 and all_bonds_v[i,0] == index2):
                    in_list = True
                    # If in list, update bond type
                    all_bonds_v[i,2] = int(bond_type)
                    break
        if not in_list:
            self._bonds = np.append(self._bonds,
                                    [(index1, index2, int(bond_type))],
                                    axis=0)
            self._max_bonds_per_atom = self._get_max_bonds_per_atom()

    def remove_bond(self, int index1, int index2):
        # Find the bond in bond list
        cdef int i
        cdef uint32[:,:] all_bonds_v = self._bonds
        for i in range(all_bonds_v.shape[0]):
            if (all_bonds_v[i,0] == index1 and all_bonds_v[i,1] == index2) \
                or (all_bonds_v[i,1] == index1 and all_bonds_v[i,0] == index2):
                    self._bonds = np.delete(self._bonds, i)
        # The maximum bonds per atom is not recalculated,
        # since the value can only be decreased on bond removal
        # Since this value is only used for pessimistic array allocation
        # in 'get_bonds()', the slightly larger memory usage is a better
        # option than the repetitive call of _get_max_bonds_per_atom()

    def __add__(self, bond_list):
        pass

    def __getitem__(self, index):
        pass

    def _get_max_bonds_per_atom(self):
        cdef int i
        cdef uint32[:,:] all_bonds_v = self._bonds
        # Find maximum atom index
        cdef int max_index = np.max(self._bonds[:,:2])
        # Create array that counts number of occurences of each index
        cdef np.ndarray index_count = np.zeros(max_index, dtype=np.uint32)
        cdef uint32[:] index_count_v = index_count
        for i in range(all_bonds_v.shape[0]):
            # Increment count of both indices found in bond list at i
            index_count_v[all_bonds_v[i,0]] += 1
            index_count_v[all_bonds_v[i,1]] += 1
        return np.max(index_count_v)
    
    def _remove_redundant_bonds(self):
        cdef int j
        cdef uint32[:,:] all_bonds_v = self._bonds
        cdef int max_index = np.max(self._bonds[:,:2])
        # Boolean mask for final removal of redundant atoms
        # Unfortunately views of boolean ndarrays are not supported
        # -> use uint8 array
        cdef np.ndarray redundancy_filter = np.ones(max_index, dtype=np.uint8)
        cdef uint8[:] redundancy_filter_v = redundancy_filter
        # Array of pointers to C-arrays
        # The array is indexed with the atom indices in the bond list
        # The respective C-array contains the indices of bonded atoms
        cdef ptr[:] ptrs_v = np.zeros(max_index, dtype=np.uint64)
        # Stores the length of the C-arrays
        cdef int[:] array_len_v = np.zeros(max_index, dtype=np.int32)
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
                if     in_array(<uint32*>ptrs_v[i1], i2, array_len_v[i1]) \
                    or in_array(<uint32*>ptrs_v[i2], i1, array_len_v[i2]) \
                        redundancy_filter_v[j] = False
                else:
                    # Store bond for the first bond partner...
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
                    # ...and the second bond partner
                    length = array_len_v[i2] +1
                    array_ptr = <uint32*>ptrs_v[i2]
                    array_ptr = <uint32*>realloc(
                        array_ptr, length * sizeof(uint32)
                    )
                    if not array_ptr:
                        raise MemoryError()
                    array_ptr[length-1] = i1
                    ptrs_v[i2] = <ptr>array_ptr
                    array_len_v[i2] = length
        finally:
            # Free pointers
            for i in range(ptrs.shape[0]):
                free(<int*>ptrs[i])
        # Eventually remove redundant bonds
        self._bonds = self._bonds[redundancy_filter]


cdef inline bint in_array(uint32* array, uint32 atom_index, int array_length):
    cdef int i = 0
    if array == NULL:
        return False
    for i in range(array_length):
        if array[i] = atom_index:
            return True
    return False