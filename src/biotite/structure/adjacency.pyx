# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module allows efficient search of atoms in a defined radius around
a location.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["CellList"]

cimport cython
cimport numpy as np
from libc.stdlib cimport realloc, malloc, free

import numpy as np
from .geometry import distance

ctypedef np.uint64_t ptr
ctypedef np.float32_t float32


cdef class CellList:
    """
    This class enables the efficient search of atoms in vicinity of a
    defined location.
    
    This class stores the indices of an atom array in virtual "cells",
    each corresponding to a specific coordinate interval. If the atoms
    in vicinity of a specific location are searched, only the atoms in
    the relevant cells are checked. Effectively this decreases the
    operation time from *O(n)* to *O(1)*, after the `CellList` has been
    created. Therefore an `CellList` saves calculation time in those
    cases, where vicinity is checked for multiple locations.
    
    Parameters
    ----------
    atom_array : AtomArray
        The `AtomArray` to create the `CellList` for.
    cell_size: float
        The coordinate interval each cell has for x, y and z axis.
        The amount of cells depends on the range of coordinates in the
        `atom_array` and the `cell_size`.
            
    Examples
    --------
    
    >>> cell_list = CellList(atom_array, cell_size=5)
    >>> near_atoms = atom_array[cell_list.get_atoms([1,2,3], radius=7)]
    """
    
    cdef float32[:,:] _coord
    cdef ptr[:,:,:] _cells
    cdef int[:,:,:] _cell_length
    cdef float _cellsize
    cdef float32[:] _min_coord
    cdef float32[:] _max_coord
    cdef int _max_cell_length
    
    def __cinit__(self, atom_array not None, float cell_size):
        cdef float32 x, y, z
        cdef int i, j, k
        cdef int atom_array_i
        cdef int* cell_ptr = NULL
        cdef int length
        
        if self._has_initialized_cells():
            raise Exception("Duplicate call of constructor")
        self._cells = None
        if cell_size <= 0:
            raise ValueError("Cell size must be greater than 0")
        if atom_array.coord is None:
            raise ValueError("Atom array must not be empty")
        if np.isnan(atom_array.coord).any():
            raise ValueError("Atom array contains NaN values")
        coord = atom_array.coord.astype(np.float32)
        self._coord = coord
        self._cellsize = cell_size
        # calculate how many cells are required for each dimension
        min_coord = np.min(coord, axis=0).astype(np.float32)
        max_coord = np.max(coord, axis=0).astype(np.float32)
        self._min_coord = min_coord
        self._max_coord = max_coord
        cell_count = (((max_coord - min_coord) / cell_size) +1).astype(int)
        # ndarray of pointers to C-arrays
        # containing indices to atom array
        self._cells = np.zeros(cell_count, dtype=np.uint64)
        # Stores the length of the C-arrays
        self._cell_length = np.zeros(cell_count, dtype=np.int32)
        # Fill cells
        for atom_array_i in range(self._coord.shape[0]):
            x = self._coord[atom_array_i, 0]
            y = self._coord[atom_array_i, 1]
            z = self._coord[atom_array_i, 2]
            # Get cell indices for coordinates
            self._get_cell_index(x, y, z, &i, &j, &k)
            # Increment cell length and reallocate
            length = self._cell_length[i,j,k] + 1
            cell_ptr = <int*>self._cells[i,j,k]
            cell_ptr = <int*>realloc(cell_ptr, length * sizeof(int))
            if not cell_ptr:
                raise MemoryError()
            # Potentially increase max cell length
            if length > self._max_cell_length:
                self._max_cell_length = length
            # Store atom array index in respective cell
            cell_ptr[length-1] = atom_array_i
            # Store new cell pointer and length
            self._cell_length[i,j,k] = length
            self._cells[i,j,k] = <ptr> cell_ptr
            
    def __dealloc__(self):
        if self._has_initialized_cells():
            deallocate_ptrs(self._cells)
    
    def get_atoms(self, np.ndarray coord, float32 radius):
        """
        Search for atoms in vicinity of the given position.
        
        Parameters
        ----------
        coord : ndarray, dtype=float
            The central coordinates, around which the atoms are
            searched.
        radius: float
            The radius around `coord`, in which the atoms are searched,
            i.e. all atoms in `radius` distance to `coord` are returned.
        
        Returns
        -------
        indices : ndarray, dtype=int
            The indices of the atom array, where the atoms are in the
            defined vicinity around `coord`.
            
        See Also
        --------
        get_atoms_in_cell
        """
        cdef np.ndarray indices = \
            self.get_atoms_in_cell(coord, int(radius/self._cellsize)+1)
        cdef np.ndarray sel_coord = np.asarray(self._coord)[indices]
        dist = distance(sel_coord, coord)
        return indices[dist <= radius]
    
    def get_atoms_in_cell(self, np.ndarray coord,
                         int cell_r=1,
                         bint efficient_mode=False,
                         np.ndarray array_indices=None):
        """
        Search for atoms in vicinity of the given cell.
        
        This is more efficient than `get_atoms()`.
        
        Parameters
        ----------
        coord : ndarray, dtype=float
            The central coordinates, around which the atoms are
            searched.
        cell_r: float, optional
            The radius around `coord` (in amount of cells), in which
            the atoms are searched. This does not correspond to the
            Euclidian distance used in `get_atoms()`. In this case, all
            atoms in the cell corresponding to `coord` and in adjacent
            cells are returned.
            By default atoms are searched in the cell of `coord`
            and adjacent cells.
        efficient_mode : bool, optional
            If enabled, the method will be much more efficient for
            multiple calls of this method with the same `cell_r`.
            Rather than creating a new `ndarray` buffer for the
            indices, the indices will be put into the provided
            `array_indices` buffer.
            (Default: false)
        array_indices : ndarray, dtype=int, optional
            If provided, the method will put the indices into the
            buffer of this `ndarray`.
            This increases the performance for multiple
            calls, since no new `ndarray` needs to be created.
            Note, that the array needs sufficient size (larger than
            the actual amount of resulting indices), that is dependent
            on `cell_r`.
            When you call the method the first time in
            `efficient_mode`, leave this parameter out, a reusable
            array of sufficient size will be created for you.
            Has no effect if `efficient_mode` is false.
            The array will only be partly updated with meaningful
            values. The range of the array, that is filled with
            meaningful values is returned in `length`.
        
        Returns
        -------
        indices : ndarray, dtype=int
            The indices of the atom array, where the atoms are in the
            defined vicinity around `coord`. If `efficient_mode`
            is enabled. the `length` return value gives the range of
            `indices`, that is filled with meaningful values.
        length : int, optional
            Exclusive range of meaningful values in `efficient_mode`.
            
        See Also
        --------
        get_atoms
        """
        # Pessimistic assumption on index array length requirement
        cdef int length
        cdef int index_array_length = (2*cell_r + 1)**3 * self._max_cell_length
        if not efficient_mode or array_indices is None:
            array_indices = np.zeros(index_array_length, dtype=np.int32)
        else:
            # Check if index array has sufficient size and dtype
            if array_indices.dtype != np.int32 \
                or len(array_indices) < index_array_length:
                    raise ValueError("Optionally provided index array"
                                     "is insufficient")
        # Fill index array
        length = self._get_atoms_in_cell(coord.astype(np.float32, copy=False),
                                        array_indices, cell_r)
        if not efficient_mode:
            return array_indices[:length]
        else:
            return array_indices, length
    
    
    def _get_atoms_in_cell(self,
                          float32[:] coord not None,
                          int[:] indices not None,
                          int cell_r=1):
        cdef int length
        cdef int* ptr
        cdef float32 x, y,z
        cdef int i=0, j=0, k=0
        cdef int adj_i, adj_j, adj_k
        cdef int array_i, cell_i
        x = coord[0]
        y = coord[1]
        z = coord[2]
        self._get_cell_index(x, y, z, &i, &j, &k)
        array_i = 0
        # Look into cells of the indices and adjacent cells
        # in all 3 dimensions
        for adj_i in range(i-cell_r, i+cell_r+1):
            if (adj_i >= 0 and adj_i < self._cells.shape[0]):
                for adj_j in range(j-cell_r, j+cell_r+1):
                    if (adj_j >= 0 and adj_j < self._cells.shape[1]):
                        for adj_k in range(k-cell_r, k+cell_r+1):
                            if (adj_k >= 0 and adj_k < self._cells.shape[2]):
                                # Fill with index array
                                # with indices in cell
                                ptr = <int*>self._cells[adj_i, adj_j, adj_k]
                                length = self._cell_length[adj_i, adj_j, adj_k]
                                for cell_i in range(length):
                                    indices[array_i] = ptr[cell_i]
                                    array_i += 1
        # return the actual length of the index array
        return array_i
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void _get_cell_index(self, float32 x, float32 y, float32 z,
                             int* i, int* j, int* k):
        i[0] = <int>((x - self._min_coord[0]) / self._cellsize)
        j[0] = <int>((y - self._min_coord[1]) / self._cellsize)
        k[0] = <int>((z - self._min_coord[2]) / self._cellsize)
    
    
    cdef inline bint _check_coord(self, float32 x, float32 y, float32 z):
        if x < self._min_coord[0] or x > self._max_coord[0]:
            return False
        if y < self._min_coord[1] or y > self._max_coord[1]:
            return False
        if z < self._min_coord[2] or z > self._max_coord[2]:
            return False
        return True
    
    cdef inline bint _has_initialized_cells(self):
        # Memoryviews are not initialized on class creation
        # This method checks if a the _cells memoryview was initialized
        # and is not None
        try:
            if self._cells is not None:
                return True
            else:
                return False
        except AttributeError:
            return False


cdef inline deallocate_ptrs(ptr[:,:,:] ptrs):
    cdef int i, j, k
    cdef int* cell_ptr
    # Free cell pointers
    for i in range(ptrs.shape[0]):
        for j in range(ptrs.shape[1]):
            for k in range(ptrs.shape[2]):
                cell_ptr = <int*>ptrs[i,j,k]
                free(cell_ptr)