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

from .atoms import coord as to_coord
import numpy as np

ctypedef np.uint64_t ptr
ctypedef np.float32_t float32
ctypedef np.uint8_t uint8


cdef class CellList:
    """
    This class enables the efficient search of atoms in vicinity of a
    defined location.
    
    This class stores the indices of an atom array in virtual "cells",
    each corresponding to a specific coordinate interval. If the atoms
    in vicinity of a specific location are searched, only the atoms in
    the relevant cells are checked. Effectively this decreases the
    operation time for finding atoms with a maximum distance to given
    coordinates from *O(n)* to *O(1)*, after the `CellList` has been
    created. Therefore an `CellList` saves calculation time in those
    cases, where vicinity is checked for multiple locations.
    
    Parameters
    ----------
    atom_array : AtomArray or ndarray, dtype=float, shape=(n,3)
        The `AtomArray` to create the `CellList` for.
        Alternatively the atom coordiantes are accepted directly.
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
    
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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
        if to_coord(atom_array) is None:
            raise ValueError("Atom array must not be empty")
        if np.isnan(to_coord(atom_array)).any():
            raise ValueError("Atom array contains NaN values")
        coord = to_coord(atom_array).astype(np.float32)
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
    
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def create_adjacency_matrix(self, float32 threshold_distance):
        """
        Create an adjacency matrix for the atoms in this cell list.

        An adjacency matrix depicts which atoms *i* and *j* have
        a distance lower than a given threshold distance.
        The values in the adjacency matrix ``m`` are
        ``m[i,j] = 1 if distance(i,j) <= threshold else 0``
        
        Parameters
        ----------
        threshold_distance : float
            The threshold distance. All atom pairs that have a distance
            lower than this value are indicated by `True` values in
            the resulting matrix.
        
        Returns
        -------
        matrix : ndarray, dtype=bool
            An *n x n* adjacency matrix.
        
        Notes
        -----
        The highest performance is achieved when the the cell size is
        equal to the threshold distance. However, this is purely
        optinal: The resulting adjacency matrix is the same for every
        cell size.

        Examples
        --------
        Create adjacency matrix for CA atoms in a structure:

        >>> atom_array = atom_array[atom_array.atom_name == "CA"]
        >>> cell_list = struc.CellList(atom_array, 5)
        >>> matrix = cell_list.create_adjacency_matrix(5)
        """
        if threshold_distance < 0:
            raise ValueError("Threshold must be a positive value")
        cdef int i=0, j=0
        cdef int index
        cdef int[:,:] adjacent_indices = self.get_atoms(
            np.asarray(self._coord), threshold_distance
        )
        cdef int array_length = self._coord.shape[0]
        cdef uint8[:,:] matrix = np.zeros(
            (array_length, array_length), dtype=np.uint8
        )
        # Fill matrix
        for i in range(adjacent_indices.shape[0]):
            for j in range(adjacent_indices.shape[1]):
                index = adjacent_indices[i,j]
                if index == -1:
                    # end of list -> jump to next position
                    break
                matrix[i, index] = True
        return np.asarray(matrix, dtype=bool)
    
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_atoms(self, np.ndarray coord, float32 radius):
        """
        Find atoms with a maximum distance from given coordinates.
        
        Parameters
        ----------
        coord : ndarray, dtype=float, shape=(3,) or shape=(n,3)
            The central coordinates, around which the atoms are
            searched.
            If a single position is given, the indices of atoms in its
            radius are returned.
            Multiple positions (2-D `ndarray`) have a vectorized
            behavior:
            Each row in the resulting `ndarray` contains the indices for
            the corresponding position.
            Since the positions may have different amounts of adjacent
            atoms, trailing `-1` values are used to indicate nonexisting
            indices.
        radius: float
            The radius around `coord`, in which the atoms are searched,
            i.e. all atoms in `radius` distance to `coord` are returned.
        
        Returns
        -------
        indices : ndarray, dtype=int32, shape=(n,) or shape=(m,n)
            The indices of the atom array, where the atoms are in the
            defined `radius` around `coord`.
            If `coord` contains multiple positions, this return value is
            two-dimensional with trailing `-1` values for empty values.
            
        See Also
        --------
        get_atoms_in_cells

        Examples
        --------
        Get adjacent atoms for single positions:

        >>> cell_list = struc.CellList(atom_array, 3)
        >>> pos = np.array([1.0, 2.0, 3.0])
        >>> indices = cell_list.get_atoms(pos, radius=2.0)
        >>> print(indices)
        [102 104 112]
        >>> print(atom_array[indices])
            A       6 TRP CE3    C         0.779    0.524    2.812
            A       6 TRP CZ3    C         1.439    0.433    4.053
            A       6 TRP HE3    H        -0.299    0.571    2.773
        >>> indices = cell_list.get_atoms(pos, radius=3.0)
        >>> print(atom_array[indices])
            A       6 TRP CD2    C         1.508    0.564    1.606
            A       6 TRP CE3    C         0.779    0.524    2.812
            A       6 TRP CZ3    C         1.439    0.433    4.053
            A       6 TRP HE3    H        -0.299    0.571    2.773
            A       6 TRP HZ3    H         0.862    0.400    4.966
            A       3 TYR CZ     C        -0.639    3.053    5.043
            A       3 TYR HH     H         1.187    3.395    5.567
            A      19 PRO HD2    H         0.470    3.937    1.260
            A       6 TRP CE2    C         2.928    0.515    1.710
            A       6 TRP CH2    C         2.842    0.407    4.120
            A      18 PRO HA     H         2.719    3.181    1.316
            A      18 PRO HB3    H         2.781    3.223    3.618
            A      18 PRO CB     C         3.035    4.190    3.187
        
        Get adjacent atoms for mutliple positions:

        >>> cell_list = struc.CellList(atom_array, 3)
        >>> pos = np.array([[1.0,2.0,3.0], [2.0,3.0,4.0], [3.0,4.0,5.0]])
        >>> indices = cell_list.get_atoms(pos, radius=3.0)
        >>> print(indices)
        [[ 99 102 104 ...  -1  -1  -1]
         [104 114  45 ...  -1  -1  -1]
         [ 46  55 273 ...  -1  -1  -1]]
        >>> # Convert to list of arrays and remove trailing -1
        >>> indices = [row[row != -1] for row in indices]
        >>> for row in indices:
        ...     print(row)
        [ 99 102 104 112 114  45  55 290 101 105 271 273 268]
        [104 114  45  46  55  44  54 105 271 273 265 268 269 272 275]
        [ 46  55 273 268 269 272 274 275]
        """
        cdef int i=0, j=0
        cdef int subset_j = 0
        cdef int coord_index
        cdef float32 x1, y1, z1, x2, y2, z2
        cdef float32 sq_dist
        cdef float32 sq_radius = radius*radius
        
        cdef int[:] indices_single
        cdef int[:] all_indices_single
        cdef float32[:] coord_single
        cdef int[:,:] indices_multi
        cdef int[:,:] all_indices_multi
        cdef float32[:,:] coord_multi

        if radius < 0:
            raise ValueError("Radius must be a positive value")

        all_indices = \
            self.get_atoms_in_cells(coord, <int>(radius/self._cellsize)+1)
        
        # Filter all indices from all_indices
        # where squared distance is smaller than squared radius
        if coord.ndim == 1 and coord.shape[0] == 3:
            # Single position
            all_indices_single = all_indices
            indices_single = np.full(all_indices.shape, -1, dtype=np.int32)
            coord_single = coord.astype(np.float32, copy=False)
            x1 = coord_single[0]
            y1 = coord_single[1]
            z1 = coord_single[2]
            subset_j = 0
            for j in range(all_indices_single.shape[0]):
                coord_index = all_indices_single[j]
                x2 = self._coord[coord_index, 0]
                y2 = self._coord[coord_index, 1]
                z2 = self._coord[coord_index, 2]
                sq_dist = squared_distance(x1, y1, z1, x2, y2, z2)
                if sq_dist < sq_radius:
                    indices_single[subset_j] = coord_index
                    subset_j += 1
            return np.asarray(indices_single)[:subset_j]
                
        elif coord.ndim == 2 and coord.shape[1] == 3:
            # Multiple positions
            all_indices_multi = all_indices
            indices_multi = np.full(all_indices.shape, -1, dtype=np.int32)
            coord_multi = coord.astype(np.float32, copy=False)
            for i in range(all_indices_multi.shape[0]):
                x1 = coord_multi[i,0]
                y1 = coord_multi[i,1]
                z1 = coord_multi[i,2]
                subset_j = 0
                for j in range(all_indices_multi.shape[1]):
                    coord_index = all_indices_multi[i,j]
                    x2 = self._coord[coord_index, 0]
                    y2 = self._coord[coord_index, 1]
                    z2 = self._coord[coord_index, 2]
                    sq_dist = squared_distance(x1, y1, z1, x2, y2, z2)
                    if sq_dist < sq_radius:
                        indices_multi[i, subset_j] = coord_index
                        subset_j += 1
            return np.asarray(indices_multi)
        
        else:
            raise ValueError("Invalid shape for input coordinates")
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_atoms_in_cells(self, np.ndarray coord, int cell_radius=1):
        """
        Find atoms with a maximum cell distance from given
        coordinates.
        
        Instead of using the radius as maximum euclidian distance to the
        given coordinates,
        the radius is measured as the amount of cells:
        A radius of 0 means, that only the atoms in the same cell
        as the given coordinates are considered. A radius of 1 means,
        that the atoms indices from this cell and the 8 surrounding
        cells are returned and so forth.
        This is more efficient than `get_atoms()`.
        
        Parameters
        ----------
        coord : ndarray, dtype=float, shape=(3,) or shape=(n,3)
            The central coordinates, around which the atoms are
            searched.
            If a single position is given, the indices of atoms in its
            cell radius are returned.
            Multiple positions (2-D `ndarray`) have a vectorized
            behavior:
            Each row in the resulting `ndarray` contains the indices for
            the corresponding position.
            Since the positions may have different amounts of adjacent
            atoms, trailing `-1` values are used to indicate nonexisting
            indices.
        cell_r: float, optional
            The radius around `coord` (in amount of cells), in which
            the atoms are searched. This does not correspond to the
            Euclidian distance used in `get_atoms()`. In this case, all
            atoms in the cell corresponding to `coord` and in adjacent
            cells are returned.
            By default atoms are searched in the cell of `coord`
            and directly adjacent cells (cell_radius=1).
        
        Returns
        -------
        indices : ndarray, dtype=int32, shape=(n,) or shape=(m,n)
            The indices of the atom array, where the atoms are in the
            defined `radius` around `coord`.
            If `coord` contains multiple positions, this return value is
            two-dimensional with trailing `-1` values for empty values.
            
        See Also
        --------
        get_atoms
        """
        if cell_radius < 0:
            raise ValueError("Radius must be a positive value")
        cdef int length
        # Pessimistic assumption on index array length requirement:
        # At maximum, tte amount of atoms can only be the maximum
        # amount of atoms per cell times the amount of cell
        # Since the cells extend in 3 dimensions the amount of cells is
        # (2*r + 1)**3
        cdef int index_array_length = \
            (2*cell_radius + 1)**3 * self._max_cell_length
        # The amount of positions,
        # for each one the adjacent atoms are obtained
        cdef int coord_count
        # Save if adjacent atoms are searched for a single position
        # or multiple positions
        cdef bint multi_coord
        # If only a single position is given,
        # the input is treated as multiple positions with an amount of 1  
        if coord.ndim == 1 and coord.shape[0] == 3:
            # Single position
            coord = coord[np.newaxis, :]
            pos_count = 1
            multi_coord = False
        elif coord.ndim == 2 and coord.shape[1] == 3:
            # Multiple positions
            pos_count = coord.shape[0]
            multi_coord = True
        else:
            raise ValueError("Invalid shape for input coordinates")
        array_indices = np.full(
            (pos_count, index_array_length), -1, dtype=np.int32
        )
        # Fill index array
        self._get_atoms_in_cells(coord.astype(np.float32, copy=False),
                                  array_indices, cell_radius)
        if multi_coord:
            return array_indices
        else:
            array_indices = array_indices[0]
            return array_indices[array_indices != -1]
            
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _get_atoms_in_cells(self,
                                 float32[:,:] coord,
                                 int[:,:] indices,
                                 int cell_r=1):
        cdef int length
        cdef int* list_ptr
        cdef float32 x, y,z
        cdef int i=0, j=0, k=0
        cdef int adj_i, adj_j, adj_k
        cdef int pos_i, array_i, cell_i
        
        cdef ptr[:,:,:] cells = self._cells
        cdef int[:,:,:] cell_length = self._cell_length

        for pos_i in range(coord.shape[0]):
            array_i = 0
            x = coord[pos_i, 0]
            y = coord[pos_i, 1]
            z = coord[pos_i, 2]
            self._get_cell_index(x, y, z, &i, &j, &k)
            # Look into cells of the indices and adjacent cells
            # in all 3 dimensions
            for adj_i in range(i-cell_r, i+cell_r+1):
                if (adj_i >= 0 and adj_i < cells.shape[0]):
                    for adj_j in range(j-cell_r, j+cell_r+1):
                        if (adj_j >= 0 and adj_j < cells.shape[1]):
                            for adj_k in range(k-cell_r, k+cell_r+1):
                                if (adj_k >= 0 and adj_k < cells.shape[2]):
                                    # Fill index array
                                    # with indices in cell
                                    list_ptr = <int*>cells[adj_i, adj_j, adj_k]
                                    length = cell_length[adj_i, adj_j, adj_k]
                                    for cell_i in range(length):
                                        indices[pos_i, array_i] = \
                                            list_ptr[cell_i]
                                        array_i += 1
    
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void _get_cell_index(self, float32 x, float32 y, float32 z,
                             int* i, int* j, int* k):
        i[0] = <int>((x - self._min_coord[0]) / self._cellsize)
        j[0] = <int>((y - self._min_coord[1]) / self._cellsize)
        k[0] = <int>((z - self._min_coord[2]) / self._cellsize)
    
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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


cdef inline void deallocate_ptrs(ptr[:,:,:] ptrs):
    cdef int i, j, k
    cdef int* cell_ptr
    # Free cell pointers
    for i in range(ptrs.shape[0]):
        for j in range(ptrs.shape[1]):
            for k in range(ptrs.shape[2]):
                cell_ptr = <int*>ptrs[i,j,k]
                free(cell_ptr)

cdef inline float32 squared_distance(float32 x1, float32 y1, float32 z1,
                    float32 x2, float32 y2, float32 z2):
    cdef float32 diff_x = x2 - x1
    cdef float32 diff_y = y2 - y1
    cdef float32 diff_z = z2 - z1
    return diff_x*diff_x + diff_y*diff_y + diff_z*diff_z