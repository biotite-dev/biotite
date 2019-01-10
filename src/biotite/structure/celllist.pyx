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
from .atoms import coord as to_coord
from .box import repeat_box_coord, move_inside_box

ctypedef np.uint64_t ptr
ctypedef np.float32_t float32
ctypedef np.uint8_t uint8


cdef class CellList:
    """
    __init__(atom_array, cell_size, periodic=False)
    
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
        Alternatively the atom coordiantes are accepted directly
        (only if `periodic` is false).
    cell_size : float
        The coordinate interval each cell has for x, y and z axis.
        The amount of cells depends on the range of coordinates in the
        `atom_array` and the `cell_size`.
    periodic : bool, optional
        If true, the cell list considers periodic copies of atoms.
        The periodicity is based on the `box` attribute of `atom_array`.
        (Default: False)
            
    Examples
    --------
    
    >>> cell_list = CellList(atom_array, cell_size=5)
    >>> near_atoms = atom_array[cell_list.get_atoms(np.array([1,2,3]), radius=7.0)]
    """
    
    # The atom coordinates
    cdef float32[:,:] _coord
    # The cells to store the coordinates in; an ndarray of pointers
    cdef ptr[:,:,:] _cells
    # The amount elements in each C-array in '_cells'
    cdef int[:,:,:] _cell_length
    # The maximum value of '_cell_length' over all cells,
    # required for worst case assumption on size of output arrays
    cdef int _max_cell_length
    # The length of the cell in each direction (x,y,z)
    cdef float _cellsize
    # The minimum and maximum coordinates for all atoms
    # Used as origin ('_min_coord' is at _cells[0,0,0])
    # and for bound checks
    cdef float32[:] _min_coord
    cdef float32[:] _max_coord
    # Indicates whether the cell list takes periodicity into account
    cdef bint _periodic
    cdef np.ndarray _box
    cdef int _orig_length
    cdef float32[:] _orig_min_coord
    cdef float32[:] _orig_max_coord
    
    
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, atom_array not None,
                  float cell_size, bint periodic=False):
        cdef float32 x, y, z
        cdef int i, j, k
        cdef int atom_array_i
        cdef int* cell_ptr = NULL
        cdef int length

        # the length of the array before appending periodic copies
        # if 'periodic' is true
        self._orig_length = atom_array.array_length()
        self._box = None
        coord = to_coord(atom_array)
        if len(coord) == 0:
            raise ValueError("Atom array must not be empty")
        if np.isnan(coord).any():
            raise ValueError("Coordinates contain NaN values")

        if periodic:
            if atom_array.box is None:
                raise ValueError(
                    "AtomArray must have a box to enable periodicity"
                )
            if atom_array.box.shape != (3,3):
                raise ValueError(
                    "Box has ionvalid shape"
                )
            if np.isnan(atom_array.box).any():
                raise ValueError("Box contains NaN values")
            self._box = atom_array.box
            coord = move_inside_box(coord, self._box)
            atom_array, indices = repeat_box_coord(coord, self._box)
        
        if self._has_initialized_cells():
            raise Exception("Duplicate call of constructor")
        self._cells = None
        if cell_size <= 0:
            raise ValueError("Cell size must be greater than 0")
        self._periodic = periodic
        self._coord = coord.astype(np.float32)
        self._cellsize = cell_size
        # calculate how many cells are required for each dimension
        min_coord = np.min(coord, axis=0).astype(np.float32)
        max_coord = np.max(coord, axis=0).astype(np.float32)
        self._min_coord = min_coord
        self._max_coord = max_coord
        cell_count = (((max_coord - min_coord) / cell_size) +1).astype(int)
        if self._periodic:
            self._orig_min_coord = np.min(coord[:self._orig_length], axis=0) \
                                   .astype(np.float32)
            self._orig_max_coord = np.max(coord[:self._orig_length], axis=0) \
                                   .astype(np.float32)
        
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
        create_adjacency_matrix(threshold_distance)
        
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
        >>> cell_list = CellList(atom_array, 5)
        >>> matrix = cell_list.create_adjacency_matrix(5)
        """
        if threshold_distance < 0:
            raise ValueError("Threshold must be a positive value")
        cdef int i=0, j=0
        cdef int index
        cdef int[:,:] adjacent_indices
        adjacent_indices = self.get_atoms(
            np.asarray(self._coord[:self._orig_length]), threshold_distance
        )
        return self._as_mask(adjacent_indices)
        
    
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_atoms(self, np.ndarray coord, radius, bint as_mask=False):
        """
        get_atoms(coord, radius, as_mask=False)
        
        Find atoms with a maximum distance from given coordinates.
        
        Parameters
        ----------
        coord : ndarray, dtype=float, shape=(3,) or shape=(m,3)
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
        radius : float or ndarray, shape=(n,), dtype=float, optional
            The radius around `coord`, in which the atoms are searched,
            i.e. all atoms in `radius` distance to `coord` are returned.
            Either a single radius can be given as scalar, or individual
            radii for each position in `coord` can be provided as
            `ndarray`.
        as_mask : bool, optional
            If true, the result is returned as boolean mask, instead
            of an index array
        
        Returns
        -------
        indices : ndarray, dtype=int32, shape=(n,) or shape=(m,n)
            The indices of the atom array, where the atoms are in the
            defined `radius` around `coord`.
            If `coord` contains multiple positions, this return value is
            two-dimensional with trailing `-1` values for empty values.
            Only returned with `as_mask` set to false.
        mask : ndarray, dtype=bool, shape=(m,p) or shape=(p,)
            Same as `indices`, but as boolean mask.
            The values are true for atoms in the atom array,
            that are in the defined vicinity.
            Only returned with `as_mask` set to true.
            
        See Also
        --------
        get_atoms_in_cells

        Examples
        --------
        Get adjacent atoms for a single position:

        >>> cell_list = CellList(atom_array, 3)
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

        >>> cell_list = CellList(atom_array, 3)
        >>> pos = np.array([[1.0,2.0,3.0], [2.0,3.0,4.0], [3.0,4.0,5.0]])
        >>> indices = cell_list.get_atoms(pos, radius=3.0)
        >>> print(indices)
        [[ 99 102 104 112 114  45  55 290 101 105 271 273 268  -1  -1]
         [104 114  45  46  55  44  54 105 271 273 265 268 269 272 275]
         [ 46  55 273 268 269 272 274 275  -1  -1  -1  -1  -1  -1  -1]]
        >>> # Convert to list of arrays and remove trailing -1
        >>> indices = [row[row != -1] for row in indices]
        >>> for row in indices:
        ...     print(row)
        [ 99 102 104 112 114  45  55 290 101 105 271 273 268]
        [104 114  45  46  55  44  54 105 271 273 265 268 269 272 275]
        [ 46  55 273 268 269 272 274 275]
        """
        cdef int i=0, j=0
        cdef int array_i = 0
        cdef int max_array_length = 0
        cdef int coord_index
        cdef float32 x1, y1, z1, x2, y2, z2
        cdef float32 sq_dist
        cdef float32 sq_radius
        cdef float32[:] sq_radii
        cdef np.ndarray cell_radii
        
        cdef int[:,:] all_indices
        cdef int[:,:] indices
        cdef float32[:,:] coord_v
        
        # Handle periodicity for the input coordinates
        if self._periodic:
            coord = move_inside_box(coord, self._box)
        # Convert input parameters into a uniform format
        coord, radius, is_multi_coord, is_multi_radius \
            = _prepare_vectorization(coord, radius, np.float32)
        if is_multi_radius:
            sq_radii = radius * radius
            cell_radii = np.floor_divide(
                radius, self._cellsize
            ).astype(np.int32) + 1
        else:
            # All radii are equal
            sq_radii = np.full(
                len(coord), radius[0]*radius[0], dtype=np.float32
            )
            cell_radii = np.full(
                len(coord), int(radius[0]/self._cellsize)+1, dtype=np.int32
            )

        # Get indices for adjacent atoms, based on a cell radius
        all_indices = self._get_atoms_in_cells(
            coord, cell_radii, is_multi_radius
        )
        # These have to be narrowed down in the next step
        # using the Euclidian distance
        
        # Filter all indices from all_indices
        # where squared distance is smaller than squared radius
        # Using the squared distance is computationally cheaper than
        # calculating the sqaure root for every distance
        indices = np.full(
            (all_indices.shape[0], all_indices.shape[1]), -1, dtype=np.int32
        )
        coord_v = coord
        for i in range(all_indices.shape[0]):
            sq_radius = sq_radii[i]
            x1 = coord_v[i,0]
            y1 = coord_v[i,1]
            z1 = coord_v[i,2]
            array_i = 0
            for j in range(all_indices.shape[1]):
                coord_index = all_indices[i,j]
                if coord_index != -1:
                    x2 = self._coord[coord_index, 0]
                    y2 = self._coord[coord_index, 1]
                    z2 = self._coord[coord_index, 2]
                    sq_dist = squared_distance(x1, y1, z1, x2, y2, z2)
                    if sq_dist <= sq_radius:
                        indices[i, array_i] = coord_index
                        array_i += 1
            if array_i > max_array_length:
                max_array_length = array_i
        
        return self.post_process(
            np.asarray(indices)[:, :max_array_length],
            as_mask, is_multi_coord
        )
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_atoms_in_cells(self, np.ndarray coord,
                           cell_radius=1, bint as_mask=False):
        """
        get_atoms_in_cells(coord, cell_radius=1, as_mask=False)
        
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
        coord : ndarray, dtype=float, shape=(3,) or shape=(m,3)
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
        cell_radius : int or ndarray, shape=(n,), dtype=int, optional
            The radius around `coord` (in amount of cells), in which
            the atoms are searched. This does not correspond to the
            Euclidian distance used in `get_atoms()`. In this case, all
            atoms in the cell corresponding to `coord` and in adjacent
            cells are returned.
            Either a single radius can be given as scalar, or individual
            radii for each position in `coord` can be provided as
            `ndarray`.
            By default atoms are searched in the cell of `coord`
            and directly adjacent cells (cell_radius = 1).
        
        Returns
        -------
        indices : ndarray, dtype=int32, shape=(n,) or shape=(m,n)
            The indices of the atom array, where the atoms are in the
            defined `radius` around `coord`.
            If `coord` contains multiple positions, this return value is
            two-dimensional with trailing `-1` values for empty values.
            Only returned with `as_mask` set to false.
        mask : ndarray, dtype=bool, shape=(m,p) or shape=(p,)
            Same as `indices`, but as boolean mask.
            The values are true for atoms in the atom array,
            that are in the defined vicinity.
            Only returned with `as_mask` set to true.

        See Also
        --------
        get_atoms
        """
        # This function is a thin wrapper around the private method
        # with the same name, with addition of handling periodicty
        # and the ability to return a mask instead of indices

        # Handle periodicity for the input coordinates
        if self._periodic:
            coord = move_inside_box(coord, self._box)
        # Convert input parameters into a uniform format
        coord, cell_radius, is_multi_coord, is_multi_radius \
            = _prepare_vectorization(coord, cell_radius, np.int32)
        # Get adjacent atom indices
        array_indices = self._get_atoms_in_cells(
            coord, cell_radius, is_multi_radius
        )
        return self.post_process(array_indices, as_mask, is_multi_coord)
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _get_atoms_in_cells(self,
                            np.ndarray coord,
                            np.ndarray cell_radii,
                            bint is_multi_radius):
        """
        Get the indices of atoms in `cell_radii` adjacency of `coord`.
        
        Parameters
        ----------
        coord : ndarray, dtype=float32, shape=(n,3)
            The position to find adjacent atoms for.
        cell_radii : ndarray, dtype=int32, shape=(n)
            The radius for each position.
        is_multi_radius : bool
            True indicates, that all values in `cell_radii` are the same.
        
        Returns
        -------
        array_indices : ndarray, dtype=int32, shape=
            Indices of adjancent atoms.
        """

        cdef int max_cell_radius
        if is_multi_radius:
            max_cell_radius = np.max(cell_radii)
        else:
            # All radii are equal
            max_cell_radius = cell_radii[0]
        # Worst case assumption on index array length requirement:
        # At maximum, the amount of adjacent atoms can only be the
        # maximum amount of atoms per cell times the amount of cells
        # Since the cells extend in 3 dimensions the amount of cells is
        # (2*r + 1)**3
        cdef int length = (2*max_cell_radius + 1)**3 * self._max_cell_length
        array_indices = np.full((len(coord), length), -1, dtype=np.int32)
        # Fill index array
        cdef int max_array_length \
            = self._find_adjacent_atoms(coord, array_indices, cell_radii)
        return array_indices[:, :max_array_length]
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _find_adjacent_atoms(self,
                                  float32[:,:] coord,
                                  int[:,:] indices,
                                  int[:] cell_radius):
        """
        This method fills the given empty index array
        with actual indices of adjacent atoms.
    
        Since the length of 'indices' (second dimension) is
        the worst case assumption, this method returns the actual
        required length, i.e. the highest length of all arrays
        in this 'array of arrays'.
        """
        cdef int length
        cdef int* list_ptr
        cdef float32 x, y,z
        cdef int i=0, j=0, k=0
        cdef int adj_i, adj_j, adj_k
        cdef int pos_i, array_i, cell_i
        cdef int max_array_length = 0
        cdef int cell_r
        
        cdef ptr[:,:,:] cells = self._cells
        cdef int[:,:,:] cell_length = self._cell_length

        for pos_i in range(coord.shape[0]):
            array_i = 0
            cell_r = cell_radius[pos_i]
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
            if array_i > max_array_length:
                max_array_length = array_i
        return max_array_length
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def post_process(self,
                     np.ndarray indices,
                     bint as_mask,
                     bint is_multi_coord):
        """
        Post process the resulting indices of adjacent atoms,
        including periodicity handling and optional conversion into a
        boolean matrix.
        """
        # Handle periodicity for the output indices
        if self._periodic:
            # Map indices of repeated coordinates to original
            # coordinates, i.e. the coordiantes in the central box
            # -> Remainder of dividing index by original array length
            # Furthermore this ensures, that the indices have valid
            # values for '_as_mask()'
            indices[indices != -1] %= self._orig_length
        if as_mask:
            matrix = self._as_mask(indices)
            if is_multi_coord:
                return matrix
            else:
                return matrix[0]
        else:
            if is_multi_coord:
                return indices
            else:
                return indices[0]
    
    
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
    
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray _as_mask(self, int[:,:] indices):
        cdef int i,j
        cdef int index
        cdef uint8[:,:] matrix = np.zeros(
            (indices.shape[0], self._orig_length), dtype=np.uint8
        )
        # Fill matrix
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                index = indices[i,j]
                if index == -1:
                    # End of list -> jump to next position
                    break
                matrix[i, index] = True
        return np.asarray(matrix, dtype=bool)
    
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


def _prepare_vectorization(np.ndarray coord, radius, radius_dtype):
    """
    Since `get_atoms()` and `get_atoms_in_cells()`, may take different
    amount of dimensions for the coordinates and the radius to enable
    vectorized compuation, each of these functions would need to handle
    the different cases.

    This function converts the input radius and coordinates into a
    uniform format and also return, whether single/multiple
    radii/coordinates were given.

    The shapes before and after conversion are:
       
       - coord: (3, ), radius: scalar -> coord: (1,3), radius: (1,)
       - coord: (n,3), radius: scalar -> coord: (n,3), radius: (n,)
       - coord: (n,3), radius: (n,  ) -> coord: (n,3), radius: (n,)
    
    Thes resulting values have the same dimensionality for all cases and
    can be handeled uniformly by `get_atoms()` and
    `get_atoms_in_cells()`.
    """
    cdef bint is_multi_coord
    cdef bint is_multi_radius

    if coord.ndim == 1 and coord.shape[0] == 3:
        # Single position
        coord = coord[np.newaxis, :].astype(np.float32, copy=False)
        is_multi_coord = False
    elif coord.ndim == 2 and coord.shape[1] == 3:
        # Multiple positions
        coord = coord.astype(np.float32, copy=False)
        is_multi_coord = True
    else:
        raise ValueError(
            f"Invalid shape for input coordinates"
        )
    
    if isinstance(radius, np.ndarray):
        # Multiple radii
        # Check whether amount of coordinates match amount of radii
        if not is_multi_coord:
            raise ValueError(
                "Cannot accept array of radii, if a single position is given"
            )
        if radius.ndim != 1:
            raise ValueError("Array of radii must be one-dimensional")
        if radius.shape[0] != coord.shape[0]:
            raise ValueError(
                f"Amount of radii ({radius.shape[0]}) "
                f"and coordinates ({coord.shape[0]}) are not equal"
            )
        if (radius < 0).any():
            raise ValueError("Radii must be a positive values")
        radius = radius.astype(radius_dtype, copy=False)
        is_multi_radius = True
    else:
        # Single radius
        if radius < 0:
            raise ValueError("Radius must be a positive value")
        # If only a single integer is given,
        # create numpy array filled with identical values
        # with the same length as the coordiantes
        radius = np.full(coord.shape[0], radius, dtype=radius_dtype)
        is_multi_radius = False

    return coord, radius, is_multi_coord, is_multi_radius


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