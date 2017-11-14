# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

"""
This module allows efficient search of atoms in a defined radius around
a location.
"""

import numpy as np
from .geometry import distance

__all__ = ["AdjacencyMap"]


class AdjacencyMap(object):
    """
    This class enables the efficient search of atoms in vicinity of a
    defined location.
    
    This class stores the indices of an atom array in virtual "boxes",
    each corresponding to a specific coordinate interval. If the atoms
    in vicinity of a specific location are searched, only the atoms in
    the relevant boxes are checked. Effectively this decreases the
    operation time from *O(n)* to *O(1)*, after the `AdjacencyMap` has been
    created. Therefore an `AdjacencyMap` saves calculation time in those
    cases, where vicinity is checked for multiple locations.
    
    Parameters
    ----------
    atom_array : AtomArray
        The `AtomArray` to create the `AdjacencyMap` for.
    box_size: float
        The coordinate interval each box has for x, y and z axis.
        The amount of boxes depend on the protein size and the
        `box_size`.
            
    Examples
    --------
    
        >>> adj_map = AdjacencyMap(atom_array, box_size=5)
        >>> near_atoms = atom_array[adj_map.get_atoms([1,2,3], radius=7)]
        
    """
    
    def __init__(self, atom_array, box_size):
        self.array = atom_array.copy()
        self.boxsize = box_size
        # calculate how many boxes are required for each dimension
        self.min_coord = np.min(self.array.coord, axis=-2)
        self.max_coord = np.max(self.array.coord, axis=-2)
        self.box_count = ((((self.max_coord-self.min_coord) / box_size)+1)
                          .astype(int))
        self.boxes = np.zeros(self.box_count, dtype=object)
        # Fill boxes with empty lists, cannot use ndarray.fill(),
        # since it fills the entire array with the same list instance
        for x in range(self.boxes.shape[0]):
            for y in range(self.boxes.shape[1]):
                for z in range(self.boxes.shape[2]):
                    self.boxes[x,y,z] = []
        for i, pos in enumerate(self.array.coord):
            self.boxes[self._get_box_index(pos)].append(i)
    
    def get_atoms(self, coord, radius):
        """
        Search for atoms in vicinity of the given position.
        
        Parameters
        ----------
        coord : ndarray(dtype=float)
            The central coordinates, around which the atoms are
            searched.
        radius: float
            The radius around `coord`, in which the atoms are searched,
            i.e. all atoms in `radius` distance to `coord` are returned.
        
        Returns
        -------
        indices : ndarray(dtype=int)
            The indices of the atom array, where the atoms are in the
            defined vicinity around `coord`.
            
        See Also
        --------
        get_atoms_in_box
        """
        indices = self.get_atoms_in_box(coord, int(radius/self.boxsize)+1)
        sel_coord = self.array.coord[indices]
        dist = distance(sel_coord, coord)
        return indices[dist <= radius]
    
    def get_atoms_in_box(self, coord, box_r=1):
        """
        Search for atoms in vicinity of the given position.
        
        Parameters
        ----------
        coord : ndarray(dtype=float)
            The central coordinates, around which the atoms are
            searched.
        box_r: float
            The radius around `coord` (in amount of boxes), in which the
            atoms are searched. This does not correspond to the
            Euclidian distance used in `get_atoms()`. In this case, all
            atoms in the box corresponding to `coord` and in adjacent
            boxes are returned.
        
        Returns
        -------
        indices : ndarray(dtype=int)
            The indices of the atom array, where the atoms are in the
            defined vicinity around `coord`.
            
        See Also
        --------
        get_atoms
        """
        box_i =  self._get_box_index(coord)
        atom_indices = []
        shape = self.boxes.shape
        for x in range(box_i[0]-box_r, box_i[0]+box_r+1):
            if (x >= 0 and x < shape[0]):
                for y in range(box_i[1]-box_r, box_i[1]+box_r+1):
                    if (y >= 0 and y < shape[1]):
                        for z in range(box_i[2]-box_r, box_i[2]+box_r+1):
                            if (z >= 0 and z < shape[2]):
                                atom_indices.extend(self.boxes[x,y,z])
        return np.array(atom_indices)
    
    def get_atom_array(self):
        """
        Get the atom array corresponding to the adjacency map.
        
        Returns
        -------
        array : AtomArray
            The atom array corresponding to the map.
            
        See Also
        --------
        get_atoms
        """
        return self.array
    
    def _get_box_index(self, coord):
        return tuple(((coord-self.min_coord) / self.boxsize).astype(int))
    
    def __str__(self):
        return(self.boxes)