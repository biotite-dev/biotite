# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Use this module to calculate the Solvent Accessible Surface Area (SASA) of
a protein or single atoms.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["sasa"]

cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free

import numpy as np
from .celllist import CellList
from .filter import filter_solvent, filter_monoatomic_ions

ctypedef np.uint8_t np_bool
ctypedef np.int64_t int64
ctypedef np.float32_t float32


@cython.boundscheck(False)
@cython.wraparound(False)
def sasa(array, float probe_radius=1.4, np.ndarray atom_filter=None,
         bint ignore_ions=True, int point_number=1000,
         point_distr="Fibonacci", vdw_radii="ProtOr"):
    """
    Calculate the Solvent Accessible Surface Area (SASA) of a protein.
    
    This function uses the Shrake-Rupley ("rolling probe")
    algorithm [1]_:
    Every atom is occupied by a evenly distributed point mesh. The
    points that can be reached by the "rolling probe", are surface
    accessible.
    
    Parameters
    ----------
    array : AtomArray
        The protein model to calculate the SASA for.
    probe_radius : float, optional
        The VdW-radius of the solvent molecules (default: 1.4).
    atom_filter : ndarray, dtype=bool, optional
        If this parameter is given, SASA is only calculated for the
        filtered atoms.
    ignore_ions : bool, optional
        If true, all monoatomic ions are removed before SASA calculation
        (default: True).
    point_number : int, optional
        The number of points in the mesh occupying each atom for SASA
        calculation (default: 100). The SASA calculation time is
        proportional to the amount of sphere points.
    point_distr : string or function, optional
        If a function is given, the function is used to calculate the
        point distribution for the mesh (the function must take `float`
        *n* as parameter and return a *(n x 3)* `ndarray`).
        Alternatively a string can be given to choose a built-in
        distribution:
            
            - **Fibonacci** - Distribute points using a golden section
              spiral.
            
        By default *Fibonacci* is used.
    vdw_radii : string or ndarray, dtype=float, optional
        Indicates the set of VdW radii to be used. If an `array`-length
        `ndarray` is given, each atom gets the radius at the
        corresponding index. Radii given for atoms that are not used in
        SASA calculation (e.g. solvent atoms) can have arbitrary values
        (e.g. `NaN`). If instead a `string` is given, one of the
        built-in sets is used:
        
            - **ProtOr** - A set, which does not require hydrogen atoms
              in the model. Suitable for crystal structures. [2]_
            - **Single** - A set, which uses a defined VdW radius for
              every single atom, therefore hydrogen atoms are required
              in the model (e.g. NMR elucidated structures). [3]_
              
        By default *ProtOr* is used.
              
    
    Returns
    -------
    sasa : 1-D ndarray, dtype=bool
        Atom-wise SASA. `NaN` for atoms where SASA has not been 
        calculated
        (solvent atoms, hydrogen atoms (ProtOr), atoms not in `filter`).
        
    References
    ----------
    
    .. [1] A Shrake and JA Rupley,
       "Environment and exposure to solvent of protein atoms.
       Lysozyme and insulin."
       J Mol Biol, 79, 351-371 (1973).
   
    .. [2] J Tsai R Taylor, C Chotia and M Gerstein,
       "The packing densitiy in proteins: standard radii and volumes."
       J Mol Biol, 290, 253-299 (1999).
       
    .. [3] A Bondi,
       "Van der Waals volumes and radii."
       J Phys Chem, 86, 441-451 (1964).
    
    """
    cdef int i=0, j=0, k=0, adj_atom_i=0, rel_atom_i=0
    
    cdef np.ndarray sasa_filter
    cdef np.ndarray occl_filter
    if atom_filter is not None:
        # Filter for all atoms to calculate SASA for
        sasa_filter = np.array(atom_filter, dtype=bool)
    else:
        sasa_filter = np.ones(len(array), dtype=bool)
    # Filter for all atoms that are considered for occlusion calculation
    # sasa_filter is subfilter of occlusion_filter
    occl_filter = np.ones(len(array), dtype=bool)
    # Remove water residues, since it is the solvent
    filter = ~filter_solvent(array)
    sasa_filter = sasa_filter & filter
    occl_filter = occl_filter & filter
    if ignore_ions:
        filter = ~filter_monoatomic_ions(array)
        sasa_filter = sasa_filter & filter
        occl_filter = occl_filter & filter
    
    cdef np.ndarray sphere_points
    if callable(point_distr):
        sphere_points = point_distr(point_number)
    elif point_distr == "Fibonacci":
        sphere_points = _create_fibonacci_points(point_number)
    else:
        raise ValueError(f"'{point_distr}' is not a valid point distribution")
    sphere_points = sphere_points.astype(np.float32)
    
    cdef np.ndarray radii
    if isinstance(vdw_radii, np.ndarray):
        radii = vdw_radii.astype(np.float32)
        if len(radii) != array.array_length():
            raise ValueError(
                f"Amount VdW radii ({len(radii)}) and "
                f"amount of atoms ({array.array_length()}) are not equal"
            )
    elif vdw_radii == "ProtOr":
        filter = (array.element != "H")
        sasa_filter = sasa_filter & filter
        occl_filter = occl_filter & filter
        radii = np.full(len(array), np.nan, dtype=np.float32)
        for i in np.arange(len(radii))[occl_filter]:
            try:
                radii[i] = _protor_radii[array.res_name[i]][array.atom_name[i]]
            except KeyError:
                radii[i] = _protor_default
    elif vdw_radii == "Single":
        radii = np.full(len(array), np.nan, dtype=np.float32)
        for i in np.arange(len(radii))[occl_filter]:
            radii[i] = _single_radii[array.element[i]]
    else:
        raise KeyError(f"'{vdw_radii}' is not a valid radii set")
    # Increase atom radii by probe size ("rolling probe")
    radii += probe_radius
    
    # Memoryview for filter
    # Problem with creating boolean memoryviews
    # -> Type uint8 is used
    cdef np_bool[:] sasa_filter_view = np.frombuffer(sasa_filter,
                                                     dtype=np.uint8)
    
    cdef np.ndarray occl_r = radii[occl_filter]
    # Atom array containing occluding atoms
    occl_array = array[occl_filter]
    
    # Memoryviews for coordinates of entire (main) array
    # and for coordinates of occluding atom array
    cdef float32[:,:] main_coord = array.coord.astype(np.float32,
                                                      copy=False)
    cdef float32[:,:] occl_coord = occl_array.coord.astype(np.float32,
                                                           copy=False)
    # Memoryviews for sphere points
    cdef float32[:,:] sphere_coord = sphere_points
    # Check if any of these arrays are empty to prevent segfault
    if     main_coord.shape[0]   == 0 \
        or occl_coord.shape[0]   == 0 \
        or sphere_coord.shape[0] == 0:
            raise ValueError("Coordinates are empty")
    # Memoryviews for radii of SASA and occluding atoms
    # their squares and their sum of sqaures
    cdef float32[:] atom_radii = radii
    cdef float32[:] atom_radii_sq = radii * radii
    cdef float32[:] occl_radii = occl_r
    cdef float32[:] occl_radii_sq = occl_r * occl_r
    # Memoryview for atomwise SASA
    cdef float32[:] sasa = np.full(len(array), np.nan, dtype=np.float32)
    
    # Area of a sphere point on a unit sphere
    cdef float32 area_per_point = 4.0 * np.pi / point_number
    
    # Define further statically typed variables
    # that are needed for SASA calculation
    cdef int n_accesible = 0
    cdef float32 radius = 0
    cdef float32 radius_sq = 0
    cdef float32 adj_radius = 0
    cdef float32 adj_radius_sq = 0
    cdef float32 dist_sq = 0
    cdef float32 point_x = 0
    cdef float32 point_y = 0
    cdef float32 point_z = 0
    cdef float32 atom_x = 0
    cdef float32 atom_y = 0
    cdef float32 atom_z = 0
    cdef float32 occl_x = 0
    cdef float32 occl_y = 0
    cdef float32 occl_z = 0
    cdef float32[:,:] relevant_occl_coord = None
    
    # Cell size is as large as the maximum distance, 
    # where two atom can intersect.
    # Therefore intersecting atoms are always in the same or adjacent cell.
    cell_list = CellList(occl_array, np.max(radii[occl_filter])*2)
    cdef np.ndarray cell_indices
    cdef int[:,:] cell_indices_view
    cdef int length
    cdef int max_adj_list_length = 0
    cdef int array_length = array.array_length()

    cell_indices = cell_list.get_atoms_in_cells(array.coord)
    cell_indices_view = cell_indices
    max_adj_list_length = cell_indices.shape[0]
        
    # Later on, this array stores coordinates for actual
    # occluding atoms for a certain atom to calculate the
    # SASA for
    # The first three indices of the second axis
    # are x, y and z, the last one is the squared radius
    # This list is as long as the maximal length of a list of
    # adjacent atoms
    relevant_occl_coord = np.zeros((max_adj_list_length, 4),
                                   dtype=np.float32)
    
    # Actual SASA calculation
    for i in range(array_length):
        # First level: The atoms to calculate SASA for
        if not sasa_filter_view[i]:
            # SASA is not calculated for this atom
            continue
        n_accesible = point_number
        atom_x = main_coord[i,0]
        atom_y = main_coord[i,1]
        atom_z = main_coord[i,2]
        radius = atom_radii[i]
        radius_sq = atom_radii_sq[i]
        # Find occluding atoms from list of adjacent atoms
        rel_atom_i = 0
        for j in range(max_adj_list_length):
            # Remove all atoms, where the distance to the relevant atom
            # is larger than the sum of the radii,
            # since those atoms do not touch
            # If distance is 0, it is the same atom,
            # and the atom is removed from the list as well
            adj_atom_i = cell_indices_view[i,j]
            if adj_atom_i == -1:
                # -1 means end of list
                break
            occl_x = occl_coord[adj_atom_i,0]
            occl_y = occl_coord[adj_atom_i,1]
            occl_z = occl_coord[adj_atom_i,2]
            adj_radius = occl_radii[adj_atom_i]
            adj_radius_sq = occl_radii_sq[adj_atom_i]
            dist_sq = distance_sq(atom_x, atom_y, atom_z,
                                      occl_x, occl_y, occl_z)
            if dist_sq != 0 \
                and dist_sq < (adj_radius+radius) * (adj_radius+radius):
                    relevant_occl_coord[rel_atom_i,0] = occl_x
                    relevant_occl_coord[rel_atom_i,1] = occl_y
                    relevant_occl_coord[rel_atom_i,2] = occl_z
                    relevant_occl_coord[rel_atom_i,3] = adj_radius_sq
                    rel_atom_i += 1
        for j in range(sphere_coord.shape[0]):
            # Second level: The sphere points for that atom
            # Transform sphere point to sphere of current atom
            point_x = sphere_coord[j,0] * radius + atom_x
            point_y = sphere_coord[j,1] * radius + atom_y
            point_z = sphere_coord[j,2] * radius + atom_z
            for k in range(rel_atom_i):
                # Third level: Compare point to occluding atoms
                dist_sq = distance_sq(point_x, point_y, point_z,
                                      relevant_occl_coord[k, 0],
                                      relevant_occl_coord[k, 1],
                                      relevant_occl_coord[k, 2])
                # Compare squared distance
                # to squared radius of occluding atom
                # (Radius is relevant_occl_coord[3])
                if dist_sq < relevant_occl_coord[k, 3]:
                    # Point is occluded
                    # -> Continue with next point
                    n_accesible -= 1
                    break
        sasa[i] = area_per_point * n_accesible * radius_sq
    return np.asarray(sasa)


cdef inline float32 distance_sq(float32 x1, float32 y1, float32 z1,
                        float32 x2, float32 y2, float32 z2):
    cdef float32 dx = x2 - x1
    cdef float32 dy = y2 - y1
    cdef float32 dz = z2 - z1
    return dx*dx + dy*dy + dz*dz


def _create_fibonacci_points(n):
    """
    Get an array of approximately equidistant points on a sphere surface
    using a golden section spiral.
    """
    phi = (3 - np.sqrt(5)) * np.pi * np.arange(n)
    z = np.linspace(1 - 1.0/n, 1.0/n - 1, n)
    radius = np.sqrt(1 - z*z)
    coords = np.zeros((n, 3))
    coords[:,0] = radius * np.cos(phi)
    coords[:,1] = radius * np.sin(phi)
    coords[:,2] = z
    return coords


_protor_default = 1.80
_protor_radii = {"GLY": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64},
                 "ALA": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88},
                 "VAL": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG1": 1.88,
                         " CG2": 1.88},
                 "LEU": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.88,
                         " CD1": 1.88,
                         " CD2": 1.88},
                 "ILE": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG1": 1.88,
                         " CG2": 1.88,
                         " CD1": 1.88},
                 "PRO": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.88,
                         " CD ": 1.88},
                 "MET": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.88,
                         " SD ": 1.77,
                         " CE ": 1.88},
                 "PHE": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.61,
                         " CD1": 1.76,
                         " CD2": 1.76,
                         " CE1": 1.76,
                         " CE2": 1.76,
                         " CZ ": 1.76},
                 "TYR": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.61,
                         " CD1": 1.76,
                         " CD2": 1.76,
                         " CE1": 1.76,
                         " CE2": 1.76,
                         " CZ ": 1.61,
                         " OH ": 1.46},
                 "TRP": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.61,
                         " CD1": 1.76,
                         " NE1": 1.64,
                         " CD2": 1.61,
                         " CE2": 1.61,
                         " CE3": 1.76,
                         " CZ3": 1.76,
                         " CZ2": 1.76,
                         " CH2": 1.76,
                         "CEH2": 1.76},
                 "SER": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " OG ": 1.46,
                         " OG1": 1.46},
                 "THR": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " OG1": 1.46,
                         " CG2": 1.88,
                         " CG ": 1.88},
                 "ASN": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.61,
                         " OD1": 1.42,
                         " ND2": 1.64},
                 "GLN": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.88,
                         " CD ": 1.61,
                         " OE1": 1.42,
                         " NE2": 1.64},
                 "CYS": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " SG ": 1.77},
                 "CSS": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " SG ": 1.77},
                 "HIS": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.61,
                         " ND1": 1.64,
                         " CD2": 1.76,
                         " CE1": 1.76,
                         " NE2": 1.64},
                 "GLU": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.88,
                         " CD ": 1.61,
                         " OE1": 1.42,
                         " OE2": 1.42},
                 "ASP": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.61,
                         " OD1": 1.42,
                         " OD2": 1.42},
                 "ARG": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.88,
                         " CD ": 1.88,
                         " NE ": 1.64,
                         " CZ ": 1.61,
                         " NH1": 1.64,
                         " NH2": 1.64},
                 "LYS": {" O  ": 1.42,
                         " C  ": 1.61,
                         " CA ": 1.88,
                         " N  ": 1.64,
                         " CB ": 1.88,
                         " CG ": 1.88,
                         " CD ": 1.88,
                         " CE ": 1.88,
                         " NZ ": 1.64}}


_single_radii = {"H":  1.20,
                 "C":  1.70,
                 "N":  1.55,
                 "O":  1.52,
                 "F":  1.47,
                 "Si": 2.10,
                 "P":  1.80,
                 "S":  1.80,
                 "Cl": 1.75,
                 "Br": 1.85,
                 "I":  1.98}