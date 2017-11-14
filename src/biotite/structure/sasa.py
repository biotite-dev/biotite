# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

"""
Use this module to calculate the Solvent Accessible Surface Area (SASA) of
a protein or single atoms.
"""

import numpy as np
from .adjacency import AdjacencyMap
from .geometry import distance
from .util import vector_dot
from .filter import filter_solvent

__all__ = ["sasa"]


def sasa(array, **kwargs):
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
    atom_filter : ndarray(dtype=bool), optional
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
    vdw_radii : string or ndarray(dtype=float), optional
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
    sasa : 1-D ndarray(dtype=bool)
        Atom-wise SASA. `NaN` for atoms where SASA has not been 
        calculated
        (solvent atoms, hydrogen atoms (ProtOr), atoms not in `filter`).
        
    References
    ----------
    
    .. [1] A Shrake and JA Rupley,
       "Environment and exposure to solvent of protein atoms.
       Lysozyme and insulin."
       J Mol Biol, 79,351-371 (1973).
   
    .. [2] J Tsai R Taylor, C Chotia and M Gerstein,
       "The packing densitiy in proteins: standard radii and volumes."
       J Mol Biol, 290, 253-299 (1999).
       
    .. [3] A Bondi,
       "Van der Waals volumes and radii."
       J Phys Chem, 86, 441-451 (1964).
    
    """
    if "probe_radius" in kwargs:
        probe_radius = float(kwargs["probe_radius"])
    else:
        probe_radius = 1.4
    
    if "atom_filter" in kwargs:
        # Filter for all atoms to calculate SASA for
        sasa_filter = np.array(kwargs["atom_filter"]).astype(bool)
    else:
        sasa_filter = np.ones(len(array), dtype=bool)
    # Filter for all atoms that are considered for occlusion calculation
    # sasa_filter is subfilter of occlusion_filter
    occl_filter = np.ones(len(array), dtype=bool)
    
    # Remove water residues, since it is the solvent
    filter = ~filter_solvent(array)
    sasa_filter = sasa_filter & filter
    occl_filter = occl_filter & filter
    if "ignore_ions" in kwargs:
        ignore_ions = np.array(kwargs["ignore_ions"])
    else:
        ignore_ions = True
    if ignore_ions:
        pass
    
    if "point_number" in kwargs:
        point_number = int(kwargs["point_number"])
    else:
        point_number = 100
    if "point_distr" in kwargs:
        point_distr = kwargs["point_distr"]
    else:
        point_distr = "Fibonacci"
    if callable(point_distr):
        sphere_points = function(point_number)
    elif point_distr == "Fibonacci":
        sphere_points = _create_fibonacci_points(point_number)
    else:
        raise ValueError("'" + str(point_distr) +
                         "' is not a valid point distribution")
    
    if "vdw_radii" in kwargs:
        vdw_radii = kwargs["vdw_radii"]
    else:
        vdw_radii = "ProtOr"
    if isinstance(vdw_radii, np.ndarray):
        radii = vdw_radii
        if len(radii) != len(array):
            raise ValueError("VdW radii array contains insufficient"
                             "amount of elements")
    elif vdw_radii == "ProtOr":
        filter = (array.element != "H")
        sasa_filter = sasa_filter & filter
        occl_filter = occl_filter & filter
        radii = np.full(len(array), np.nan)
        for i in np.arange(len(radii))[occl_filter]:
            try:
                radii[i] = _protor_radii[array.res_name[i]][array.atom_name[i]]
            except KeyError:
                radii[i] = _protor_default
    elif vdw_radii == "Single":
        radii = np.full(len(array), np.nan)
        for i in np.arange(len(radii))[occl_filter]:
            radii[i] = _single_radii[array.element[i]]
    else:
        raise KeyError("'" + str(vdw_radii) + 
                       "' is not a valid radii set")
    # Increase atom radii by probe size ("rolling probe")
    radii += probe_radius
    
    # Box size is as large as the maximum distance, 
    # where two atom can intersect.
    # Therefore intersecting atoms are always in the same or adjacent box.
    occl_array = array[occl_filter]
    occl_radii = radii[occl_filter]
    adj_map = AdjacencyMap(occl_array, np.max(occl_radii)*2)
    
    # Only calculate SASA for relevant atoms
    area_per_point = 4.0 * np.pi / point_number
    sasa = np.full(len(array), np.nan)
    for index in np.arange(len(array))[sasa_filter]:
        coord = array.coord[index]
        radius = radii[index]
        # Transform the sphere dots to the current atom
        sphere_points_transformed = sphere_points * radius + coord
        # Get coordinates of adjacent atoms
        adj_indices = adj_map.get_atoms_in_box(coord)
        adj_radii = occl_radii[adj_indices]
        adj_atom_coord = occl_array.coord[adj_indices]
        # Remove all atoms, where the distance to the relevant atom
        # is larger than the sum of the radii,
        # since those atoms do not touch
        # If distance is 0, it is the same atom,
        # and the atom is removed from the list as well
        dist = distance(adj_atom_coord, coord)
        dist_filter = ((dist < adj_radii+radius) & (dist != 0))
        adj_atom_coord = adj_atom_coord[dist_filter]
        adj_radii = adj_radii[dist_filter]
        # Calculate distances between sphere points and adjacent atoms
        diff = (adj_atom_coord[np.newaxis, :, :]
                - sphere_points_transformed[:, np.newaxis, :])
        sq_distance = vector_dot(diff, diff)
        # Calculate the difference between the adjacent atom radius
        # and the distance from the atom to the sphere points
        # If the value for all atoms is larger than 0,
        # the point is solvent accessible
        sq_radius_distance = sq_distance - adj_radii*adj_radii
        min_sq_radius_distance = np.min(sq_radius_distance, axis=1)
        accessible_point_count = np.count_nonzero(min_sq_radius_distance > 0)
        sasa[index] = area_per_point * accessible_point_count * radius*radius
    return sasa


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