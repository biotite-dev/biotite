"""
Calculation of protein diameter
========================================

This script calculates the diameter of a protein
defined as the maximum pairwise atom distance.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb

def get_diameter(pdb_id):
    file_name = rcsb.fetch(pdb_id, "mmtf", biotite.temp_dir())
    atom_array = strucio.load_structure(file_name)
    # Remove all non-amino acids
    atom_array = atom_array[struc.filter_amino_acids(atom_array)]
    coord = atom_array.coord
    # Calculate all pairwise difference vectors
    diff = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]
    # Calculate absolute of difference vectors -> square distances
    sq_dist = np.sum(diff*diff, axis=-1)
    # Maximum distance is diameter
    diameter = np.sqrt(np.max(sq_dist))
    return diameter

# Example application
print("Diameter of 1QAW:", get_diameter("1QAW"), "Angstrom")