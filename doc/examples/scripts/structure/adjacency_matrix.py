r"""
Construction of an adjacency matrix
===================================

In this example we create an adjacency matrix of the CA atoms in the
lysozyme crystal structure (PDB: 1AKI).
The values in the adjacency matrix ``m`` are
``m[i,j] = 1 if distance(i,j) <= threshold else 0``. 
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

file_name = rcsb.fetch("1aki", "mmtf", biotite.temp_dir())
array = strucio.load_structure(file_name)
# We only consider CA atoms
ca = array[array.atom_name == "CA"]
# 7 Angstrom adjacency threshold
threshold = 7
# Create cell list of the CA atom array
# for efficient measurement of adjacency
cell_list = struc.CellList(ca, cell_size=threshold)
adjacency_matrix = cell_list.create_adjacency_matrix(threshold)

figure = plt.figure()
ax = figure.add_subplot(111)
cmap = ListedColormap(["white", biotite.colors["dimgreen"]])
ax.matshow(adjacency_matrix, cmap=cmap)
plt.show()