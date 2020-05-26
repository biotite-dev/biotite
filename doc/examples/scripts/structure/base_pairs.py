"""
Plotting the basepairs of a tRNA-like-structure
===============================================

In this example we plot a linear secondary-structure diagram of a tRNA
mimic (PDB id: 4P5J) from the Turnip Yellow Mosaic Virus (TYMV).
"""

# Code source: Tom David Müller
# License: BSD 3 clause

from tempfile import gettempdir
import biotite.structure.io.pdb as pdb
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Arc
import numpy as np

# Download the PDB file and read the structure
pdb_file_path = rcsb.fetch("4p5j", "pdb", gettempdir())
pdb_file = pdb.PDBFile.read(pdb_file_path)
atom_array = pdb.get_structure(pdb_file)[0]

# Get the residue names and residue ids of the nucleotides
nucleotides = atom_array[struc.filter_nucleotides(atom_array)]
residue_names = nucleotides[struc.get_residue_starts(nucleotides)].res_name
residue_ids = nucleotides[struc.get_residue_starts(nucleotides)].res_id

# Create a matplotlib pyplot
fig, ax = plt.subplots(figsize=(12, 4))

# Setup the axis
ax.set_xlim(0, len(residue_ids)+1)
ax.set_ylim(0, len(residue_ids)/2)
ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
ax.set_yticks([])

# Remove the frame
plt.box(False)

# Plot the residue names in order
for residue_name, residue_id in zip(residue_names, residue_ids):
    ax.text(residue_id, 0, residue_name, ha='center')

# Draw the arcs between basepairs
for base1, base2 in struc.base_pairs(atom_array):
    arc_center = (
        np.mean((atom_array.res_id[base1],atom_array.res_id[base2])), 1.5
    )
    arc_diameter = abs(atom_array.res_id[base2] - atom_array.res_id[base1])
    arc = Arc(
        arc_center, arc_diameter, arc_diameter, 180, theta1=180, theta2=0 
    )
    ax.add_patch(arc)

# Display the plot
plt.show()