r"""
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
plt.axis([0, len(residue_ids)+1, 0, 100])
ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
ax.set_yticks([])

# Plot the residue names in order
for residue_name, residue_id in zip(residue_names, residue_ids):
    ax.annotate(residue_name, (residue_id, 1), xycoords='data', ha='center')

# Draw the arrows between basepairs
for base1, base2 in struc.base_pairs(atom_array):
    ax.annotate(
        "", xy=(atom_array.res_id[base2], 4), xycoords='data',
        xytext=(atom_array.res_id[base1], 4), textcoords='data', ha='center',
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=-0.5",
        ),
    )

# Display the plot
plt.show()