"""
Molecular visualization
=======================

.. currentmodule:: biotite.structure.graphics

*Biotite* provides simple interactive molecular visualization via
:func:`plot_atoms()`.
Although it does not produce publication-suitable images,
this function can be a convenient tool for a quick visual analysis of a
structure.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import biotite
import biotite.structure as struc
import biotite.structure.graphics as graphics
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
import numpy as np
import matplotlib.pyplot as plt

file_name = rcsb.fetch("1l2y", "mmtf", biotite.temp_dir())
# Load only one model
# The bonds are also loaded form the file,
# so that 'plot_atoms()' knows which lines it should draw
atom_array = strucio.load_structure(file_name, model=1, include_bonds=True)

# The structure should be colored by element
colors = np.zeros((atom_array.array_length(), 3))
colors[atom_array.element == "H"] = (1.0, 1.0, 1.0) # white
colors[atom_array.element == "C"] = (0.0, 1.0, 0.0) # green
colors[atom_array.element == "N"] = (0.0, 0.0, 1.0) # blue
colors[atom_array.element == "O"] = (1.0, 0.0, 0.0) # red
colors[atom_array.element == "S"] = (1.0, 1.0, 0.0) # yellow

# Create a quadratic figure to ensure a correct aspect ratio
# in the visualized molecule
fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.add_subplot(111, projection="3d")
graphics.plot_atoms(ax, atom_array, colors, line_width=2)
fig.set_facecolor("black")
ax.set_facecolor("black")
# Restrain the axes to quadratic extents
# to ensure a correct aspect ratio
plt.subplots_adjust(left=-0.3, right=1.3, bottom=-0.3, top=1.3)
plt.show()