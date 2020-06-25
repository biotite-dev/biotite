"""
Molecular visualization of caffeine
===================================

.. currentmodule:: biotite.structure.graphics

*Biotite* provides simple interactive molecular visualization via
:func:`plot_atoms()`.
Although it does not produce publication-suitable images,
this function can be a convenient tool for a quick visual analysis of a
structure.

This example displays the small molecule caffeine.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.graphics as graphics
import numpy as np
import matplotlib.pyplot as plt


# Get an atom array for caffeine
# Caffeine has the PDB reside name 'CFF'
caffeine = info.residue("CFF")

# For cosmetic purposes align central rings to x-y plane
n1 = caffeine[caffeine.atom_name == "N1"][0]
n3 = caffeine[caffeine.atom_name == "N3"][0]
n7 = caffeine[caffeine.atom_name == "N7"][0]
# Normal vector of ring plane
normal = np.cross(n1.coord - n3.coord, n1.coord - n7.coord)
# Align ring plane normal to x-y plane normal
caffeine = struc.align_vectors(caffeine, normal, np.array([0,0,1]))

# Caffeine should be colored by element
colors = np.zeros((caffeine.array_length(), 3))
colors[caffeine.element == "H"] = (0.8, 0.8, 0.8) # gray
colors[caffeine.element == "C"] = (0.0, 0.8, 0.0) # green
colors[caffeine.element == "N"] = (0.0, 0.0, 0.8) # blue
colors[caffeine.element == "O"] = (0.8, 0.0, 0.0) # red

# Create a quadratic figure to ensure a correct aspect ratio
# in the visualized molecule
fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.add_subplot(111, projection="3d")
graphics.plot_atoms(
    ax, caffeine, colors, line_width=5, background_color="white",
    zoom=1.5
)
fig.tight_layout()
plt.show()