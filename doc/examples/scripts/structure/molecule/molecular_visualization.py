"""
Molecular visualization of a small molecule using Matplotlib
============================================================

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import biotite.structure as struc
import biotite.structure.graphics as graphics
import biotite.structure.info as info

# Get an atom array for caffeine
# Caffeine has the PDB reside name 'CFF'
caffeine = info.residue("CFF")

# For cosmetic purposes align central rings to x-y plane
n1 = caffeine[caffeine.atom_name == "N1"][0]
n3 = caffeine[caffeine.atom_name == "N3"][0]
n7 = caffeine[caffeine.atom_name == "N7"][0]
# Normal vector of ring plane
normal = np.cross(n1.coord - n3.coord, n1.coord - n7.coord)
# Align ring plane normal to z-axis
caffeine = struc.align_vectors(caffeine, normal, np.array([0, 0, 1]))

# Caffeine should be colored by element
colors = np.zeros((caffeine.array_length(), 3))
colors[caffeine.element == "H"] = (0.8, 0.8, 0.8)  # gray
colors[caffeine.element == "C"] = (0.0, 0.8, 0.0)  # green
colors[caffeine.element == "N"] = (0.0, 0.0, 0.8)  # blue
colors[caffeine.element == "O"] = (0.8, 0.0, 0.0)  # red

fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.add_subplot(111, projection="3d")
graphics.plot_atoms(
    ax, caffeine, colors, line_width=5, background_color="white", zoom=1.5
)
fig.tight_layout()


# Create an animation that rotates the molecule about the x-axis
def update(angle):
    ax.elev = angle


FPS = 50
DURATION = 4
angles = np.linspace(-180, 180, DURATION * FPS)
# Start at 90 degrees
angles = np.concatenate(
    [
        np.linspace(90, 180, int(DURATION * FPS * 1 / 4)),
        np.linspace(-180, 90, int(DURATION * FPS * 3 / 4)),
    ]
)
animation = FuncAnimation(fig, update, angles, interval=int(1000 / FPS))
plt.show()
