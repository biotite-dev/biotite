import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.graphics as graphics


MOLECULE_NAME = "CFF"

ITERATION_NUMBER = 6
ELEMENT_FONT_SIZE = 10
BALL_SCALE = 200
RAY_SCALE = 3000
CMAP_NAME = "bwr_r"


# Get an atom array for the selected molecule
# Caffeine has the PDB reside name 'CFF'
molecule = info.residue(MOLECULE_NAME)

# Align molecule with principal component analysis:
# The component with the least variance, i.e. the axis with the lowest
# number of atoms lying over each other, is aligned to the z-axis,
# which points into the plane of the figure
pca = PCA(n_components=3)
pca.fit(molecule.coord)
molecule = struc.align_vectors(molecule, pca.components_[-1], [0, 0, 1])

# Balls should be colored by partial charge
charges = struc.partial_charges(molecule, ITERATION_NUMBER)
# Later this variable stores values between 0 and 1 for use in color map
charge_in_color_range = charges.copy()
# Norm charge values to highest absolute value
charge_in_color_range /= np.max(np.abs(charge_in_color_range))
# Transform range (-1, 1) to range (0,1)
charge_in_color_range = (charge_in_color_range + 1) / 2
# Calculate colors
color_map = plt.get_cmap(CMAP_NAME)
colors = color_map(charge_in_color_range)

# Ball size should be proportional to VdW radius of the respective atom
ball_sizes = np.array(
    [info.vdw_radius_single(e) for e in molecule.element]
) * BALL_SCALE



fig = plt.figure(figsize=(8.0, 16.0))
ax = fig.add_subplot(111, projection="3d")
graphics.plot_ball_and_stick_model(
    ax, molecule, colors, ball_size=ball_sizes, line_width=3,
    line_color=color_map(0.5), background_color=(.05, .05, .05), zoom=1.5
)

for atom in molecule:
    ax.text(
        *atom.coord, atom.element,
        fontsize=ELEMENT_FONT_SIZE, color="black",
        ha="center", va="center", zorder=100
    )

# Gradient of ray strength
ray_full_size = ball_sizes + np.abs(charges) * RAY_SCALE
ray_half_size = ball_sizes + np.abs(charges) * RAY_SCALE/2
ax.scatter(
    *molecule.coord.T, s=ray_full_size, c=colors, linewidth=0, alpha=0.2
)
ax.scatter(
    *molecule.coord.T, s=ray_half_size, c=colors, linewidth=0, alpha=0.2
)

color_bar = fig.colorbar(ScalarMappable(cmap=color_map))
color_bar.set_label("Partial charge", color="white")
color_bar.ax.yaxis.set_tick_params(color="white")
color_bar.outline.set_edgecolor("white")
for label in color_bar.ax.get_yticklabels():
    label.set_color("white")

fig.tight_layout()
plt.show()