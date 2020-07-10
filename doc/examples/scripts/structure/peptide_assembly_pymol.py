import numpy as np
from matplotlib.colors import to_rgb
import biotite
import biotite.structure as struc
import ammolite


PNG_SIZE = (1000, 300)


# Define colors
for color_name, color_value in biotite.colors.items():
    ammolite.cmd.set_color(
        "biotite_" + color_name,
        to_rgb(color_value)
    )

# Convert to PyMOL
atom_array.bonds = struc.connect_via_distances(atom_array)
pymol_obj = ammolite.PyMOLObject.from_structure(atom_array)

# Visualize as stick model
pymol_obj.show_as("sticks")
pymol_obj.color(
    "biotite_lightgreen",
    (atom_array.res_id % 2 == 0) & (atom_array.element == "C")
)
pymol_obj.color(
    "biotite_dimgreen",
    (atom_array.res_id % 2 != 0) & (atom_array.element == "C")
)
ammolite.cmd.set("depth_cue", 0)

# Adjust camera
pymol_obj.orient()
pymol_obj.zoom(buffer=-9)

# Save image
ammolite.cmd.ray(*PNG_SIZE)
ammolite.cmd.png(__image_destination__)