import numpy as np
from matplotlib.colors import to_rgb
import biotite
import biotite.structure as struc
import ammolite


PNG_SIZE = (1000, 400)


# Define colors
for color_name, color_value in biotite.colors.items():
    ammolite.cmd.set_color(
        "biotite_" + color_name,
        to_rgb(color_value)
    )

# Convert to PyMOL
chain.bonds = struc.connect_via_distances(chain)
pymol_obj = ammolite.PyMOLObject.from_structure(chain)

# Visualize as stick model
pymol_obj.show_as("sticks")
pymol_obj.color(
    "biotite_lightgreen",
    (chain.res_id % 2 == 0) & (chain.element == "C")
)
pymol_obj.color(
    "biotite_dimgreen",
    (chain.res_id % 2 != 0) & (chain.element == "C")
)
ammolite.cmd.set("depth_cue", 0)

# Adjust camera
pymol_obj.orient()
pymol_obj.zoom(buffer=-7.5)

# Save image
ammolite.cmd.ray(*PNG_SIZE)
ammolite.cmd.png(__image_destination__)