import ammolite
import numpy as np
from matplotlib.colors import to_rgb
import biotite
import biotite.structure as struc

PNG_SIZE = (1000, 700)


# General configuration
ammolite.cmd.set("sphere_scale", 1.5)
ammolite.cmd.set("depth_cue", 0)

# Define colors
for color_name, color_value in biotite.colors.items():
    ammolite.cmd.set_color("biotite_" + color_name, to_rgb(color_value))

# Remove hydrogen and water and convert to PyMOL
structure = structure[(structure.element != "H") & (structure.res_name != "TIP")]
structure.bonds = struc.connect_via_distances(structure)
pymol_obj = ammolite.PyMOLObject.from_structure(structure)

# Configure lipid tails
pymol_obj.color("biotite_lightgreen", structure.chain_id == "A")
pymol_obj.color("biotite_brightorange", structure.chain_id == "B")
pymol_obj.show("sticks", np.isin(structure.chain_id, ("A", "B")))

# Configure lipid heads
pymol_obj.color(
    "biotite_darkgreen", (structure.chain_id == "A") & (structure.atom_name == "P")
)
pymol_obj.color(
    "biotite_dimorange", (structure.chain_id == "B") & (structure.atom_name == "P")
)
pymol_obj.show(
    "spheres", np.isin(structure.chain_id, ("A", "B")) & (structure.atom_name == "P")
)

# Adjust camera
pymol_obj.orient()
ammolite.cmd.turn("x", 90)
pymol_obj.zoom(buffer=-10)

# Save image
ammolite.cmd.ray(*PNG_SIZE)
ammolite.cmd.png(__image_destination__)
