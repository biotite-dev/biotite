import numpy as np
from matplotlib.colors import to_rgb
import biotite
import biotite.structure as struc
import ammolite


PNG_SIZE = (1000, 750)

# General configuration
ammolite.cmd.set("cartoon_oval_length", 0.8)
ammolite.cmd.set("depth_cue", 0)

# Define colors
for color_name, color_value in biotite.colors.items():
    ammolite.cmd.set_color(
        "biotite_" + color_name,
        to_rgb(color_value)
    )

# Add bonds to structures and convert to PyMOL
ku_dna.bonds          = struc.connect_via_residue_names(ku_dna)
ku_superimposed.bonds = struc.connect_via_residue_names(ku_superimposed)
pymol_obj_1 = ammolite.PyMOLObject.from_structure(ku_dna)
pymol_obj_2 = ammolite.PyMOLObject.from_structure(ku_superimposed)


# Set overall colors
pymol_obj_1.color("biotite_lightorange", ku_dna.chain_id == "A")
pymol_obj_1.color("biotite_dimorange",   ku_dna.chain_id == "B")
pymol_obj_2.color("biotite_lightgreen",  ku_superimposed.chain_id == "A")
pymol_obj_2.color("biotite_green",       ku_superimposed.chain_id == "B")

# Set view
pymol_obj_1.orient()

# Save image
ammolite.cmd.ray(*PNG_SIZE)
ammolite.cmd.png(__image_destination__)