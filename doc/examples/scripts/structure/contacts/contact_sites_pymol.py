import ammolite
import numpy as np
from matplotlib.colors import to_rgb
import biotite
import biotite.structure as struc

PNG_SIZE = (1000, 550)

# General configuration
ammolite.cmd.set("cartoon_side_chain_helper", 1)
ammolite.cmd.set("cartoon_discrete_colors", 1)
ammolite.cmd.set("depth_cue", 0)
ammolite.cmd.set("cartoon_oval_length", 0.8)

# Define colors
for color_name, color_value in biotite.colors.items():
    ammolite.cmd.set_color("biotite_" + color_name, to_rgb(color_value))

# Add bonds to structure and convert to PyMOL
structure = structure[~struc.filter_solvent(structure)]
structure.bonds = struc.connect_via_residue_names(structure)
pymol_obj = ammolite.PyMOLObject.from_structure(structure)

# Set overall colors
pymol_obj.color("gray", np.isin(structure.chain_id, ["A", "B"]))
pymol_obj.color("biotite_brightorange", structure.chain_id == "L")
pymol_obj.color("biotite_lightgreen", structure.chain_id == "R")

# Set view
ammolite.cmd.set_view(
    (
        -0.044524662,
        0.767611504,
        0.639355302,
        0.998693943,
        0.018437184,
        0.047413416,
        0.024606399,
        0.640637815,
        -0.767439663,
        0.000000000,
        0.000000000,
        -115.614288330,
        56.031833649,
        23.317802429,
        3.761308193,
        73.517341614,
        157.711288452,
        -20.000000000,
    )
)

# Highlight contacts
residue_mask = np.isin(structure.res_id, common_ids)
pymol_obj.show("sticks", np.isin(structure.chain_id, ["L", "R"]) & residue_mask)
for chain, color in zip(("L", "R"), ("biotite_dimorange", "biotite_darkgreen")):
    pymol_obj.color(
        color,
        (structure.chain_id == chain) & (structure.atom_name != "CA") & residue_mask,
    )

# Save image
ammolite.cmd.ray(*PNG_SIZE)
ammolite.cmd.png(__image_destination__)
