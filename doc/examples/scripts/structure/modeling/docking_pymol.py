import ammolite
from matplotlib.colors import to_rgb
import biotite

PNG_SIZE = (1000, 400)


# Define colors
for color_name, color_value in biotite.colors.items():
    ammolite.cmd.set_color("biotite_" + color_name, to_rgb(color_value))

# Convert to PyMOL
pymol_receptor = ammolite.PyMOLObject.from_structure(receptor)
pymol_ref_ligand = ammolite.PyMOLObject.from_structure(ref_ligand)
pymol_docked_ligand = ammolite.PyMOLObject.from_structure(docked_ligand)

# Visualize receptor as surface
pymol_receptor.show_as("surface")
pymol_receptor.color("white")
ammolite.cmd.set("surface_quality", 2)

# Visualize as stick model
ammolite.cmd.set("stick_radius", 0.15)
ammolite.cmd.set("sphere_scale", 0.25)
ammolite.cmd.set("sphere_quality", 4)

# The reference is a blue 'shadow'
REF_COLOR = "skyblue"
REF_ALPHA = 0.4
pymol_ref_ligand.show("spheres")
pymol_ref_ligand.color(REF_COLOR)
pymol_ref_ligand.set("stick_color", REF_COLOR)
pymol_ref_ligand.set("sphere_transparency", 1 - REF_ALPHA)
pymol_ref_ligand.set_bond("stick_transparency", 1 - REF_ALPHA)

pymol_docked_ligand.show("spheres")
pymol_docked_ligand.color("black", docked_ligand.element == "C")
pymol_docked_ligand.set("stick_color", "grey80")

# Adjust camera
pymol_docked_ligand.orient()
ammolite.cmd.rotate("y", 180)
ammolite.cmd.rotate("x", -15)
pymol_docked_ligand.zoom(buffer=-1)
ammolite.cmd.set("depth_cue", 0)
ammolite.cmd.clip("slab", 100)

# Save image
ammolite.cmd.ray(*PNG_SIZE)
ammolite.cmd.png(__image_destination__)
