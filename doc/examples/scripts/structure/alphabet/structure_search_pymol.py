import ammolite
from matplotlib.colors import to_rgb
import biotite

PNG_SIZE = (1000, 750)

ammolite.cmd.set("cartoon_rect_length", 1.0)
ammolite.cmd.set("depth_cue", 0)
ammolite.cmd.set("cartoon_cylindrical_helices", 1)
ammolite.cmd.set("cartoon_helix_radius", 1.5)

# Define colors
for color_name, color_value in biotite.colors.items():
    ammolite.cmd.set_color("biotite_" + color_name, to_rgb(color_value))

pymol_query = ammolite.PyMOLObject.from_structure(query_chain)
pymol_target = ammolite.PyMOLObject.from_structure(target_chain)
pymol_query.show_as("cartoon")
pymol_target.show_as("cartoon")
pymol_query.color("biotite_lightgreen")
pymol_target.color("biotite_lightorange")
pymol_query.orient()

# Save image
ammolite.cmd.ray(*PNG_SIZE)
ammolite.cmd.png(__image_destination__)
