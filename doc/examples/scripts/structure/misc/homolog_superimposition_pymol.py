import ammolite
from matplotlib.colors import to_rgb
import biotite

PNG_SIZE = (1000, 750)

# General configuration
ammolite.cmd.set("cartoon_oval_length", 0.8)
ammolite.cmd.set("depth_cue", 0)

# Define colors
for color_name, color_value in biotite.colors.items():
    ammolite.cmd.set_color("biotite_" + color_name, to_rgb(color_value))

# Convert to PyMOL
pymol_avidin = ammolite.PyMOLObject.from_structure(avidin)
pymol_streptavidin = ammolite.PyMOLObject.from_structure(streptavidin)

# Set overall colors
pymol_avidin.color("biotite_lightgreen")
pymol_streptavidin.color("biotite_lightorange")

# Set view
pymol_avidin.show_as("cartoon")
pymol_streptavidin.show_as("cartoon")
pymol_avidin.orient()

# Save image
ammolite.cmd.ray(*PNG_SIZE)
ammolite.cmd.png(__image_destination__)
