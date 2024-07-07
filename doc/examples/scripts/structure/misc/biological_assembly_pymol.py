import ammolite
import matplotlib.pyplot as plt
import numpy as np

PNG_SIZE = (1000, 1000)


# Convert to PyMOL
# Skip adding bonds, since all atoms but CA are filtered out
pymol_obj = ammolite.PyMOLObject.from_structure(biological_unit)

# Set spheres and color
pymol_obj.zoom(buffer=50)
ammolite.cmd.set("sphere_scale", 1.5)
pymol_obj.show_as("spheres")
chain_ids = np.unique(biological_unit.chain_id)
for chain_id, color in zip(chain_ids, plt.get_cmap("tab20").colors):
    pymol_obj.color(color, biological_unit.chain_id == chain_id)

# Save image
ammolite.cmd.ray(*PNG_SIZE)
ammolite.cmd.png(__image_destination__)
