# This script creates a Ramachandran plot
# of the motor domain of dynein (PDB: 3VKH)
# The protein mainly consists of alpha-helices,
# as the plot clearly indicates

import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
import matplotlib.pyplot as plt
import numpy as np

# Download and import file
file = rcsb.fetch("3vkh", "cif", biotite.temp_dir())
atom_array = strucio.get_structure_from(file)
# Calculate backbone dihedral angles
# from one of the two identical chains in the asymmetric unit
phi, psi, omega = struc.dihedral_backbone(atom_array, "A")
# Conversion from radians into degree
phi *= 180/np.pi
psi *= 180/np.pi
# Creation of 2D-histogram
hist, psi_edges, phi_edges = np.histogram2d(
    psi, phi, bins=72, normed=True, range=[[-180,180],[-180,180]])

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.contourf(phi_edges[:-1], psi_edges[:-1], hist, cmap="afmhot",
                 levels=np.arange(0,0.0007,0.000025))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Residue density")
ax.set_xlim(-180, 175)
ax.set_ylim(-180, 175)
ax.set_xlabel("$\phi$")
ax.set_ylabel("$\psi$")
ax.set_title("Ramachandran plot of dynein motor domain")
plt.show()