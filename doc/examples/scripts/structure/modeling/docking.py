"""
Docking a ligand to a receptor
==============================

This example shows how to use *AutoDock Vina* :footcite:`Trott2010`
from within *Biotite* for docking a ligand molecule to a known receptor
structure.
As example, we examine the famous streptavidin-biotin interaction.

At first we download a high resolution structure of the complex.
The resolution is important here:
For one thing, the docking procedure requires annotated hydrogen atoms
for the receptor molecule, which seldom is the case for low resolution
X-ray structures.
On the other hand, we would like to have a reliable reference binding
mode of the ligand, to evaluate how well out docking procedure went

After separation of the receptor and the reference ligand,
a biotin model is loaded from the chemical components dictionary and
docked into the binding cavity of streptavidin.

Finally, the docked model is compared to the reference model, with
respect to their RMSD.

.. footbibliography::
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import biotite.application.autodock as autodock
import biotite.database.rcsb as rcsb
import biotite.interface.pymol as pymol_interface
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx

# Get the receptor structure
# and the original 'correct' conformation of the ligand
pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch("2RTG", "bcif"))
structure = pdbx.get_structure(
    # Include formal charge for accurate partial charge calculation
    pdbx_file,
    model=1,
    include_bonds=True,
    extra_fields=["charge"],
)
# The asymmetric unit describes a streptavidin homodimer
# However, we are only interested in a single monomer
structure = structure[structure.chain_id == "B"]
receptor = structure[struc.filter_amino_acids(structure)]

ref_ligand = structure[structure.res_name == "BTN"]
ref_ligand_center = struc.centroid(ref_ligand)


# Independently, get the ligand without optimized conformation
# from the chemical components dictionary
ligand = info.residue("BTN")

# Search for a binding mode in a 20 Å radius
# of the original ligand position
app = autodock.VinaApp(ligand, receptor, ref_ligand_center, [20, 20, 20])
# For reproducibility
app.set_seed(0)
# This is the maximum number:
# Vina may find less interesting binding modes
# and thus output less models
app.set_max_number_of_models(100)
# Effectively no limit
app.set_energy_range(100.0)
# Start docking run
app.start()
app.join()
docked_coord = app.get_ligand_coord()
energies = app.get_energies()

# Create an AtomArrayStack for all docked binding modes
docked_ligand = struc.from_template(ligand, docked_coord)
# As Vina discards all nonpolar hydrogen atoms, their respective
# coordinates are NaN -> remove these atoms
docked_ligand = docked_ligand[..., ~np.isnan(docked_ligand.coord[0]).any(axis=-1)]


# For comparison of the docked pose with the experimentally determined
# reference conformation, the atom order of both must be exactly the
# same
# Therefore, all atoms, that are additional in one of both models,
# e.g. carboxy or nonpolar hydrogen atoms, are removed...
docked_ligand = docked_ligand[
    ..., np.isin(docked_ligand.atom_name, ref_ligand.atom_name)
]
docked_ligand = docked_ligand[..., info.standardize_order(docked_ligand)]
# ...and the atom order is standardized
ref_ligand = ref_ligand[np.isin(ref_ligand.atom_name, docked_ligand.atom_name)]
ref_ligand = ref_ligand[info.standardize_order(ref_ligand)]

# Calculate the RMSD of the docked models to the correct binding mode
# No superimposition prior to RMSD calculation, as we want to see
# conformation differences with respect to the binding pocket
rmsd = struc.rmsd(ref_ligand, docked_ligand)

# Evaluate correlation between RMSD and binding energies
correlation, p_value = spearmanr(energies, rmsd)

figure, ax = plt.subplots(figsize=(8.0, 6.0))
ax.set_title(f"$r_s$ = {correlation:.2f} ($p$ = {p_value*100:.1f}%)")
ax.scatter(energies, rmsd, marker="+", color="black")
ax.set_xlabel("Energy (kcal/mol)")
ax.set_ylabel("RMSD (Å)")
figure.tight_layout()
plt.show()

########################################################################
# For this specific case *AutoDock Vina* shows only a low Spearman
# correlation between the RMSD of the calculated models to the
# correct binding mode and the associated calculated binding energy.
# A high correlation is desireable to ensure that docking results with
# good binding energies correspond to the correct binding mode for cases
# in which the correct binding conformation is unknown.
# However, at least the calculated model with highest predicted affinity
# is also the conformation with the lowest deviation from the
# experimental result in this instance.
# Hence, *AutoDock Vina* was able to predict an almost correct
# binding mode as its best *guess*.
#
# In a final step, we want to visually compare the experimentally
# determined conformation of biotin in the binding pocket with the
# minimum-energy docked conformation, which is also the conformation
# with the lowest RMSD in this case.
# The docked conformation is shown as ball-and-stick model, the original
# experimentally determined biotin conformation is shown in transparent
# blue.


# Get the best fitting model,
# i.e the model with the lowest RMSD to the reference conformation
docked_ligand = docked_ligand[np.argmin(rmsd)]

# Vina only keeps polar hydrogens in the modeled structure
# For consistency, remove all hydrogen atoms in the reference and
# modelled structure
ref_ligand = ref_ligand[ref_ligand.element != "H"]
docked_ligand = docked_ligand[docked_ligand.element != "H"]


# Visualization with PyMOL
# sphinx_gallery_thumbnail_number = 2
pymol_receptor = pymol_interface.PyMOLObject.from_structure(receptor)
pymol_ref_ligand = pymol_interface.PyMOLObject.from_structure(ref_ligand)
pymol_docked_ligand = pymol_interface.PyMOLObject.from_structure(docked_ligand)
# Visualize receptor as surface
pymol_receptor.show_as("surface")
pymol_receptor.color("white")
pymol_interface.cmd.set("surface_quality", 2)
# Visualize as stick model
pymol_interface.cmd.set("stick_radius", 0.15)
pymol_interface.cmd.set("sphere_scale", 0.25)
pymol_interface.cmd.set("sphere_quality", 4)
# The reference is a blue 'shadow'
pymol_ref_ligand.show("spheres")
pymol_ref_ligand.color("blue")
pymol_ref_ligand.set("stick_color", "blue")
pymol_ref_ligand.set("sphere_transparency", 0.6)
pymol_ref_ligand.set_bond("stick_transparency", 0.6)
# Visualize docked model
pymol_docked_ligand.show("spheres")
pymol_docked_ligand.color("gray", docked_ligand.element == "C")
pymol_docked_ligand.set("stick_color", "grey80")
# Adjust camera
pymol_docked_ligand.orient()
pymol_interface.cmd.rotate("y", 180)
pymol_interface.cmd.rotate("x", -15)
pymol_docked_ligand.zoom(buffer=-1)
pymol_interface.cmd.set("depth_cue", 0)
pymol_interface.cmd.clip("slab", 100)
# Display
pymol_interface.show((1500, 600))
