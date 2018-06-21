r"""
Basic analysis of MD simulation
===============================

In this example, we will analyze a trajectory of a Gromacs MD
simulation: The trajectory contains simulation data of the miniprotein
TC5b (PDB: 1L2Y) over 1 ns. Water and ions have already been removed
from the trajectory file. As template we use a PDB file, that represents
the state at the start of the simulation (including water and ions).

The trajectory file can be downloaded
:download:`here </static/assets/download/1l2y_md.xtc>`
and the template PDB can be downloaded
:download:`here </static/assets/download/1l2y_md_start.pdb>`.

We begin by loading the template PDB file as `AtomArray`, sanitizing it
and using it to load the trajectory as `AtomArrayStack`.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import matplotlib.pyplot as plt
import re

# Put here the path of the downloaded files
templ_file_path = "../../../static/assets/download/1l2y_md_start.pdb"
traj_file_path  = "../../../static/assets/download/1l2y_md.xtc"

template = strucio.load_structure(templ_file_path)
# In contrast to the trajectory, the template still has water and ions,
# that need to be removed
template = template[(template.res_name != "CL") & (template.res_name != "SOL")]
# Gromacs does not set the element symbol in its PDB files
# Therefore we simply determine the symbol
# from the first character in the atom name
# Since hydrogens may have leading numbers we simply ignore numbers
for i in range(template.array_length()):
    template.element[i] = re.sub(r"\d", "", template.atom_name[i])[0]
trajectory = strucio.load_structure(traj_file_path, template=template)

########################################################################
# At first we want to see if the simulation converged.
# For this purpose we take the RMSD of a frame compared to the starting
# structure as measure. In order to calculate the RMSD we must
# superimpose all models onto a reference, in this case we choose the
# starting structure. 

trajectory, transform = struc.superimpose(template, trajectory)
rmsd = struc.rmsd(template, trajectory)
# Simulation was 1000 ps long
time = np.linspace(0, 1000, len(trajectory))

figure = plt.figure()
ax = figure.add_subplot(111)
ax.plot(time, rmsd, color=biotite.colors["dimorange"])
ax.set_xlim(0,1000)
ax.set_xlabel("time (ps)")
ax.set_ylabel("RMSD (Angstrom)")

########################################################################
# As we can see the simulation seems to converge already in the
# beginning of the simulation. After a few ps the RMSD stays in a range
# of approx. 2 - 3 Angstrom. However it seems like there are two kinds
# of quasi-dicrete states as the two plateaus suggest. For further
# investigation we would require more simulation time. 
# 
# In order to better evaluate the unfolding of our miniprotein in the
# course of simulation, we calculate and plot the radius of gyration
# (a measure for the protein radius).

radius = struc.gyration_radius(trajectory)

figure = plt.figure()
ax = figure.add_subplot(111)
ax.plot(time, radius, color=biotite.colors["dimorange"])
ax.set_xlim(0,1000)
ax.set_xlabel("time (ps)")
ax.set_ylabel("Radius of gyration (Angstrom)")

########################################################################
# From this perspective, the protein seems really stable.
# The radius does merely fluctuate in a range of approx. 0.5 Angstrom
# during the entire simulation.
# 
# Let's have a look at single amino acids:
# Which residues fluctuate most?
# For answering this question we calculate the RMSF
# (Root mean square fluctuation). It is similar to the RMSD, but instead
# of averaging over the atoms and looking at each time step, we
# average over the time and look at each residue. Usually the average
# model is taken as reference (compared to the starting model for RMSD).
# 
# Since side chain atoms fluctuate quite a lot, they are not suitable
# for evaluation of the residue flexibility. Therefore, we consider only
# CA atoms.

# In all models, mask the CA atoms
ca_trajectory = trajectory[:, trajectory.atom_name == "CA"]
rmsf = struc.rmsf(struc.average(ca_trajectory), ca_trajectory)

figure = plt.figure()
ax = figure.add_subplot(111)
ax.plot(np.arange(1, 21), rmsf, color=biotite.colors["dimorange"])
ax.set_xlim(1, 20)
ax.set_xlabel("Residue")
ax.set_ylabel("RMSF (Angstrom)")
ax.set_xticks(np.arange(1, 21))
ax.set_xticklabels(np.arange(1, 21))

plt.show()