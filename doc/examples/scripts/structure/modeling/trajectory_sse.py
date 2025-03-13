r"""
Secondary structure during an MD simulation
===========================================

This script displays positional changes of secondary structure elements
(SSE) in the course of time of an MD simulation.

The trajectory file can be downloaded
:download:`here </examples/download/lysozyme_md.xtc>`
and the template PDB can be downloaded
:download:`here </examples/download/lysozyme_md.pdb>`.
"""

# Code source: Daniel Bauer, Patrick Kunzmann
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D
import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.xtc as xtc

# Put here the path of the downloaded files
templ_file_path = "../../../download/lysozyme_md.pdb"
traj_file_path = "../../../download/lysozyme_md.xtc"


xtc_file = xtc.XTCFile.read(traj_file_path)
traj = xtc_file.get_structure(template=strucio.load_structure(templ_file_path))
time = xtc_file.get_time()
traj = traj[:, struc.filter_amino_acids(traj)]

sse = np.array([struc.annotate_sse(frame) for frame in traj])


# Matplotlib needs numbers to assign colors correctly
def sse_to_num(sse):
    num = np.empty(sse.shape, dtype=float)
    num[sse == "a"] = 0
    num[sse == "b"] = 1
    num[sse == "c"] = np.nan
    return num


sse = sse_to_num(sse)


# Plotting
# SSE colormap
color_assign = {
    r"$\alpha$-helix": biotite.colors["dimgreen"],
    r"$\beta$-sheet": biotite.colors["lightorange"],
}
cmap = colors.ListedColormap(color_assign.values())

plt.figure(figsize=(8.0, 6.0))
plt.imshow(sse.T, cmap=cmap, origin="lower")
plt.xlabel("Time / ps")
plt.ylabel("Residue")
ticks = np.arange(0, len(traj), 10)
plt.xticks(ticks, time[ticks].astype(int))

# Custom legend below the SSE plot
custom_lines = [Line2D([0], [0], color=cmap(i), lw=4) for i in range(len(color_assign))]
plt.legend(
    custom_lines,
    color_assign.keys(),
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=len(color_assign),
    fontsize=8,
)
plt.tight_layout()
plt.show()
