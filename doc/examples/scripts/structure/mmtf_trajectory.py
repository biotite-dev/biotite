r"""
MMTF as trajectory format
=========================

This example demonstrates how the MMTF format can be used as an
alterntive to classical trajecotry formats (TRR, XTC, etc.).

For this purpose a trajectory file obtained from a MD simulation
(Gromacs) of the miniprotein TC5b (PDB: 1L2Y) was loaded
(1001 frames, 304 atoms), and the coordinates are saved as the default
``xCoordList``, ``yCoordList`` and ``zCoordList`` in an MMTF file.

The trajectory file can be downloaded
:download:`here </static/assets/download/1l2y_md.xtc>`.

Using the MMTF format for macromolecular trajectories takes advantage
of the precise and open specification of the format and the wide support
by a multitude of software.
This comes at cost of a slightly higher file size compared to the XTC
format, the limited size of each field (max. 2\ :sup:`32`\ -1 bytes)
and the violation of the format itself
(most mandatory fields are omitted).
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import biotite
import biotite.structure as struc
import biotite.structure.io.xtc as xtc
import biotite.structure.io.mmtf as mmtf
import numpy as np
import matplotlib.pyplot as plt
import os.path

# Put here the path of the downloaded trajectory file
xtc_file_path = "../../../static/assets/download/1l2y_md.xtc"
mmtf_file_path = biotite.temp_file("1l2y_md.mmtf")

xtc_file = xtc.XTCFile()
xtc_file.read(xtc_file_path)
coord = xtc_file.get_coord()
coord_x = coord[:,:,0].flatten()
coord_y = coord[:,:,1].flatten()
coord_z = coord[:,:,2].flatten()
mmtf_file = mmtf.MMTFFile()
# The usual codec and param for MMTF's coordinate encoding
mmtf_file.set_array("xCoordList", coord_x, codec=10, param=1000)
mmtf_file.set_array("yCoordList", coord_y, codec=10, param=1000)
mmtf_file.set_array("zCoordList", coord_z, codec=10, param=1000)
mmtf_file.write(mmtf_file_path)
xtc_size = os.path.getsize(xtc_file_path)
mmtf_size = os.path.getsize(mmtf_file_path)

figure = plt.figure()
ax = figure.add_subplot(111)
ax.bar([1,2], [xtc_size/1000, mmtf_size/1000], width=0.3,
       color=[biotite.colors["green"], biotite.colors["orange"]], linewidth=0)
ax.set_xticks([1,2])
ax.set_xticklabels(["XTC", "MMTF"])
ax.set_xlim(0.5,2.5)
ax.set_ylim(0,3000)
ax.set_ylabel("File size (kB)")
plt.show()