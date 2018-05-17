"""
Superimposition of two protein structures
=========================================

This script superimposes the atom coordinates of the Ku70/80
heterodimer onto the corresponding structure bound to DNA
(superimposition of 1JEQ onto 1JEY).
The visualisation was coducted with PyMOL.

* *Orange*: Ku dimer originally bound to DNA
* *Green*:  Free Ku dimer

.. image:: /static/assets/figures/ku_superimposition.png
"""

import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb
import numpy as np

ku_dna_file = biotite.temp_file("ku_dna.cif")
ku_file = biotite.temp_file("ku.cif")

# Download and parse structure files
file = rcsb.fetch("1JEY", "mmtf", biotite.temp_dir())
ku_dna = strucio.load_structure(file)
file = rcsb.fetch("1JEQ", "mmtf", biotite.temp_dir())
ku = strucio.load_structure(file)
# Remove DNA and water
ku_dna = ku_dna[(ku_dna.chain_id == "A") | (ku_dna.chain_id == "B")]
ku_dna = ku_dna[~struc.filter_solvent(ku_dna)]
ku = ku[~struc.filter_solvent(ku)]
# The structures have a differing amount of atoms missing
# at the the start and end of the structure
# -> Find common structure
ku_dna_common = ku_dna[struc.filter_intersection(ku_dna, ku)]
ku_common = ku[struc.filter_intersection(ku, ku_dna)]
# Superimpose
ku_superimposed, transformation = struc.superimpose(ku_dna_common, ku_common)
# We do not want the cropped structures
# -> apply superimposition on structures before intersection filtering
ku_superimposed = struc.superimpose_apply(ku, transformation)
# Write PDBx files as input for PyMOL
cif_file = pdbx.PDBxFile()
pdbx.set_structure(cif_file, ku_dna, data_block="ku_dna")
cif_file.write(ku_dna_file)
cif_file = pdbx.PDBxFile()
pdbx.set_structure(cif_file, ku_superimposed, data_block="ku")
cif_file.write(ku_file)
# Visualization with PyMOL...