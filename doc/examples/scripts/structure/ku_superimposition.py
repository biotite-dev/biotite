"""
Superimposition of two protein structures
=========================================

This script superimposes the atom coordinates of the Ku70/80
heterodimer onto the corresponding structure bound to DNA
(superimposition of 1JEQ onto 1JEY).
The visualisation was coducted with PyMOL.

* *Orange*: Ku dimer originally bound to DNA
* *Green*:  Free Ku dimer
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause


from tempfile import NamedTemporaryFile
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb
import numpy as np


ku_dna_file = NamedTemporaryFile(suffix=".cif")
ku_file     = NamedTemporaryFile(suffix=".cif")
# The output file names
# Modify these values for actual file output
ku_dna_file_name = ku_dna_file.name
ku_file_name = ku_file.name

# Download and parse structure files
ku_dna = pdbx.get_structure(pdbx.PDBxFile.read(rcsb.fetch("1JEY", "cif")))[0]
ku     = pdbx.get_structure(pdbx.PDBxFile.read(rcsb.fetch("1JEQ", "cif")))[0]
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
ku_superimposed, transformation = struc.superimpose(
    ku_dna_common, ku_common, (ku_common.atom_name == "CA")
)
# We do not want the cropped structures
# -> apply superimposition on original structures
ku_superimposed = struc.superimpose_apply(ku, transformation)
# Write PDBx files as input for PyMOL
cif_file = pdbx.PDBxFile()
pdbx.set_structure(cif_file, ku_dna, data_block="ku_dna")
cif_file.write(ku_dna_file_name)
cif_file = pdbx.PDBxFile()
pdbx.set_structure(cif_file, ku_superimposed, data_block="ku")
cif_file.write(ku_file_name)
# Visualization with PyMOL...

ku_dna_file.close()
ku_file.close()