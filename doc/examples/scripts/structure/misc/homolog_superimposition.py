"""
Superimposition of homologous protein structures
================================================

This script superimposes the structure of a streptavidin monomer into an avidin monomer.

- *Green*:  Avidin
- *Orange*: Strepatvidin
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import biotite.database.rcsb as rcsb
import biotite.interface.pymol as pymol_interface
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx


def _extract_monomer(complex):
    complex = complex[struc.filter_amino_acids(complex)]
    # Get the monomer that belongs to the first atom in the structure
    return complex[struc.get_chain_masks(complex, [0])[0]]


avidin_file = pdbx.BinaryCIFFile.read(rcsb.fetch("1vyo", "bcif"))
avidin = _extract_monomer(pdbx.get_structure(avidin_file, model=1, include_bonds=True))
streptavidin_file = pdbx.BinaryCIFFile.read(rcsb.fetch("6j6j", "bcif"))
streptavidin = _extract_monomer(
    pdbx.get_structure(streptavidin_file, model=1, include_bonds=True)
)

streptavidin, _, _, _ = struc.superimpose_homologs(avidin, streptavidin)

# Visualization with PyMOL
pymol_avidin = pymol_interface.PyMOLObject.from_structure(avidin)
pymol_streptavidin = pymol_interface.PyMOLObject.from_structure(streptavidin)
pymol_avidin.color("biotite_lightgreen")
pymol_streptavidin.color("biotite_lightorange")
pymol_avidin.show_as("cartoon")
pymol_streptavidin.show_as("cartoon")
pymol_avidin.orient()
pymol_interface.show((1500, 1000))
