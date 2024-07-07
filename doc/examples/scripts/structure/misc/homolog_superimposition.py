"""
Superimposition of homologous protein structures
================================================

This script superimposes the structure of a streptavidin monomer
into an avidin monomer.
The visualization was conducted with PyMOL.

- *Green*:  Avidin
- *Orange*: Strepatvidin
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import biotite.database.rcsb as rcsb
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
# Visualization with PyMOL...
# sphinx_gallery_ammolite_script = "homolog_superimposition_pymol.py"
