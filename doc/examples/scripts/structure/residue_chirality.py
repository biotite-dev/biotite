"""
Determination of amino acid enantiomers
=======================================

This script determines whether a protein structure contains
L- or D-amino acids.
D-amino acids would suggest, that there is something wrong with the
protein structure.
As example the miniprotein TC5b (PDB: 1L2Y) was chosen.

* L = 1
* D = -1
* N/A = 0 (e.g. Glycine)
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

from tempfile import gettempdir
import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb


def get_enantiomer(n, ca, c, cb):
    # Enantiomer is determined by constructing a plane with N, CA and C
    # When CB is inserted, the sign of the resulting scalar describes
    # the enantiomer:
    # L = 1
    # D = -1
    n = np.cross(ca-n, ca-c)
    sign = np.sign(np.dot(cb - ca, n))
    return sign

def analyze_chirality(array):
    # Filter backbone + CB
    array = array[struc.filter_amino_acids(array)]
    array = array[
        (array.atom_name == "CB") | (struc.filter_peptide_backbone(array))
    ]
    # Iterate over each residue
    ids, names = struc.get_residues(array)
    enantiomers = np.zeros(len(ids), dtype=int)
    for i, id in enumerate(ids):
        coord = array.coord[array.res_id == id]
        if len(coord) != 4:
            # Glyine -> no chirality
            enantiomers[i] = 0
        else:
            enantiomers[i] = get_enantiomer(coord[0], coord[1],
                                            coord[2], coord[3])
    return enantiomers

# Fetch and parse structure file
file = rcsb.fetch("1l2y", "mmtf", gettempdir())
stack = strucio.load_structure(file)
# Get first model
array = stack[0]
# Get enantiomers
print("1l2y            ", analyze_chirality(array))
# Reflected structures have opposite enantiomers
# Test via reflection at x-y-plane, z -> -z
array_reflect = array.copy()
array_reflect.coord[:,2] *= -1
print("1l2y (reflected)", analyze_chirality(array_reflect))