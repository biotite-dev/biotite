"""
Visualization of normal modes from an elastic network model
===========================================================

The elastic network model is a fast method to estimate movements in a
protein structure, without the need to run time-consuming MD
simulations.

The eigenvectors file can be downloaded
:download:`here </examples/download/lysozyme_md.xtc>`.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

from os.path import join
import numpy as np
from numpy import newaxis
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb


# A CSV file containing the eigenvectors for the CA atoms
VECTOR_FILE = "../../download/glycosylase_anm_vectors.csv"
# The corresponding structure
PDB_ID = "1MUG"
# The normal mode to be visualized
# '-1' is the last (and most significant) one
MODE = -1
# The amount of frames (models) per oscillation
FRAMES = 60
# The maximum oscillation amplitude for an atom
# (The length of the ANM's eigenvectors make only sense when compared
# relative to each other, the absolute values have no significance)
MAX_AMPLITUDE = 5


# Load structure
mmtf_file = mmtf.MMTFFile()
mmtf_file.read(rcsb.fetch(PDB_ID, "mmtf"))
structure = mmtf.get_structure(mmtf_file, model=1)


# Filter first peptide chain
protein_chain = structure[
    struc.filter_amino_acids(structure)
    & (structure.chain_id == structure.chain_id[0])
]
# Filter CA atoms
ca = protein_chain[protein_chain.atom_name == "CA"]


# Load eigenvectors for CA atoms
# The first axis indicates the normal mode,
# the second axis indicates the vector component
vectors = np.loadtxt(VECTOR_FILE, delimiter=",").transpose()
# Discard the last 6 modes, as these are movements of the entire system:
# A system with N atoms has only 3N - 6 degrees of freedom
#                                   ^^^
vectors = vectors[:-6]
# Extract vectors for given mode and reshape to (n,3) array
mode_vectors = vectors[MODE].reshape((-1, 3))
# Rescale, so that the largest vector has the length 'MAX_AMPLITUDE'
vector_lenghts = np.sqrt(np.sum(mode_vectors**2, axis=-1))
scale = MAX_AMPLITUDE / np.max(vector_lenghts)
mode_vectors *= scale 


# Stepwise application of eigenvectors as smooth sine oscillation
time = np.linspace(0, 2*np.pi, FRAMES, endpoint=False)
deviation = np.sin(time)[:, newaxis, newaxis] * mode_vectors

# Apply oscillation of CA atom to all atoms in the corresponding residue
oscillation = np.zeros((FRAMES, len(protein_chain), 3))
residue_starts = struc.get_residue_starts(
    protein_chain,
    # The last array element will be the length of the atom array,
    # i.e. no valid index
    add_exclusive_stop=True
)
for i in range(len(residue_starts) -1):
    res_start = residue_starts[i]
    res_stop = residue_starts[i+1]
    oscillation[:, res_start:res_stop, :] \
        = protein_chain.coord[res_start:res_stop, :] + deviation[:, i:i+1, :]

# An atom array stack containing all frames
oscillating_structure = struc.from_template(protein_chain, oscillation)
# Save as PDB for rendering a video with PyMOL
#strucio.save_structure("glycosylase_oscillation.pdb", oscillating_structure)

# biotite_static_image = glycosylase_oscillation.png