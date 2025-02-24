r"""
Visualization of normal modes from an elastic network model
===========================================================

The *elastic network model* (ENM) is a fast method to estimate movements
in a protein structure, without the need to run time-consuming MD
simulations.
A protein is modelled as *mass-and-spring* model, with the masses being
the :math:`C_\alpha` atoms and the springs being the non-covalent bonds
between adjacent residues.
Via *normal mode analysis* distinct movements/oscillations can be
extracted from the model.

An *anisotropic network model* (ANM), is an ENM that includes
directional information.
Hence, the normal mode analysis yields eigenvectors, where each atom is
represented by three vector components (*x*, *y*, *z*).
Thus these vectors can be used for 3D representation.

In the case of this example a normal mode analysis on an ANM was already
conducted.
This script merely takes the structure and obtained eigenvectors
to add a smooth oscillation of the chosen normal mode to the structure.
The newly created structure has multiple models, where each model
depicts a different time in the oscillation period.
Then the multi-model structure can be used to create a video of the
oscillation using a molecular visualization program.

The file containing the eigenvectors can be downloaded via this
:download:`link </examples/download/glycosylase_anm_vectors.csv>`.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
from numpy import newaxis
import biotite.database.rcsb as rcsb
import biotite.interface.pymol as pymol_interface
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

# A CSV file containing the eigenvectors for the CA atoms
VECTOR_FILE = "../../../download/glycosylase_anm_vectors.csv"
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
pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch(PDB_ID, "bcif"))
structure = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)


# Filter first peptide chain
protein_chain = structure[
    struc.filter_amino_acids(structure) & (structure.chain_id == structure.chain_id[0])
]
# Filter CA atoms
ca = protein_chain[protein_chain.atom_name == "CA"]


# Load eigenvectors for CA atoms
# The first axis indicates the mode,
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
time = np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)
deviation = np.sin(time)[:, newaxis, newaxis] * mode_vectors

# Apply oscillation of CA atom to all atoms in the corresponding residue
oscillation = np.zeros((FRAMES, len(protein_chain), 3))
residue_starts = struc.get_residue_starts(
    protein_chain,
    # The last array element will be the length of the atom array,
    # i.e. no valid index
    add_exclusive_stop=True,
)
for i in range(len(residue_starts) - 1):
    res_start = residue_starts[i]
    res_stop = residue_starts[i + 1]
    oscillation[:, res_start:res_stop, :] = (
        protein_chain.coord[res_start:res_stop, :] + deviation[:, i : i + 1, :]
    )

# An atom array stack containing all frames
oscillating_structure = struc.from_template(protein_chain, oscillation)

# Visualization with PyMOL
pymol_object = pymol_interface.PyMOLObject.from_structure(oscillating_structure)
pymol_object.color("biotite_lightgreen", oscillating_structure.chain_id == "A")
# Set custom view
pymol_interface.cmd.set_view(
    (
        0.605540633,
        0.363677770,
        -0.707855821,
        -0.416691631,
        0.902691007,
        0.107316799,
        0.678002179,
        0.229972601,
        0.698157668,
        0.000000000,
        0.000000000,
        -115.912551880,
        32.098876953,
        31.005725861,
        78.377349854,
        89.280677795,
        142.544403076,
        -20.000000000,
    )
)
# Prepare output video frames
pymol_interface.cmd.mset()
pymol_interface.play((600, 600))
