"""
Identification of lipid bilayer leaflets
========================================

This script implements the *LeafletFinder* algorithm [1]_ used by
*MDAnalysis*. The algorithm detects which lipid molecules belong to the
same membrane leaflet, i.e. the same side of a lipid bilayer,
irrespective of the shape of the bilayer.

At first the algorithm creates an adjacency matrix of all lipid head
groups, where the cutoff distance is smaller than the minimum distance
between a head group of one leaflet to a head group of another leaflet.
A graph is created from the matrix.
Each leaflet is a connected subgraph.

.. [1] N Michaud-Agrawal, EJ Denning, TB Woolf and O Beckstein,
       "MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics
       Simulations."
       J Comput Chem, 32, 2319–2327 (2011).
"""

import numpy as np
import networkx as nx
import biotite.structure as struc
import biotite.structure.io as strucio

# The bilayer structure file can be downloaded from
# http://www.charmm-gui.org/archive/pure_bilayer/dppc.tar.gz
PDB_FILE_PATH = "/home/kunzmann/Documents/coding/biotite/doc/examples/download/dppc_n128.pdb"


def find_leaflets(structure, head_atom_mask,
                  cutoff_distance=15.0, periodic=False):
    """
    Identify which lipids molecules belong to the same lipid bilayer
    leaflet.

    Parameters
    ----------
    structure : AtomArray, shape=(n,)
        The structure containing the membrane.
        May also include other molecules, e.g. water or an embedded
        protein.
    head_atom_mask : ndarray, dtype=bool, shape=(n,)
        A boolean mask that selects atoms from `structure` that
        represent lipid head groups.
    cutoff_distance : float, optional
        When the distance of two head groups is larger than this value,
        they are not (directly) connected in the same leaflet.
    periodic : bool, optional,
        If true, periodic boundary conditions are considered.
        This requires that `structure` has an associated `box`.
    """
    
    cell_list = struc.CellList(
        structure, cell_size=cutoff_distance, selection=head_atom_mask,
        periodic=periodic
    )
    adjacency_matrix = cell_list.create_adjacency_matrix(cutoff_distance)
    graph = nx.Graph(adjacency_matrix)

    head_leaflets = [sorted(c) for c in nx.connected_components(graph)
                     # A leaflet cannot consist of a single lipid
                     # This also removes all entries
                     # for atoms not in 'head_atom_mask'
                     if len(c) > 1]
    
    # 'leaflets' contains indices to head atoms 
    # Broadcast each head atom index to all atoms in its corresponding
    # residue
    leaflets = []
    for head_leaflet in head_leaflets:
        leaflet_mask = np.zeros(structure.array_length(), dtype=bool)
        for index in head_leaflet:
            leaflet_mask[
                (structure.chain_id == structure.chain_id[index]) &
                (structure.res_id == structure.res_id[index])
            ] = True
        leaflets.append(leaflet_mask)
    return np.array(leaflets)


structure = strucio.load_structure(PDB_FILE_PATH)
# We cannot go over periodic boundaries in this case,
# because the input PDB does not define a box -> periodic=False
# However, as we have a planer lipid bilayer,
# periodicity should not matter
leaflets = find_leaflets(
    structure,
    head_atom_mask=(structure.res_name == "DPP") & (structure.atom_name == "P")
)
# Bilayer -> Expect two leaflets
assert len(leaflets) == 2
# Mark leaflets using different chain IDs
for chain_id, leaflet_mask in zip(("A", "B"), leaflets):
    structure.chain_id[leaflet_mask] = chain_id

# Save marked lipids to structure file for visulaization with PyMOL
strucio.save_structure("leaflets.pdb", structure)
# biotite_static_image = leaflet.png