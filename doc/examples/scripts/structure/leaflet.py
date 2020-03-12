import numpy as np
import networkx as nx
import biotite.structure as struc
import biotite.structure.io as strucio


PDB_FILE_PATH = "/home/kunzmann/Documents/coding/biotite/doc/examples/download/dppc_n256.pdb"


def find_leaflets(structure, membrane_res_name, head_atom_name,
                  cutoff_distance=15.0, periodic=False):
    head_atom_mask = (structure.res_name == membrane_res_name) & \
                     (structure.atom_name == head_atom_name)
    
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
                (structure.res_id == structure.res_id[index]) &
                (structure.res_name == membrane_res_name)
            ] = True
        leaflets.append(leaflet_mask)
    return leaflets


structure = strucio.load_structure(PDB_FILE_PATH)
# We cannot go over periodic boundaries in this case,
# because the input PDB does not define a box -> periodic=False
# However, as we have a planer lipid bilayer,
# periodicity should not matter
leaflets = find_leaflets(
    structure, membrane_res_name="DPP", head_atom_name="P", periodic=False
)
# Bilayer -> Expect two leaflets
assert len(leaflets) == 2
# Mark leaflets using different chain IDs
for chain_id, leaflet_mask in zip(("A", "B"), leaflets):
    structure.chain_id[leaflet_mask] = chain_id

strucio.save_structure("leaflets.pdb", structure)
strucio.save_structure("leaflet_heads.pdb", structure[structure.atom_name == "P"])

###
assert (~(leaflets[0] & leaflets[1])).all()
###