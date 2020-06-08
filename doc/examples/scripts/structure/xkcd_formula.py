import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx


SEPARATION = 0.01
NODE_SIZE = 20
COLORS = {
    "N": "royalblue",
    "O": "firebrick",
    "S": "gold",
}


def split_bond(i_pos, j_pos, separation, number):
    if number == 1:
        return [(i_pos, j_pos)]
    
    diff = np.asarray(j_pos) - np.asarray(i_pos)
    orth = [[0,-1], [1,0]] @ diff
    sep = orth / np.linalg.norm(orth) * separation
    return [
        (i_pos + p*sep , j_pos + p*sep )
        for p in np.linspace(-number/2, number/2, number)
    ]


def get_residue(components_file, res_name):
    BOND_ORDERS = {
        "SING" : struc.BondType.SINGLE,
        "DOUB" : struc.BondType.DOUBLE,
        "TRIP" : struc.BondType.TRIPLE,
        "QUAD" : struc.BondType.QUADRUPLE
    }

    cif_atoms = components_file.get_category("chem_comp_atom", block=res_name)
    cif_bonds = components_file.get_category("chem_comp_bond", block=res_name)

    array = struc.AtomArray(len(list(cif_atoms.values())[0]))

    array.res_name = cif_atoms["comp_id"]
    array.atom_name = cif_atoms["atom_id"]
    array.element = cif_atoms["type_symbol"]
    array.charge = cif_atoms["charge"]
    array.hetero[:] = True
    
    array.coord[:,0] = cif_atoms["pdbx_model_Cartn_x_ideal"]
    array.coord[:,1] = cif_atoms["pdbx_model_Cartn_y_ideal"]
    array.coord[:,2] = cif_atoms["pdbx_model_Cartn_z_ideal"]
        
    bonds = struc.BondList(array.array_length())
    if cif_bonds is not None:
        for atom1, atom2, order, aromatic_flag in zip(
            cif_bonds["atom_id_1"], cif_bonds["atom_id_2"],
            cif_bonds["value_order"], cif_bonds["pdbx_aromatic_flag"]
        ):
            atom_i = np.where(array.atom_name == atom1)[0][0]
            atom_j = np.where(array.atom_name == atom2)[0][0]
            bond_type = BOND_ORDERS[order]
            bonds.add_bond(atom_i, atom_j, bond_type)
    array.bonds = bonds

    return array


components_file = pdbx.PDBxFile.read("/home/kunzmann/downloads/components.cif")
molecule = get_residue(components_file, "CFF")
#molecule = molecule[molecule.element != "H"]

graph = nx.Graph()
graph.add_edges_from(
    [(i, j, {"order": int(order)})
     for i, j, order in molecule.bonds.as_array()]
)
pos = nx.kamada_kawai_layout(graph)


with plt.xkcd(scale=1, length=100, randomness=2):
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("KEEPS THE WORLD GOING...", fontsize=30)
    
    for i, j in graph.edges():
        i_pos = pos[i]
        j_pos = pos[j]
        order = graph.edges[i,j]["order"]
        for i_pos, j_pos in split_bond(i_pos, j_pos, SEPARATION, order):
            x_data = (i_pos[0], j_pos[0])
            y_data = (i_pos[1], j_pos[1])
            ax.plot(x_data, y_data, color="black")
    
    for i in graph.nodes():
        symbol = molecule.element[i]
        if symbol != "C":
            x = pos[i][0]
            y = pos[i][1]
            ax.plot(
                [x], [y],
                marker="o", markersize=NODE_SIZE, color="white"
            )
            ax.text(
                x, y, symbol,
                fontsize=NODE_SIZE, ha="center", va="center",
                color=COLORS.get(symbol, "black")
            )
    
    fig.tight_layout()

plt.show()