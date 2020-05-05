r"""
Visualization of glycosylation patterns
=======================================

"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb


SACCHARIDES = {
    "GLA": ("o", "gold",            "Gal"),    # alpha
    "GAL": ("o", "gold",            "Gal"),    # beta
    "NGA": ("s", "gold",            "GalNAc"),
    "X6X": ("P", "gold",            "GalN"),
    "AGC": ("o", "royalblue",       "Glc"),    # alpha
    "BGC": ("o", "royalblue",       "Glc"),    # beta
    "NAG": ("s", "royalblue",       "GlcNAc"),
    "GCS": ("P", "royalblue",       "GlcN"),
    "MAN": ("o", "forestgreen",     "Man"),    # alpha
    "BMA": ("o", "forestgreen",     "Man"),    # beta
    "BM3": ("s", "forestgreen",     "ManNAc"),
    "95Z": ("P", "forestgreen",     "ManN"),
    "XYS": ("*", "darkorange",      "Xyl"),    # alpha
    "XYP": ("*", "darkorange",      "Xyl"),    # beta
    "XYZ": ("*", "darkorange",      "Xyl"),    # beta (furanose)
    "SI3": ("D", "mediumvioletred", "Neu5Ac"),
    "NGC": ("D", "turquoise",       "Neu5Gc"),
    "KDN": ("D", "forestgreen",     "Kdn"),
    "FUC": ("^", "crimson",         "Fuc"),    # alpha
    "FUL": ("^", "crimson",         "Fuc"),    # beta
    "GCU": (6,   "royalblue",       "GlcA"),   # alpha
    "BDP": (6,   "royalblue",       "GlcA"),   # beta
    "IDR": (7,   "chocolate",       "IdoA"),
    "ADA": (8,   "gold",            "GalA"),   # alpha
    "GTR": (8,   "gold",            "GalA"),   # beta
    "MAV": (9,   "forestgreen",     "ManA"),   # alpha
    "BEM": (9,   "forestgreen",     "ManA"),   # beta
}


def plot_graph(ax, structure):
    if struc.get_chain_count(structure) != 1:
        raise struc.BadStructureError(
            "A structure with a single chain is required"
        )
    
    graph = nx.Graph()

    for res_id in np.unique(structure.res_id):
        connected_res_ids = set()
        # Iterate over the index of each atom in this residue
        for i in np.where(structure.res_id == res_id)[0]:
            # Get indices for each atom connected to this atom
            connected, _ = structure.bonds.get_bonds(i)
            for j in connected:
                connected_res_id = structure.res_id[j]
                # Omit bonds to other atoms in the same residue
                if connected_res_id != res_id:
                    graph.add_edge(res_id, connected_res_id)
    
    # A dictionary that maps the 3-letter abbreviation
    # to full residue names
    full_res_names = {name: info.full_name(name) for name
                      in np.unique(structure.res_name)}
    
    amino_acid_res_ids = np.unique(structure.res_id[~structure.hetero])
    
    # Remove edges between amino acids
    for res_id in amino_acid_res_ids:
        # Put into list, as the number of
        # neighbors is reduced during iteration
        for connected_res_id in list(graph.neighbors(res_id)):
            if connected_res_id in amino_acid_res_ids:
                graph.remove_edge(res_id, connected_res_id)
    
    # Get connected subgraphs containing a glycosylation
    # -> any subgraph with more than one node
    glycan_graphs = [graph.subgraph(nodes).copy()
                     for nodes in nx.connected_components(graph)
                     if len(nodes) > 1]
    
    for glycan_graph in glycan_graphs:
        # Convert into a directed graph for correct plot layout
        glycan_graph = nx.DiGraph(
            [(min(res_id_1, res_id_2), max(res_id_1, res_id_2))
             for res_id_1, res_id_2 in glycan_graph.edges()]
        )
        
        root = [
            res_id for res_id in glycan_graph.nodes()
            if res_id in amino_acid_res_ids
        ][0]

        pos = graphviz_layout(glycan_graph, prog="dot")
        
        nodes = list(pos.keys())
        pos_array = np.array(list(pos.values()))
        pos_array -= pos_array[0]
        pos_array[:,1] /= pos_array[nodes.index(list(glycan_graph.neighbors(root))[0]), 1] - pos_array[nodes.index(root), 1]
        WIDTH = 5
        non_zero = pos_array[(pos_array[:,0] != 0), 0]
        if len(non_zero) != 0:
            pos_array[:,0] *= WIDTH / np.min(non_zero)
        pos_array[:,0] += root
        pos = {node: tuple(coord) for node, coord in zip(nodes, pos_array)}
        
        NODE_SIZE = 30
        LINE_WIDTH = 0.5
        nx.draw_networkx_edges(
            glycan_graph, pos, ax=ax,
            arrows=False, node_size=0, width=LINE_WIDTH
        )
        for res_name, (shape, color, name) in GLYCAN_COMPOUNDS.items():
            included_res_ids = np.unique(
                structure.res_id[structure.res_name == res_name]
            )
            node_list = [res_id for res_id in glycan_graph.nodes()
                         if res_id in included_res_ids]
            nx.draw_networkx_nodes(
                glycan_graph, pos, ax=ax, nodelist=node_list,
                node_size=NODE_SIZE, node_shape=shape, node_color=color,
                edgecolors="black", linewidths=LINE_WIDTH
            )
        ax.axis("on")
        plt.tick_params(
            axis='both',
            which='both',
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelbottom=True,
            labelleft=True
        )
    
    _, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)


file_name = rcsb.fetch("3HDL", "mmtf", ".")
#file_name = rcsb.fetch("2K33", "mmtf", ".")
mmtf_file = mmtf.MMTFFile.read(file_name)
structure = mmtf.get_structure(mmtf_file, model=1, include_bonds=True)
structure = structure[structure.chain_id == "A"]


fig, ax = plt.subplots(figsize=(8.0, 1.5))
plot_graph(ax, structure)
fig.tight_layout()

plt.show()