r"""
Visualization of glycosylation patterns
=======================================

"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb


NODE_SIZE = 50
HORIZONTAL_NODE_DISTANCE = 3
LINE_WIDTH = 0.5


# Adapted from "Mol*" Software
# The dictionary maps residue names of saccharides to their common names
SACCHARIDE_NAMES = {
    res_name : common_name for common_name, res_names in [
        ("Glc", ["GLC", "BGC", "Z8T", "TRE", "MLR"]),
        ("Man", ["MAN", "BMA"]),
        ("Gal", ["GLA", "GAL", "GZL", "GXL", "GIV"]),
        ("Gul", ["4GL", "GL0", "GUP", "Z8H"]),
        ("Alt", ["Z6H", "3MK", "SHD"]),
        ("All", ["AFD", "ALL", "WOO", "Z2D"]),
        ("Tal", ["ZEE", "A5C"]),
        ("Ido", ["ZCD", "Z0F", "4N2"]),
        ("GlcNAc", ["NDG", "NAG", "NGZ"]),
        ("ManNAc", ["BM3", "BM7"]),
        ("GalNAc", ["A2G", "NGA", "YYQ"]),
        ("GulNAc", ["LXB"]),
        ("AllNAc", ["NAA"]),
        ("IdoNAc", ["LXZ"]),
        ("GlcN", ["PA1", "GCS"]),
        ("ManN", ["95Z"]),
        ("GalN", ["X6X", "1GN"]),
        ("GlcA", ["GCU", "BDP"]),
        ("ManA", ["MAV", "BEM"]),
        ("GalA", ["ADA", "GTR", "GTK"]),
        ("GulA", ["LGU"]),
        ("TalA", ["X1X", "X0X"]),
        ("IdoA", ["IDR"]),
        ("Qui", ["G6D", "YYK"]),
        ("Rha", ["RAM", "RM4", "XXR"]),
        ("6dGul", ["66O"]),
        ("Fuc", ["FUC", "FUL", "FCA", "FCB"]),
        ("QuiNAc", ["Z9W"]),
        ("FucNAc", ["49T"]),
        ("Oli", ["DDA", "RAE", "Z5J"]),
        ("Tyv", ["TYV"]),
        ("Abe", ["ABE"]),
        ("Par", ["PZU"]),
        ("Dig", ["Z3U"]),
        ("Ara", ["64K", "ARA", "ARB", "AHR", "FUB", "BXY", "BXX"]),
        ("Lyx", ["LDY", "Z4W"]),
        ("Xyl", ["XYS", "XYP", "XYZ", "HSY", "LXC"]),
        ("Rib", ["YYM", "RIP", "RIB", "BDR", "0MK", "Z6J", "32O"]),
        ("Kdn", ["KDM", "KDN"]),
        ("Neu5Ac", ["SIA", "SLB"]),
        ("Neu5Gc", ["NGC", "NGE"]),
        ("LDManHep", ["GMH"]),
        ("Kdo", ["KDO"]),
        ("DDManHep", ["289"]),
        ("MurNAc", ["MUB", "AMU"]),
        ("Mur", ["1S4", "MUR"]),
        ("Api", ["XXM"]),
        ("Fru", ["BDF", "Z9N", "FRU", "LFR"]),
        ("Tag", ["T6T"]),
        ("Sor", ["SOE"]),
        ("Psi", ["PSV", "SF6", "SF9"]),
    ]
    for res_name in res_names
}


# Colors and shapes were adapted from the 'Carbohydrate Structure Database'
# http://csdb.glycoscience.ru/database/index.html?help=eog
SACCHARIDE_REPRESENTATION = {
    "Glc": ("o", "royalblue"),
    "Man": ("o", "forestgreen"),
    "Gal": ("o", "gold"),
    "Gul": ("o", "darkorange"),
    "Alt": ("o", "pink"),
    "All": ("o", "purple"),
    "Tal": ("o", "lightsteelblue"),
    "Ido": ("o", "chocolate"),
    
    "GlcNAc": ("s", "royalblue"),
    "ManNAc": ("s", "forestgreen"),
    "GalNAc": ("s", "gold"),
    "GulNAc": ("s", "darkorange"),
    "AllNAc": ("s", "purple"),
    "IdoNAc": ("s", "chocolate"),
    
    "GlcN": ("1", "royalblue"),
    "ManN": ("1", "forestgreen"),
    "GalN": ("1", "gold"),
    
    "GlcA": ("v", "royalblue"),
    "ManA": ("v", "forestgreen"),
    "GalA": ("v", "gold"),
    "GulA": ("v", "darkorange"),
    "TalA": ("v", "lightsteelblue"),
    "IdoA": ("v", "chocolate"),
    
    "Qui": ("^", "royalblue"),
    "Rha": ("^", "forestgreen"),
    "6dGul": ("^", "darkorange"),
    "Fuc": ("^", "crimson"),
    
    "QuiNAc": ("P", "royalblue"),
    "FucNAc": ("P", "crimson"),
    
    "Oli": ("X", "royalblue"),
    "Tyv": ("X", "forestgreen"),
    "Abe": ("X", "darkorange"),
    "Par": ("X", "pink"),
    "Dig": ("X", "purple"),
    
    "Ara": ("*", "forestgreen"),
    "Lyx": ("*", "gold"),
    "Xyl": ("*", "darkorange"),
    "Rib": ("*", "pink"),
    
    "Kdn": ("D", "forestgreen"),
    "Neu5Ac": ("D", "mediumvioletred"),
    "Neu5Gc": ("D", "turquoise"),
    
    "LDManHep": ("H", "forestgreen"),
    "Kdo": ("H", "gold"),
    "DDManHep": ("H", "pink"),
    "MurNAc": ("H", "purple"),
    "Mur": ("H", "chocolate"),
    
    "Api": ("p", "royalblue"),
    "Fru": ("p", "forestgreen"),
    "Tag": ("p", "gold"),
    "Sor": ("p", "darkorange"),
    "Psi": ("p", "pink"),
    
    # Default representation
    None: ("h", "black")
}


def plot_glycans(ax, structure):
    if struc.get_chain_count(structure) != 1:
        raise struc.BadStructureError(
            "A structure with a single chain is required"
        )
    
    # These are required later
    amino_acid_res_ids = np.unique(structure.res_id[~structure.hetero])
    ids_to_names = {res_id : res_name for res_id, res_name
                    in zip(structure.res_id, structure.res_name)}
    

    # Create a graph that depicts which residues are connected
    # Use residue IDs as nodes
    graph = nx.Graph()
    # Convert BondList to array and omit bond order
    bonds = structure.bonds.as_array()[:, :2]
    connected = structure.res_id[bonds.flatten()].reshape(bonds.shape)
    # Omit bonds with the same residue
    connected = connected[connected[:,0] != connected[:,1]]
    graph.add_edges_from(connected)
    
    # Remove edges between amino acids
    for res_id in amino_acid_res_ids:
        # Put into list, as the number of
        # neighbors is reduced during iteration
        for connected_res_id in list(graph.neighbors(res_id)):
            if connected_res_id in amino_acid_res_ids:
                graph.remove_edge(res_id, connected_res_id)
    
    # Get connected subgraphs containing glycans
    # -> any subgraph with more than one node
    glycan_graphs = [graph.subgraph(nodes).copy()
                     for nodes in nx.connected_components(graph)
                     if len(nodes) > 1]
    

    # Plot each glycan graph individually
    # Save the ID and symbol of each glycosylated amino acid
    # for ticks on x-axis
    glycosylated_residue_ids = []
    glycosylated_residue_symbols = []
    # Use node markers for the legend,
    # use dictionary to avoid redundant entries
    legend_elements = {}
    for glycan_graph in glycan_graphs:
        # Convert into a directed graph for correct plot layout
        glycan_graph = nx.DiGraph(
            [(min(res_id_1, res_id_2), max(res_id_1, res_id_2))
             for res_id_1, res_id_2 in glycan_graph.edges()]
        )
        
        # The 'root' is the glycosylated amino acid
        root = [
            res_id for res_id in glycan_graph.nodes()
            if res_id in amino_acid_res_ids
        ]
        # No attachment to amino acid -> Ignore glycan
        if len(root) == 0:
            continue
        else:
            root = root[0]
        glycosylated_residue_ids.append(root)
        glycosylated_residue_symbols.append(
            seq.ProteinSequence.convert_letter_3to1(ids_to_names[root])
        )

        pos = graphviz_layout(glycan_graph, prog="dot")
        
        nodes = list(pos.keys())
        # Convert dictionary to array
        pos_array = np.array(list(pos.values()))
        # Position the root at coordinate origin
        pos_array -= pos_array[nodes.index(root)]
        # Set vertical distances between nodes to 1
        root_neighbor = nodes.index(list(glycan_graph.neighbors(root))[0])
        pos_array[:,1] /= (
            pos_array[root_neighbor, 1] - pos_array[nodes.index(root), 1]
        )
        # Set horizontal distances between nearest nodes to 1
        non_zero_dist = np.abs(pos_array[(pos_array[:,0] != 0), 0])
        if len(non_zero_dist) != 0:
            pos_array[:,0] *= HORIZONTAL_NODE_DISTANCE / np.min(non_zero_dist)
        # Move graph to residue ID position on x-axis
        pos_array[:,0] += root
        # Convert array back to dictionary
        pos = {node: tuple(coord) for node, coord in zip(nodes, pos_array)}
        
        nx.draw_networkx_edges(
            glycan_graph, pos, ax=ax,
            arrows=False, node_size=0, width=LINE_WIDTH
        )
        
        for res_id in glycan_graph.nodes():
            # Only plot glycans, not amino acids
            if res_id in amino_acid_res_ids:
                continue
            res_name = ids_to_names[res_id]
            common_name = SACCHARIDE_NAMES.get(res_name)
            shape, color = SACCHARIDE_REPRESENTATION[common_name]
            nx.draw_networkx_nodes(
                glycan_graph, pos, ax=ax, nodelist=[res_id],
                node_size=NODE_SIZE, node_shape=shape, node_color=color,
                edgecolors="black", linewidths=LINE_WIDTH
            )
            legend_elements[common_name] = Line2D(
                [0], [0], label=common_name, linestyle="None",
                marker=shape, markerfacecolor=color,
                markeredgecolor="black", markeredgewidth=LINE_WIDTH
            )
    

    ax.legend(handles=legend_elements.values(), bbox_to_anchor=(1, 1))
    # Show the bottom x-axis with glycosylated residue positions
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="x", bottom=False, labelbottom=True)
    ax.set_xticks(glycosylated_residue_ids)
    ax.set_xticklabels(
        [symbol + str(res_id) for symbol, res_id
         in zip(glycosylated_residue_symbols, glycosylated_residue_ids)]
    )
    _, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)


file_name = rcsb.fetch("4CUO", "mmtf", ".")
mmtf_file = mmtf.MMTFFile.read(file_name)
structure = mmtf.get_structure(mmtf_file, model=1, include_bonds=True)
structure = structure[structure.chain_id == "A"]

fig, ax = plt.subplots(figsize=(8.0, 1.5))
plot_glycans(ax, structure)
fig.tight_layout()

plt.show()