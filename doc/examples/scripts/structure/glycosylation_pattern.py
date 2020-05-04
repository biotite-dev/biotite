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
    glycosyl_graphs = [graph.subgraph(nodes).copy()
                       for nodes in nx.connected_components(graph)
                       if len(nodes) > 1]
    
    for glycosyl_graph in glycosyl_graphs:
        node_colors = ["blue" if res_id in amino_acid_res_ids else "red"
                    for res_id in glycosyl_graph.nodes()]
        root = [
            res_id for res_id in glycosyl_graph.nodes()
            if res_id in amino_acid_res_ids
        ][0]
        pos = graphviz_layout(glycosyl_graph, prog="dot")
        
        nodes = list(pos.keys())
        pos_array = np.array(list(pos.values()))
        pos_array -= pos_array[0]
        pos_array[:,1] *= -1
        pos_array[:,1] /= pos_array[1,1] - pos_array[0,1]
        WIDTH = 10
        pos_array[:,0] *= WIDTH / np.min(pos_array[:,0])
        pos_array[:,0] += root
        pos = {node: tuple(coord) for node, coord in zip(nodes, pos_array)}
        
        nx.draw(
            glycosyl_graph, pos, ax=ax,
            node_size=10, linewidths=0.5, node_color=node_colors
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
###
#structure = structure[structure.hetero]
###
plot_graph(ax, structure)
fig.tight_layout()

plt.show()