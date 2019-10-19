"""
Homology of G-protein coupled receptors
=======================================

This example plots an unrooted phylogenetic tree depicting the evolution
of different G-protein coupled receptors (GPCRs).

The Uniprot/NCBI Entrez IDs and gene names of the GPCRs are obtained
from `<https://www.uniprot.org/docs/7tmrlist.txt>`_.
The corresponding sequences are downloaded and aligned.
Based on the pairwise sequence identity in the multiple sequence
alignment a tree is created via the *neighbor-joining* method.
Finally the unrooted tree is plotted using the graph drawing
capabilities of the *NetworkX* package.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
from matplotlib.text import Text
import matplotlib.pyplot as plt
import networkx as nx
import requests
import biotite
import biotite.sequence as seq
import biotite.sequence.graphics as graphics
import biotite.sequence.align as align
import biotite.sequence.phylo as phylo
import biotite.sequence.io.fasta as fasta
import biotite.database.entrez as entrez
import biotite.application.clustalo as clustalo


# The bovine GPCRs are investigated
SPECIES = "Bovine"


response = requests.get("https://www.uniprot.org/docs/7tmrlist.txt")
lines = [line.strip() for line in response.text.split("\n")]
lines = lines[23:-34]
genes = []
ids = []
for line in lines:
    gene_start = line.find("[")
    gene_end   = line.find("]")
    # Filter title and empty lines
    # -> these have no square brackets indicating a gene name
    if gene_start != -1 and gene_end != -1:
         # Only the genes from the chosen organism are selected
         if SPECIES in line:
            # Uniprot/NCBI ID in second column, surrounded by brackets
            ncbi_id = line.split()[1].replace("(","").replace(")","")
            # Gene is surrounded by square brackets
            gene = line[gene_start : gene_end+1] \
                   .replace("[","").replace("]","")
            # Sometimes alternative gene names are separated via a
            # semicolon -> Choose the first gene name
            gene = gene.split(";")[0].strip()
            genes.append(gene)
            ids.append(ncbi_id)

fasta_file = fasta.FastaFile()
# Download sequences a file-like object and read the sequences from it
fasta_file.read(entrez.fetch_single_file(
    ids, file_name=None, db_name="protein", ret_type="fasta"
))
sequences = [seq.ProteinSequence(seq_str) for seq_str in fasta_file.values()]
# Create multiple sequence alignment with Clustal Omega
alignment = clustalo.ClustalOmegaApp.align(sequences)


# The distance measure required for the tree calculation is the
# percentage of non-identical amino acids in the respective two
# sequences
distances = 1 - align.get_pairwise_sequence_identity(
    alignment, mode="shortest"
)
# Create tree via neighbor joining
tree = phylo.neighbor_joining(distances)

# Convert the Biotite 'Tree' object into a NetworkX 'Graph' object
# via a recursive function
# Since NetworkX has no dedicated class for nodes, but accepts any
# immutable Python object, the reference index of the node is used for
# this
def convert_node(graph, tree_node):
    """
    Add a tree node to a graph.

    Parameters
    ----------
    graph : Graph
        The graph where the node should be added.
    tree_node : Node
        The node to be added.
    
    Returns
    -------
    node_id : int or tuple
        The identifier of the added node in the graph.
        If the node is a leaf node, this is the reference index,
        otherwise this is a tuple, containing the identifier of the
        child nodes.
    distance : float
        The distance of the given node to its parent node.
    """
    if tree_node.is_leaf():
        return tree_node.index, tree_node.distance
    else:
        child_ids = []
        child_distances = []
        for child_node in tree_node.children:
            id, dist = convert_node(graph, child_node)
            child_ids.append(id)
            child_distances.append(dist)
        this_id = tuple(child_ids)
        for id, dist in zip(child_ids, child_distances):
            # Add connection to children as edge in the graph
            # Distance is added as attribute
            graph.add_edge(this_id, id, distance=dist)
        return this_id, tree_node.distance

graph = nx.Graph()
convert_node(graph, tree.root)

fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.gca()
ax.axis("off")
# Calculate position of nodes in the plot
pos = nx.kamada_kawai_layout(graph)
# Assign the gene names to the nodes that represent a reference index
node_labels = {i: name for i, name in enumerate(genes)}
nx.draw_networkx(
    graph, pos, ax=ax, labels=node_labels, node_color="white", font_size=8,
    # Draw a white background behind the labeled nodes
    node_size=[300 if isinstance(node, int) else 0 for node in graph]
)
fig.tight_layout()

plt.show()