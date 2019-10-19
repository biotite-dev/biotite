"""
Homology of G-protein coupled receptors
=======================================

using the graph drawing capabilities of the *NetworkX* package.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
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


# We are investigating the bovine GPCRs
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
         # We only want genes from the chosen organism
         if SPECIES in line:
            # Uniprot/NCBI ID in second column, surrounded by brackets
            ncbi_id = line.split()[1].replace("(","").replace(")","")
            # Gene is surrounded by square brackets
            gene = line[gene_start : gene_end+1] \
                   .replace("[","").replace("]","")
            # Sometimes alternative gene names are separated via a
            # semicolon -> We only want the first gene name
            gene = gene.split(";")[0].strip()
            genes.append(gene)
            ids.append(ncbi_id)

fasta_file = fasta.FastaFile()
fasta_file.read(entrez.fetch_single_file(
    ids, file_name=None, db_name="protein", ret_type="fasta"
))
sequences = [seq.ProteinSequence(seq_str) for seq_str in fasta_file.values()]
alignment = clustalo.ClustalOmegaApp.align(sequences)



distances = 1 - align.get_pairwise_sequence_identity(alignment, mode="all")
tree = phylo.neighbor_joining(distances)

def convert_node(graph, tree_node):
    if tree_node.is_leaf():
        return tree_node.index, tree_node.distance
    else:
        child_names = []
        child_distances = []
        for child_node in tree_node.children:
            name, dist = convert_node(graph, child_node)
            child_names.append(name)
            child_distances.append(dist)
        this_name = tuple(child_names)
        for name, dist in zip(child_names, child_distances):
            graph.add_edge(this_name, name, distance=dist)
        return this_name, tree_node.distance

graph = nx.Graph()
convert_node(graph, tree.root)

fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.gca()
# Calculate position of nodes
pos = nx.kamada_kawai_layout(graph)
node_labels = {i: name for i, name in enumerate(genes)}
edge_labels = {
    edge: int(attr["distance"] * 100) for edge, attr in graph.edges.items()
}
ax.axis("off")
nx.draw_networkx(
    graph, pos, ax=ax, labels=node_labels, node_color="white", font_size=8,
    node_size=[300 if isinstance(node, int) else 0 for node in graph]
)
fig.tight_layout()

plt.show()