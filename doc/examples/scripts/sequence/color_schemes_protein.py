"""
Biotite color schemes for protein sequences
===========================================

This script shows the same multiple protein sequence alignment
in the different color schemes available in *Biotite*.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import biotite
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import biotite.database.entrez as entrez


# Generate example alignment
# (the same as in the bacterial luciferase example)
query =   entrez.SimpleQuery("luxA", "Gene Name") \
        & entrez.SimpleQuery("srcdb_swiss-prot", "Properties")
uids = entrez.search(query, db_name="protein")
file_name = entrez.fetch_single_file(
    uids, biotite.temp_file("fasta"), db_name="protein", ret_type="fasta"
)
fasta_file = fasta.FastaFile()
fasta_file.read(file_name)
sequences = [seq.ProteinSequence(seq_str) for seq_str in fasta_file.values()]
matrix = align.SubstitutionMatrix.std_protein_matrix()
alignment, order, _, _ = align.align_multiple(sequences, matrix)
# Order alignment according to the guide tree
alignment = alignment[:, order]
alignment = alignment[220:300]

# Get color scheme names
alphabet = seq.ProteinSequence.alphabet
names = sorted(graphics.list_color_scheme_names(alphabet))
count = len(names)

# Visualize each scheme using the example alignment
fig = plt.figure(figsize=(8.0, count*2.0))
for i, name in enumerate(names):
    for j, color_symbols in enumerate([False, True]):
        ax = fig.add_subplot(len(names), 2, 2*i + j + 1)
        if j == 0:
            alignment_part = alignment[:40]
            ax.set_ylabel(name)
        else:
            alignment_part = alignment[40:]
        graphics.plot_alignment_type_based(
            ax, alignment_part, symbols_per_line=len(alignment_part),
            color_scheme=name, color_symbols=color_symbols, symbol_size=8
        )
fig.subplots_adjust(hspace=0)
fig.tight_layout()
plt.show()