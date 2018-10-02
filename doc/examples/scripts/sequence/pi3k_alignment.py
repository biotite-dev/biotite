"""
Comparison of human PI3K family
===============================
"""

import numpy as np
import matplotlib.pyplot as plt
import biotite
import biotite.database.entrez as entrez
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import biotite.sequence.io.fasta as fasta
import biotite.application.clustalo as clustalo

uids  = ["5JHB_A", "5LUQ_A",   "5FLC_B", "5YZ0_A", "5NP0_A", "4FUL_A"]
names = ["PI3K",   "DNA-PKcs", "mTOR",   "ATR",    "ATM",    "hSMG-1"]

sequences = []
file_name = entrez.fetch_single_file(
    uids, biotite.temp_file("fasta"), db_name="protein", ret_type="fasta"
)

file = fasta.FastaFile()
file.read(file_name)
for header, seq_str in file:
    sequences.append(seq.ProteinSequence(seq_str))

alignment = clustalo.ClustalOmegaApp.align(sequences)

# Find start position and exclusive stop position of 'PI3K' sequence
# (index 0)
trace = alignment.trace
# From beginning of the sequence...
for i in range(len(trace)):
    # Check if all sequences have no gap at the given position
    if trace[i,0] != -1:
        start_index = i
        break
# ...and the end of the sequence
for i in range(len(trace)-1, -1, -1):
    # Check if all sequences have no gap at the given position
    if trace[i,0] != -1:
        stop_index = i+1
        break

# Truncate alignment to region where the 'PI3K' sequence exists
alignment.trace = alignment.trace[start_index:stop_index]

matrix = align.SubstitutionMatrix.std_protein_matrix()
visualizer = graphics.AlignmentSimilarityVisualizer(alignment, matrix)
# The alignment is quite long
# -> Reduce font and box size to reduce figure size
visualizer.set_alignment_properties(
    box_size=(8,14), symbols_per_line=80, font_size=6
)
visualizer.add_labels(names, font_size=10, size=80)
visualizer.add_location_numbers(size=40, font_size=10)
visualizer.set_color(color=biotite.colors["orange"])
visualizer.set_margin(5.0)
figure = visualizer.generate()

plt.show()