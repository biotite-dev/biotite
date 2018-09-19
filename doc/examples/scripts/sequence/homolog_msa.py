"""
Multiple sequence alignment of Cas9 homologs
============================================

This script searches for proteins homologous to Cas9 from
*Streptococcus pyogenes* via NCBI BLAST and performs a multiple
sequence alignment of the hit sequences afterwards, using MUSCLE.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import biotite
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import biotite.sequence.graphics as graphics
import biotite.application.muscle as muscle
import biotite.application.blast as blast
import biotite.database.entrez as entrez
import matplotlib.pyplot as plt

# Download sequence of Streptococcus pyogenes Cas9
file_name = entrez.fetch("Q99ZW2", biotite.temp_dir(), "fa", "protein", "fasta")
file = fasta.FastaFile()
file.read(file_name)
ref_seq = fasta.get_sequence(file)
# Find homologous proteins using NCBI Blast
# Search only the UniProt/SwissProt database
blast_app = blast.BlastWebApp("blastp", ref_seq, "swissprot", obey_rules=False)
blast_app.start()
blast_app.join()
alignments = blast_app.get_alignments()
# Get hit IDs for hits with score > 200
hits = []
for ali in alignments:
    if ali.score > 200:
        hits.append(ali.hit_id)
# Get the sequences from hit IDs
hit_seqs = []
for hit in hits:
    file_name = entrez.fetch(hit, biotite.temp_dir(), "fa", "protein", "fasta")
    file = fasta.FastaFile()
    file.read(file_name)
    hit_seqs.append(fasta.get_sequence(file))

# Perform a multiple sequence alignment using MUSCLE
app = muscle.MuscleApp(hit_seqs)
app.start()
app.join()
alignment = app.get_alignment()
# Print the MSA with hit IDs
print("MSA results:")
gapped_seqs = alignment.get_gapped_sequences()
for i in range(len(gapped_seqs)):
    print(hits[i], " "*3, gapped_seqs[i])

# Visualize the first 200 columns of the alignment
# using AlignmentSymbolVisualizer
# Reorder alignments to reflect sequence distance
order = app.get_alignment_order()
vis = graphics.AlignmentSymbolVisualizer(alignment[:200, order.tolist()])
vis.add_labels(labels=[hits[i] for i in order])
vis.add_location_numbers()
vis.set_alignment_properties(symbols_per_line=40)
# Use Clustal X color scheme
vis.set_color_scheme("clustalx")
figure = vis.generate()
plt.show()