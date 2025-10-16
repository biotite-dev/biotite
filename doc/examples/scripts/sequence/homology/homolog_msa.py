"""
Homology search and multiple sequence alignment
===============================================

This script searches for proteins homologous to Cas9 from
*Streptococcus pyogenes* via NCBI BLAST and performs a multiple
sequence alignment of the hit sequences afterwards, using MUSCLE.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause
from tempfile import gettempdir
import matplotlib.pyplot as plt
import biotite.application.blast as blast
import biotite.application.muscle as muscle
import biotite.database.entrez as entrez
import biotite.sequence.graphics as graphics
import biotite.sequence.io.fasta as fasta

# Download sequence of Streptococcus pyogenes Cas9
file_name = entrez.fetch("Q99ZW2", gettempdir(), "fa", "protein", "fasta")
fasta_file = fasta.FastaFile.read(file_name)
ref_seq = fasta.get_sequence(fasta_file)
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
    file_name = entrez.fetch(hit, gettempdir(), "fa", "protein", "fasta")
    fasta_file = fasta.FastaFile.read(file_name)
    hit_seqs.append(fasta.get_sequence(fasta_file))

# Perform a multiple sequence alignment using MUSCLE
app = muscle.MuscleApp(hit_seqs)
app.start()
app.join()
alignment = app.get_alignment()
# Print the MSA with hit IDs
print("MSA results:")
gapped_seqs = alignment.get_gapped_sequences()
for i in range(len(gapped_seqs)):
    print(hits[i], " " * 3, gapped_seqs[i])

# Visualize the first 200 columns of the alignment
# Reorder alignments to reflect sequence distance

fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.add_subplot(111)
order = app.get_alignment_order()
graphics.plot_alignment_type_based(
    ax,
    alignment[:200, order.tolist()],
    labels=[hits[i] for i in order],
    show_numbers=True,
)
fig.tight_layout()

plt.show()
