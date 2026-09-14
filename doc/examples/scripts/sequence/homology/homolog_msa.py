"""
Homology search and multiple sequence alignment
===============================================

This script searches for proteins homologous to Cas9 from
*Streptococcus pyogenes* in the *UniProtKB/Swiss-Prot* database via
*MMseqs2* and performs a multiple sequence alignment of the hit
sequences afterwards, using MUSCLE.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause
from tempfile import gettempdir
import matplotlib.pyplot as plt
import biotite.application_v2.mmseqs as mmseqs
import biotite.application_v2.muscle as muscle
import biotite.database.entrez as entrez
import biotite.sequence.graphics as graphics
import biotite.sequence.io.fasta as fasta

# Download sequence of Streptococcus pyogenes Cas9
file_name = entrez.fetch("Q99ZW2", gettempdir(), "fa", "protein", "fasta")
fasta_file = fasta.FastaFile.read(file_name)
ref_seq = fasta.get_sequence(fasta_file)

# Find homologous proteins using MMseqs2
# Search only the UniProt/SwissProt database
app = mmseqs.MMseqsApp()
swissprot_db = app.databases("UniProtKB/Swiss-Prot").result()
query_db = mmseqs.create_database_from_sequences(app, {"query": ref_seq})
# Backtracing is required to obtain the alignments later
alignment_db = app.search(query_db, swissprot_db, a=True).result()
table = mmseqs.get_alignment_table(app, alignment_db, ["target", "bits"])
# Get hit IDs for hits with bit score > 200
hits = [
    target for target, bits in zip(table["target"], table["bits"]) if float(bits) > 200
]
alignments = mmseqs.get_alignments(app, alignment_db)
hit_seqs = [alignments["query", hit].sequences[1] for hit in hits]

# Perform a multiple sequence alignment using MUSCLE
msa_result = muscle.Muscle3App().run(hit_seqs).result()
alignment = msa_result.alignment
# Print the MSA with hit IDs
print("MSA results:")
gapped_seqs = alignment.get_gapped_sequences()
for i in range(len(gapped_seqs)):
    print(hits[i], " " * 3, gapped_seqs[i])

# Visualize the first 200 columns of the alignment
# Reorder alignments to reflect sequence distance

fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.add_subplot(111)
order = msa_result.order
graphics.plot_alignment_type_based(
    ax,
    alignment[:200, order.tolist()],
    labels=[hits[i] for i in order],
    show_numbers=True,
)
fig.tight_layout()

plt.show()
