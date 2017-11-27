# This script searches for proteins homologous to Cas9 from
# Streptococcus pyogenes via NCBI BLAST and performs a multiple
# sequence alignment of the hit sequences afterwards, using MUSCLE.

import biotite
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import biotite.application.muscle as muscle
import biotite.application.blast as blast
import biotite.database.entrez as entrez

# Download sequence of Streptococcus pyogenes Cas9
file_name = entrez.fetch("Q99ZW2", biotite.temp_dir(), "fa", "protein", "fasta")
file = fasta.FastaFile()
file.read(file_name)
ref_seq = fasta.get_sequence(file)
# Find homologous proteins using NCBI Blast
# Search only the UniProt/SwissProt database
blast_app = blast.BlastWebApp("blastp", ref_seq, "swissprot")
blast_app.start()
blast_app.join()
alignments = blast_app.get_alignments()
# Get hid IDs for hits with score > 200
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
muscle_app = muscle.MuscleApp(hit_seqs)
muscle_app.start()
muscle_app.join()
ali = muscle_app.get_alignment()
# Print the MSA with hit IDs
print("MSA results:")
gapped_seqs = ali.get_gapped_sequences()
for i in range(len(gapped_seqs)):
    print(hits[i], " "*3, gapped_seqs[i])