# This script performs a pairwise sequence alignment of
# avidin (Gallus gallus) with streptavidin (Streptomyces lavendulae).

import biotite
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.io.fasta as fasta
import biotite.database.entrez as entrez

# Download and parse protein sequences of avidin and streptavidin
file_name = entrez.fetch_single_file(["CAC34569", "ACL82594"],
                                     biotite.temp_file("sequences.fasta"),
                                     "protein", "fasta")
file = fasta.FastaFile()
file.read(file_name)
for name, sequence in file:
    if "CAC34569" in name:
        avidin_seq = seq.ProteinSequence(sequence)
    elif "ACL82594" in name:
        streptavidin_seq = seq.ProteinSequence(sequence)
# Get BLOSUM62 matrix
matrix = align.SubstitutionMatrix.std_protein_matrix()
# Perform pairwise sequence alignment with affine gap penalty
# Terminal gaps are not penalized
alignments = align.align_optimal(avidin_seq, streptavidin_seq, matrix,
                                 gap_penalty=(-10, -1), terminal_penalty=False)
# Output first and only alignment
print(alignments[0])