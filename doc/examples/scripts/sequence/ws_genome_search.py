r"""
Finding homologs of short sequences in a genome
===============================================

In this example we use an local alignment (dynamic programming) for
identification of homologs of a short sequence in an entire bacterial
genome.
Specifically we take the *leuL* gene (*leu* operon leader peptide)
from *E. coli* BL21 and search for it in the genome of a
*Salmonella enterica* strain.

This method only works for short sequences, since the dynmic programming
method requires too much RAM (dozens of gigabyte) for longer sequences.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import biotite
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import biotite.sequence.io.genbank as gb
import biotite.sequence.graphics as graphics
import biotite.sequence.align as align
import biotite.database.entrez as entrez
import numpy as np
import matplotlib.pyplot as plt

# Download E. coli BL21 genome
file_name = entrez.fetch("CP001509", biotite.temp_dir(), "gb", "nuccore", "gb")
gb_file = gb.GenBankFile()
gb_file.read(file_name)
annot_seq = gb_file.get_annotated_sequence(include_only=["gene"])
# Find leuL gene
for feature in annot_seq.annotation:
    if "gene" in feature.qual and feature.qual["gene"] == "leuL":
        leul_feature = feature
# Get leuL sequence
leul_seq = annot_seq[leul_feature]

# Download Salmonella enterica genome without annotations
file_name = entrez.fetch("CP019649", biotite.temp_dir(),
                         "fa", "nuccore", "fasta")
fasta_file = fasta.FastaFile()
fasta_file.read(file_name)
se_genome = fasta.get_sequence(fasta_file)
# Find leuL in genome by local alignment
matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
# Use general gap penalty to save RAM
alignments = align.align_optimal(
    leul_seq, se_genome, matrix, gap_penalty=-7, local=True
)
# Do the same for reverse complement genome
se_genome_rev = se_genome.reverse().complement()
rev_alignments = align.align_optimal(
    leul_seq, se_genome_rev, matrix, gap_penalty=-7, local=True
)

########################################################################
# Now that we have both alignments (forward and reverse strand),
# we can can check which of them has a higher score.
# We simply take the score of the first element in each list, since due
# to the nature of the dynamic programming algorithm, every element in
# a list has the same score.

print("Forward:")
print("Alignment count:", len(alignments))
print("Score:", alignments[0].score)
print()
print("Reverse:")
print("Alignment count:", len(rev_alignments))
print("Score:", rev_alignments[0].score)

########################################################################
# Clearly the the alignment with the reverse genome seems to be the
# right one.
# For visualization puposes we have to apply a renumbering function
# for the genomic sequence,
# since the original indices refer to the reverse complement sequence,
# but we want the numbers to refer to the original one.

# Use first and only alignment 
alignment = rev_alignments[0]
# Reverse sequence numbering for second sequence (genome) in alignment
number_funcs = [None,   lambda x: len(alignment.sequences[1]) - x]
# Visualize alignment, use custom color
vis = graphics.AlignmentSimilarityVisualizer(alignment, matrix)
vis.set_color(color="orange")
vis.add_labels(labels=["E. coli (leuL)", "S. enterica"])
vis.set_alignment_properties(symbols_per_line=30)
# Apply new numbering scheme
vis.add_location_numbers(size=100, number_functions=number_funcs)
figure = vis.generate()

########################################################################
# We will now go even further and align the translated protein
# sequences.

leul_ec = leul_seq
# Obtain the S enterica leuL sequence
# using the first and last index in the alignment trace
first_i = alignment.trace[0, 1]
last_i  = alignment.trace[-1,1]
# Get sequence slice (mind the exclusive stop)
leul_se = se_genome_rev[first_i : last_i+1]
# Translate sequences into protein sequence using bacterial codon table
codon_table = seq.CodonTable.load("Bacterial, Archaeal and Plant Plastid")
leul_ec = leul_ec.translate(complete=True, codon_table=codon_table)
leul_se = leul_se.translate(complete=True, codon_table=codon_table)

# Align the protein sequences (using BLOSUM62 and affine gap penalty)
matrix = align.SubstitutionMatrix.std_protein_matrix()
alignments = align.align_optimal(
    leul_ec, leul_se, matrix, gap_penalty=(-10,-1)
)
alignment = alignments[0]

# Lets try a matplotlib colormap this time
vis = graphics.AlignmentSimilarityVisualizer(alignment, matrix)
vis.set_color(cmap="summer_r")
vis.add_labels(labels=["E. coli", "S. enterica"])
vis.set_alignment_properties(symbols_per_line=30)
figure = vis.generate()
plt.show()