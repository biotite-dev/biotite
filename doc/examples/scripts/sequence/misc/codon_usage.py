"""
Calculation of codon usage
==========================

This script creates a table for *codon usage* in the *Escherichia coli*
K-12 strain.
Codon usage describes the frequencies of the codons that code for an
amino acid.
These frequencies are expected to reflect the abundance of the
respective tRNAs in the target organism.
The codon usage differs from species to species.
Hence, it is important to look at the codon usage of the organism of
interest for applications like *codon optimization* etc.

For the computation of the codon usage we will have a look into the
annotated *Escherichia coli* K-12 genome:
The script extracts all coding sequences (CDS) from the genome and
counts the total number of each codon in these sequences.
Then the relative frequencies are calculated by dividing the total
number of occurrences of each codon by the total number of occurrences
of the respective amino acid.
In order to improve the performance, the script mostly works with symbol
codes (see tutorial) instead of the symbols itself.

At first we fetch the genome from the NCBI Entrez database
(Accession: U00096) as GenBank file and parse it into an
:class:`AnnotatedSequence`.
Then we create a dictionary that will store the total codon frequencies
later on.
As already mentioned, the script works with symbol codes.
Consequently, each codon in the dictionary
is described as 3 integers instead of 3 letters.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import tempfile
import itertools
import numpy as np
import biotite.sequence as seq
import biotite.sequence.io.genbank as gb
import biotite.sequence.io.fasta as fasta
import biotite.database.entrez as entrez


# Get the E. coli K-12 genome as annotated sequence
gb_file = gb.GenBankFile.read(
    entrez.fetch("U00096", tempfile.gettempdir(), "gb", "nuccore", "gb")
)
# We are only interested in CDS features
k12_genome = gb.get_annotated_sequence(gb_file, include_only=["CDS"])


# This dictionary will count how often each codon occurs in the genome
# For increased performance the dictionary uses symbol codes ([0 3 2])
# instead of symbols (['A' 'T' 'G']) as keys
codon_counter = {
    codon: 0 for codon
    in itertools.product( *([range(len(k12_genome.sequence.alphabet))] * 3) )
}
# For demonstration purposes print the 64 codons in symbol code form
print(list(codon_counter.keys()))

########################################################################
# As expected the dictionary encodes each codon as tuple of 3 numbers,
# where ``0`` represents ``A``, ``1`` ``C``, ``2`` ``G`` and ``3`` ``T``.
# These mappings are defined by the alphabet of the genomic sequence.
#
# In the next step the occurrences of codons in the coding sequences
# are counted, the relative frequencies are calculated and the codon
# table is printed.

# Iterate over all CDS features
for cds in k12_genome.annotation:
    # Obtain the sequence of the CDS
    cds_seq = k12_genome[cds]
    if len(cds_seq) % 3 != 0:
        # A CDS' length should be a multiple of 3,
        # otherwise the CDS is malformed
        continue
    # Iterate over the sequence in non-overlapping frames of 3
    # and count the occurence of each codon
    for i in range(0, len(cds_seq), 3):
        codon_code = tuple(cds_seq.code[i:i+3])
        codon_counter[codon_code] += 1

# Convert the total frequencies into relative frequencies
# for each amino acid
# The NCBI codon table with ID 11 is the bacterial codon table
table = seq.CodonTable.load(11)
# As the script uses symbol codes, each amino acid is represented by a
# number between 0 and 19, instead of the single letter code
for amino_acid_code in range(20):
    # Find all codons coding for the amino acid
    # The codons are also in symbol code format, e.g. ATG -> (0, 3, 2)
    codon_codes_for_aa = table[amino_acid_code]
    # Get the total amount of codon occurrences for the amino acid
    total = 0
    for codon_code in codon_codes_for_aa:
        total += codon_counter[codon_code]
    # Replace the total frequencies with relative frequencies
    # and print it
    for codon_code in codon_codes_for_aa:
        # Convert total frequencies into relative frequencies
        codon_counter[codon_code] /= total
        # The rest of this code block prints the codon usage table
        # in human readable format
        amino_acid = seq.ProteinSequence.alphabet.decode(amino_acid_code)
        # Convert the symbol codes for each codon into symbols...
        # ([0,3,2] -> ['A' 'T' 'G'])
        codon = k12_genome.sequence.alphabet.decode_multiple(codon_code)
        # ...and represent as string
        # (['A' 'T' 'G'] -> "ATG")
        codon = "".join(codon)
        freq = codon_counter[codon_code]
        print(f"{amino_acid}   {codon}   {freq:.2f}")
    print()

########################################################################
# The codon usage table can be used to optimize recombinant protein
# expression by designing the DNA sequence for the target protein
# according to the codon usage.
# This is called *codon optimization*.
# However, there is a variety of different algorithms for codon
# optimization.
# For simplicity reasons, this example uses an approach, where for every
# amino acid always the most frequently occuring codon is used.
#
# In the follwing we will derive a codon optimized DNA sequence of
# streptavidin for expression in *E. coli* K-12.


# For each amino acid, select the codon with maximum codon usage
# Again, symbol codes are used here
opt_codons = {}
for amino_acid_code in range(20):
    codon_codes_for_aa = table[amino_acid_code]
    # Find codon with maximum frequency
    max_freq = 0
    best_codon_code = None
    for codon_code in codon_codes_for_aa:
        if codon_counter[codon_code] > max_freq:
            max_freq = codon_counter[codon_code]
            best_codon_code = codon_code
    # Map the amino acid to the codon with maximum frequency
    opt_codons[amino_acid_code] = best_codon_code

# Fetch the streptavidin protein sequence from Streptomyces avidinii
fasta_file = fasta.FastaFile.read(
    entrez.fetch("P22629", None, "fasta", "protein", "fasta")
)
strep_prot_seq = fasta.get_sequence(fasta_file)
# Create a DNA sequence from the protein sequence
# using the optimal codons
strep_dna_seq = seq.NucleotideSequence()
strep_dna_seq.code = np.concatenate(
    [opt_codons[amino_acid_code] for amino_acid_code in strep_prot_seq.code]
)
# Add stop codon
strep_dna_seq += seq.NucleotideSequence("TAA")
# Put the DNA sequence into a FASTA file
fasta_file = fasta.FastaFile()
fasta_file["Codon optimized streptavidin"] = str(strep_dna_seq)
# Print the contents of the created FASTA file
print(fasta_file)
# In a real application it would be written onto the hard drive via
# fasta_file.write("some_file.fasta")