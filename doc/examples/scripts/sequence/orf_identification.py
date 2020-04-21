r"""
Identification of potential open reading frames
===============================================

This example script searches for potential open reading frames (ORFs) in
the Porcine circovirus genome.

At first we will download and read the Porcine circovirus genome.
For translation we will use the default codon table (eukaryotes),
since domestic pigs are the host of the virus.

Since we want to perform a six-frame translation we have to look at
the complementary strand of the genome as well.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import biotite
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import biotite.database.entrez as entrez
import matplotlib.pyplot as plt

# Download Porcine circovirus genome
file_name = entrez.fetch("KP282147", biotite.temp_dir(),
                         "fa", "nuccore", "fasta")
fasta_file = fasta.FastaFile.read(file_name)
genome = fasta.get_sequence(fasta_file)
# Perform translation for forward strand
proteins, positions = genome.translate()
print("Forward strand:")
for i in range(len(proteins)):
    print("{:4d} - {:4d}:   {:}"
          .format(positions[i][0], positions[i][1], str(proteins[i])))
print("\n")
# Perform translation for complementary strand
genome_rev = genome.reverse().complement()
proteins, positions = genome_rev.translate()
print("Reverse strand:")
for i in range(len(proteins)):
    print("{:5d} - {:5d}:   {:}"
          .format(positions[i][0], positions[i][1], str(proteins[i])))