"""
Sequence logo of the Anderson promoter collection
=================================================

This script creates a sequence logo for the
`Anderson promoter collection <http://parts.igem.org/Promoters/Catalog/Anderson>`_.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics

# The list of Anderson promoters
seqs = [seq.NucleotideSequence("ttgacagctagctcagtcctaggtataatgctagc"),
        seq.NucleotideSequence("ttgacagctagctcagtcctaggtataatgctagc"),
        seq.NucleotideSequence("tttacagctagctcagtcctaggtattatgctagc"),
        seq.NucleotideSequence("ttgacagctagctcagtcctaggtactgtgctagc"),
        seq.NucleotideSequence("ctgatagctagctcagtcctagggattatgctagc"),
        seq.NucleotideSequence("ttgacagctagctcagtcctaggtattgtgctagc"),
        seq.NucleotideSequence("tttacggctagctcagtcctaggtactatgctagc"),
        seq.NucleotideSequence("tttacggctagctcagtcctaggtatagtgctagc"),
        seq.NucleotideSequence("tttacggctagctcagccctaggtattatgctagc"),
        seq.NucleotideSequence("ctgacagctagctcagtcctaggtataatgctagc"),
        seq.NucleotideSequence("tttacagctagctcagtcctagggactgtgctagc"),
        seq.NucleotideSequence("tttacggctagctcagtcctaggtacaatgctagc"),
        seq.NucleotideSequence("ttgacggctagctcagtcctaggtatagtgctagc"),
        seq.NucleotideSequence("ctgatagctagctcagtcctagggattatgctagc"),
        seq.NucleotideSequence("ctgatggctagctcagtcctagggattatgctagc"),
        seq.NucleotideSequence("tttatggctagctcagtcctaggtacaatgctagc"),
        seq.NucleotideSequence("tttatagctagctcagcccttggtacaatgctagc"),
        seq.NucleotideSequence("ttgacagctagctcagtcctagggactatgctagc"),
        seq.NucleotideSequence("ttgacagctagctcagtcctagggattgtgctagc"),
        seq.NucleotideSequence("ttgacggctagctcagtcctaggtattgtgctagc")]
# Sequences do not need to be aligned
# -> Create alignment with trivial trace
# [[0 0 0 ...]
#  [1 1 1 ...]
#  [2 2 2 ...]
#     ...     ]
alignment = align.Alignment(
    sequences = seqs,
    trace     = np.tile(np.arange(len(seqs[0])), len(seqs)) \
                .reshape(len(seqs), len(seqs[0])) \
                .transpose(),
    score     = 0
)
# Create sequence logo from alignment
logo = graphics.SequenceLogo(alignment, 800, 100)
fig = logo.generate()
plt.show()