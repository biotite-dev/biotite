"""
Quality control of sequencing data
==================================

This script performs quality control on sequence reads similar to the
`FastQC <https://www.bioinformatics.babraham.ac.uk/projects/fastqc/>`_
software.

Inspired by the
`Galaxy tutorial <https://galaxyproject.github.io/training-material/topics/sequence-analysis/tutorials/quality-control/tutorial.html>`_.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause
# sphinx_gallery_thumbnail_number = 2

import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import biotite
import biotite.sequence as seq
import biotite.application.sra as sra


FIG_SIZE = (8.0, 6.0)


sequences_and_scores = sra.FastqDumpApp.fetch("SRR6986517")[0]
sequence_codes = np.stack([
    sequence.code for sequence, _ in sequences_and_scores.values()
])
scores = np.stack([
    scores for _, scores in sequences_and_scores.values()
])
seq_count = scores.shape[0]
seq_length = scores.shape[1]
positions = np.arange(1, seq_length + 1)

########################################################################
#

first_quartile, median, third_quartile = np.quantile(
    scores, (0.25, 0.5, 0.75), axis=0
)
print(first_quartile.shape)
assert (median == np.median(scores, axis=0)).all()

fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.fill_between(
    positions, first_quartile, third_quartile,
    facecolor=biotite.colors["brightorange"],
    label="Lower/upper quartile"
)
ax.plot(
    positions, median,
    color=biotite.colors["dimorange"],
    label="Median"
)
ax.set_xlim(positions[0], positions[-1])
ax.set_xlabel("Position in read")
ax.set_ylabel("Phred score")
ax.legend(loc="lower left")
fig.tight_layout()

########################################################################
#

BIN_NUMBER = 500

fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.hist(
    # Definition range of Sanger Phred scores is 0 to 40
    np.mean(scores, axis=1), bins=np.linspace(0, 40, BIN_NUMBER),
    color=biotite.colors["dimorange"]
)
ax.set_xlabel("Mean Phred score of sequence")
ax.set_ylabel("Sequence count")
fig.tight_layout()

########################################################################
#

# Use ambiguous DNA alphabet,
# as ambiguous bases might occur in some sequencing datasets
alphabet = seq.NucleotideSequence.alphabet_amb

counts = np.stack([
    np.bincount(codes, minlength=len(alphabet))
    for codes in sequence_codes.T
], axis=-1)
frequencies = counts / seq_count * 100

fig, ax = plt.subplots(figsize=FIG_SIZE)
for character, freq in zip(alphabet.get_symbols(), frequencies):
    if (freq > 0).any():
        ax.plot(positions, freq, label=character)
ax.set_xlim(positions[0], positions[-1])
ax.set_xlabel("Position in read")
ax.set_ylabel("Frequency at position (%)")
ax.legend(loc="upper left")
fig.tight_layout()

########################################################################
#

gc_count = np.count_nonzero(
    (sequence_codes == alphabet.encode("G")) |
    (sequence_codes == alphabet.encode("C")),
    axis=1
)
at_count = np.count_nonzero(
    (sequence_codes == alphabet.encode("A")) |
    (sequence_codes == alphabet.encode("T")),
    axis=1
)
gc_content = gc_count / (gc_count + at_count)

# Exclusive range -> 0 to seq_length inclusive
number_of_gc = np.arange(seq_length+1)
exp_gc_content = binom.pmf(
    k=number_of_gc,
    n=seq_length,
    p=np.mean(gc_content)
)

fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.hist(
    # Definition range of Sanger Phred scores is 0 to 40
    gc_content * 100, bins=np.linspace(0, 100, seq_length),
    color=biotite.colors["brightorange"]
)
ax.plot(
    number_of_gc / seq_length * 100,
    exp_gc_content * seq_count,
    color=biotite.colors["dimorange"]
)
ax.set_xlim(0, 100)
ax.set_xlabel("Sequence GC content (%)")
ax.set_ylabel("Sequence count")
fig.tight_layout()

########################################################################
#

duplications = {}
for code in sequence_codes:
    code = tuple(code)
    if code in duplications:
        duplications[code] += 1
    else:
        duplications[code] = 1
duplication_level_count = np.bincount(list(duplications.values()))
duplication_level_freq = (
    duplication_level_count
    * np.arange(len(duplication_level_count))
    / seq_count * 100
)

fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.bar(
    np.arange(0, len(duplication_level_freq)),
    duplication_level_freq,
    width=0.6, 
    color=biotite.colors["dimorange"]
)
ax.set_xlim(0.5, len(duplication_level_freq) + 0.5)
ax.set_ylim(0, 100)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xlabel("Number of duplications")
ax.set_ylabel("Sequence percentage (%)")
fig.tight_layout()

plt.show()