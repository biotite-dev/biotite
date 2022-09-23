"""
Genome comparison between chloroplasts and cyanobacteria
========================================================

.. currentmodule:: biotite.sequence.align

Presumably, plant cells obtained its ability for photosynthesis through
endosymbiosis:
In the past, eukaryotic cells probably have incorporated a
cyanobacterium that evolved to the current chloroplasts in plant cells.
As a side effect, chloroplasts contain its own genome, that has a
high similarity to cyanobacteria.
This example highlights regions in the chloroplast genome that have
been conserved, by comparing the chloroplast genome of
*Arabidopsis thaliana* to the genome of the cyanobacterium
*Synechocystis sp.* PCC 6803.

To compare the genomes this script creates local alignments between both
sequences using a *k-mer* based multi-step process, known from software
like *BLAST* :footcite:`Altschul1990` or *MMseqs*
:footcite:`Steinegger2017`:
Fast *k-mer* matching, ungapped alignment at the hit positions and
a final time-consuming local gapped alignment.
Between each step only promising hits are filtered and used in the next
step.

At first, the genomic sequences are fetched and loaded.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import biotite
import biotite.sequence as seq
import biotite.sequence.io as seqio
import biotite.sequence.io.genbank as gb
import biotite.sequence.align as align
import biotite.database.entrez as entrez
import biotite.application.tantan as tantan


fasta_file = entrez.fetch(
    "NC_000932", tempfile.gettempdir(), "fasta",
    db_name="Nucleotide", ret_type="fasta"
)
chloroplast_seq = seqio.load_sequence(fasta_file)

fasta_file = entrez.fetch(
    "NC_000911", tempfile.gettempdir(), "fasta",
    db_name="Nucleotide", ret_type="fasta"
)
bacterium_seq = seqio.load_sequence(fasta_file)


########################################################################
# For the *k-mer* matching step the genome of the cyanobacterium is
# indexed into a :class:`KmerTable`.
# As homologous regions between both genomes may also appear on the
# complementary strand, both, the original genome sequence and its
# reverse complement, are indexed.
# Two additional techniques are used here:
# First, low complexity regions in the sequence are filtered out using
# *Tantan* :footcite:`Frith2011` to reduce the number of spurious
# homologies.
# Second, spaced *k-mers* :footcite:`Ma2002` are used instead of
# continuous ones to increase the sensitivity.
# However, not every spacing model performs equally well, so the proven
# one ``111∗1∗11∗1∗∗11∗111`` :footcite:`Choi2004` is used here.

repeat_mask = tantan.TantanApp.mask_repeats(bacterium_seq)
bacterium_seqs = [
    bacterium_seq, bacterium_seq.reverse(copy=False).complement()
]

table = align.KmerTable.from_sequences(
    k = 12,
    sequences = bacterium_seqs,
    spacing = "111∗1∗11∗1∗∗11∗111",
    ignore_masks = [repeat_mask, repeat_mask[::-1].copy()]
)

########################################################################
# Now the sequence of the chloroplast genome is matched to the indexed
# sequences.
# Again, low complexity regions are removed.

matches = table.match(
    chloroplast_seq, ignore_mask=tantan.TantanApp.mask_repeats(chloroplast_seq)
)
print("Exemplary matches")
print(matches[:10])
print()
print("Number of hits:", len(matches))

########################################################################
# Each row in the returned array represents one match:
# The first and the third column are the matched positions in the
# chloroplast and bacterial sequence, respectively.
# The second column indicates whether the match in the bacterial
# genome, refers to the original (``0``) or reverse complement (``1``)
# strand.
#
# To reduce the number of alignments, that need to be created from these
# matches, only *double hits* progress into the next step.
# A match becomes a double hit, if there is at least one more match on
# the same diagonal.
# In a more sophisticated scenario, you could also require that the
# second match has a limited distance to the first one.

diagonals = matches[:, 2] - matches[:, 0]

# Store the indices to the match array
# for each combination of diagonal and strand on the bacterial genome
matches_for_diagonals = {}
for i, (diag, strand) in enumerate(zip(diagonals, matches[:,1])):
    if (diag, strand) not in matches_for_diagonals:
        matches_for_diagonals[(diag, strand)] = [i]
    else:
        matches_for_diagonals[(diag, strand)].append(i)

# If a diagonal has more than one match,
# the first match on this diagonal is a double hit
double_hit_indices = [indices[0] for indices
                      in matches_for_diagonals.values() if len(indices) > 1]
double_hits = matches[double_hit_indices]
print("Number of double hits:", len(double_hits))

########################################################################
# The next step is a local ungapped alignment at the positions of the
# double hits using :func:`align_local_ungapped()`:
# For each hit, the alignment is extended into both directions from the
# match until the similarity score drops more than a given threshold
# below the maximum score found.
# Only those hits, where the alignment exceeds a defined threshold
# score, are considered in the next step.
# Therefore, only the similarity score of the alignment is of interest,
# it is not necessary to return an actual alignment trace.
# As a result, the ``score_only=True`` parameter increases the
# performance drastically.


X_DROP = 20
ACCEPT_THRESHOLD = 100

matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
ungapped_scores = np.array([
    align.align_local_ungapped(
        chloroplast_seq, bacterium_seqs[strand], matrix,
        seed=(i,j), threshold=X_DROP, score_only=True
    )
    for i, strand, j in double_hits
])

accepted_hits = double_hits[ungapped_scores > ACCEPT_THRESHOLD]
print("Number of accepted ungapped alignments:", len(accepted_hits))

########################################################################
# Finally the filtered match positions are used as seed for
# a gapped alignment using :func:`align_local_gapped`.
# For each match, this is by far the most time consuming step.
# However, since only a few matches remain, the total runtime is
# reasonable.
#
# Again, only the score should be returned from the gapped alignments
# to improve the performance.
# Only those alignments that show 'significant homology' are aligned
# again for later evaluation of the actual alignment.
#
# The significance of an homology is assessed by means of an
# `E-value <https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=FAQ#expect>`_.
# The :class:`EValueEstimator` computes the E-value from the similarity
# score.
# This calculation requires scoring scheme specific parameters that are
# estimated from a time-consuming sampling process.

X_DROP = 100
GAP_PENALTY = (-12, -4)
EVALUE_THRESHOLD = 0.1

# Initialize the estimator
# Use the symbol frequencies in the bacterial genome to sample
# the parameters
background = np.array(list(bacterium_seq.get_symbol_frequency().values()))
np.random.seed(0)
estimator = align.EValueEstimator.from_samples(
    chloroplast_seq.alphabet,
    # The scoring scheme must be the same as used for the alignment
    matrix, GAP_PENALTY,
    background
)

# Compute similarity scores for each hit
gapped_scores = np.array([
    align.align_local_gapped(
        chloroplast_seq, bacterium_seqs[strand], matrix,
        seed=(i,j), gap_penalty=GAP_PENALTY, threshold=X_DROP, score_only=True,
        max_table_size=100_000_000
    )
    for i, strand, j in accepted_hits
])

# Calculate the E-values
# For numeric stability reasons the method returns the common logarithm
log_evalues = estimator.log_evalue(
    gapped_scores, len(chloroplast_seq), 2 * len(bacterium_seq)
)

# Align regions with significant homology again
# This time report the actual alignment
accepted_alignments = [
    (
        align.align_local_gapped(
            chloroplast_seq, bacterium_seqs[strand], matrix,
            seed=(i,j), gap_penalty=GAP_PENALTY, threshold=X_DROP,
        )[0],
        log_evalue
    )
    for (i, strand, j), log_evalue in zip(accepted_hits, log_evalues)
    if log_evalue <= np.log10(EVALUE_THRESHOLD)
]

print("Number of accepted gapped alignments:", len(accepted_alignments))

########################################################################
# Some of the obtained alignments may be duplicates that should be
# removed:
# although the previous filter for double hits limits the number
# of alignments to one per diagonal, different matches on similar
# diagonals may still result in the same alignment due to the
# insertion of gaps.

# Sort by significance, i.e. the E-value
accepted_alignments = sorted(accepted_alignments, key=lambda x: x[1])

# For each position in the chloroplast genome report only one alignment
# The most significant ones take precedence
unique_alignments = []
covered_range = np.zeros(len(chloroplast_seq), dtype=bool)
for alignment, log_evalue in accepted_alignments:
    # The start and exclusive end position of the aligned region
    # in chloroplast genome
    start = alignment.trace[0, 0]
    stop = alignment.trace[-1, 0]
    # If this region was not covered by any other alignment before,
    # accept it and mark the region as covered
    if not covered_range[start : stop].any():
        unique_alignments.append((alignment, log_evalue))
        covered_range[start : stop] = True

print("Number of unique alignments:",  len(unique_alignments))

########################################################################
# To take a closer look on the found homologous regions, they are viewed
# in its functional context.
# For this purpose, the aligned region is displayed for each local
# alignment (*gray box*) and the features of the chloroplast genome
# are plotted alongside.
# Functional RNAs and protein coding regions are shown in orange and
# green, respectively.

N_COL = 4
MAX_NAME_LENGTH = 30
EXCERPT_SIZE = 3000
MARGIN_SIZE = 250

COLORS = {
    "CDS" : biotite.colors["dimgreen"],
    "tRNA": biotite.colors["orange"],
    "rRNA": biotite.colors["orange"]
}


# Fetch features of the chloroplast genome
gb_file = gb.GenBankFile.read(
    entrez.fetch("NC_000932", None, "gb", db_name="Nucleotide", ret_type="gb")
)
annotation = gb.get_annotation(gb_file, include_only=["CDS", "rRNA", "tRNA"])



def draw_arrow(ax, feature, loc):
    x = loc.first
    dx = loc.last - loc.first + 1
    if loc.strand == seq.Location.Strand.FORWARD:
        x = loc.first
        dx = loc.last - loc.first + 1
    else:
        x = loc.last
        dx = loc.first - loc.last + 1

    # Create head with 90 degrees tip -> head width/length ratio = 1/2
    ax.add_patch(biotite.AdaptiveFancyArrow(
        x, 0.5, dx, 0, tail_width=0.4, head_width=0.7, head_ratio=0.5,
        draw_head=True, color=COLORS[feature.key], linewidth=0
    ))

    label = feature.qual.get("gene")

    if label is not None:
        ax.text(
            x + dx/2, 0.5, label, color="black",
            ha="center", va="center", size=8
        )


# Fetch features of the chloroplast genome
gb_file = gb.GenBankFile.read(
    entrez.fetch("NC_000932", None, "gb", db_name="Nucleotide", ret_type="gb")
)
annotation = gb.get_annotation(gb_file, include_only=["CDS", "rRNA", "tRNA"])

n_rows = int(np.ceil(len(unique_alignments) / N_COL))
fig, axes = plt.subplots(
    n_rows, N_COL,
    figsize=(8.0, 24.0),
    constrained_layout=True
)

for (alignment, log_evalue), ax in zip(
    unique_alignments, axes.flatten()
):
    # Transform 0-based sequence index to 1-based sequence position
    first = alignment.trace[0, 0] + 1
    last = alignment.trace[-1, 0] + 1
    center = (first + last) // 2
    if last - first < EXCERPT_SIZE - MARGIN_SIZE * 2:
        excerpt_loc = (center - EXCERPT_SIZE//2, center + EXCERPT_SIZE//2)
    else:
        # Exceed excerpt size to show entire alignment range
        excerpt_loc = (first - MARGIN_SIZE, last + MARGIN_SIZE)
    # Move excerpt into bounds of the sequence
    if excerpt_loc[0] < 1:
        offset = -excerpt_loc[0] + 1
        excerpt_loc = (excerpt_loc[0] + offset, excerpt_loc[1] + offset)
    excerpt = annotation[excerpt_loc[0] : excerpt_loc[1] + 1]

    ax.axhline(0.5, color="lightgray", linewidth=2, zorder=-1)
    # Draw features
    for feature in excerpt:
        for loc in feature.locs:
            draw_arrow(ax, feature, loc)
    # Draw rectangle representing homologuous region
    ax.add_patch(Rectangle(
        (first, 0.1), last - first + 1, 1 - 2*0.1,
        facecolor="None", edgecolor="black", alpha=0.2, linewidth=1,
        clip_on=False
    ))

    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.tick_params(labelsize=6)
    ax.set_xlim(excerpt_loc)
    ax.set_ylim([0, 1])
    ax.set_frame_on(False)
    ax.get_yaxis().set_tick_params(left=False, right=False, labelleft=False)

    exponent = int(np.floor(log_evalue))
    mantissa = 10**(log_evalue-exponent)
    homolog_excerpt = annotation[first : last + 1]
    if len(homolog_excerpt) > 0:
        # Select the longest feature in range for name display in title
        representative_feature = max(
            homolog_excerpt,
            key=lambda feature: -np.subtract(*feature.get_location_range())
        )
        feature_name = representative_feature.qual["product"]
    else:
        # No feature in homologous region -> no name
        feature_name = ""
    # Truncate feature name if it is too long
    if len(feature_name) > MAX_NAME_LENGTH:
        feature_name = feature_name[:MAX_NAME_LENGTH] + "..."

    ax.set_title(
        f"{feature_name}\n"
        fr"E-Value: ${mantissa:.2f} \times 10^{{{exponent}}}$"
        f"\nIdentity: {align.get_sequence_identity(alignment) * 100:3.1f} %",
        loc="left", size=8
    )

# Hide empty axes
for ax in axes.flatten()[len(unique_alignments):]:
    ax.axis('off')

fig.tight_layout(h_pad=3.0, w_pad=0.5)

########################################################################
# This plot shows that the conserved regions sharply match the position
# of genes.
# Especially genes that are part of the gene expression machinery or
# participate in the composition of the photosystems seem to be highly
# conserved.
#
# References
# ----------
#
# .. footbibliography::
#