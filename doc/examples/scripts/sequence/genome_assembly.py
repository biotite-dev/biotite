"""
Comparative genome assembly of SARS-CoV-2 B.1.1.7 variant
=========================================================

In the following script we will perform a comparative genome assembly of
the emerging SARS-CoV-2 B.1.1.7 variant in a simple manner.
We will use publicly available sequencing data of the virus variant
produced from a *Oxford Nanopore MinION*, map these sequence snippets
(reads) to the reference SARS-CoV-2 genome.
Then we will create a single consensus sequence from the mapped reads
and analyze where the differences to the reference genome are
(variant calling).
At last, we will focus on the mutations in the *spike protein* sequence.

To begin with, we download the relevant sequencing data from the *NCBI*
*sequence read archive* (SRA) using *Biotite's* interface to the
*SRA Toolkit*.
The software stores the downloaded sequence reads in one or multiple
FASTQ files, one for each read per *spot*:
A spot is a 'location' on the sequencing device.
One spot may produce more than one read, e.g. *Illumina* sequencers
use paired-end sequencing to produce a read starting from both ends
of the original sequence [1]_.
However, the *MinION* technology only creates one read per input
sequence, so we expect only a single FASTQ file.

A FASTQ file provides for each read it contains

    1. the sead sequence,
    2. associated *Phred* quality scores

Phred scores :math:`Q` describe the accuracy of each base in the read in
terms of base-call error probability :math:`P` [2]_:

.. math::

    Q = -10 \log_{10} P
"""
# Code source: Patrick Kunzmann
# License: BSD 3 clause

import itertools
import warnings
import tempfile
from concurrent.futures import ProcessPoolExecutor
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import biotite
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.io as seqio
import biotite.sequence.io.fasta as fasta
import biotite.sequence.io.fastq as fastq
import biotite.sequence.io.genbank as gb
import biotite.sequence.graphics as graphics
import biotite.database.entrez as entrez
import biotite.application.sra as sra


### Download the sequencing data
app = sra.FastqDumpApp("SRR13453793")
app.start()
app.join()

### Load sequences and quality scores from the sequencing data
# There is only one read per spot
file_path = app.get_file_paths()[0]
fastq_file = fastq.FastqFile.read(file_path, offset="Sanger")
reads = [seq.NucleotideSequence(seq_str)
         for seq_str, score_array in fastq_file.values()]
score_arrays = [score_array for seq_str, score_array in fastq_file.values()]

print(f"Number of reads: {len(reads)}")

########################################################################
# General sequencing data analysis
# --------------------------------
#
# First we will have a first glance on the quality of the sequencing
# data: the length of the reads and the Phred scores.

N_BINS = 200


fig, (length_ax, score_ax) = plt.subplots(nrows=2, figsize=(8.0, 6.0))

length_ax.hist(
    [len(score_array) for score_array in score_arrays],
    bins=np.logspace(1, 5, N_BINS), color="gray"
)
length_ax.set_xlabel("Read length")
length_ax.set_ylabel("Number of reads")
length_ax.set_xscale("log")
length_ax.set_yscale("log")

score_ax.hist(
    [np.mean(score_array) for score_array in score_arrays],
    bins=N_BINS, color="gray", 
)
score_ax.set_xlim(0, 30)
score_ax.set_xlabel("Phred score")
score_ax.set_ylabel("Number of reads")

fig.tight_layout()

########################################################################
# We can see the reads in the dataset are rather long, with most reads
# longer than 1000 bases.
# This is one of the big advantages of the employed sequencing
# technology.
# Especially for *de novo* genome assembly (which we will no do here),
# long reads facilitate the process.
# However, the sequencing method also comes with a disadvantage:
# The base-call accuracy is relatively low, as the Phred scores
# indicate.
# Keep in mind, that :math:`Q = 10` means that the called base at the
# respective position has a probability of 10 % to be wrong!
# Hence, a high sequencing depth, i.e. a large number of overlapping
# reads at each sequence position, is required to achieve accurate
# results.
# The partially low accuracy becomes even more visible, when creating
# a histogram over quality scores of individual bases, instead of
# averaging the scores over each read.

score_histogram = np.bincount(np.concatenate(score_arrays))

fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.fill_between(
    # Value in megabases -> 1e-6
    np.arange(len(score_histogram)), score_histogram * 1e-6,
    linewidth=0, color="gray"
)
ax.set_xlim(
    np.min(np.where(score_histogram > 0)[0]),
    np.max(np.where(score_histogram > 0)[0]),
)
ax.set_ylim(0, np.max(score_histogram * 1e-6) * 1.05)
ax.set_xlabel("Phred score")
ax.set_ylabel("Number of Mb")
fig.tight_layout()

########################################################################
# Optionally, you could exclude reads with exceptionally low Phred
# scores [?]_.
# But instead we will rely on a high sequencing depth to filter out
# erroneous base calls
#
# Read mapping
# ------------
#
# In the next step we will map each read to its respective position
# in the reference genome.
# An additional challenge is to find the correct sense of the read:
# In the library preparation both, sense and complementary DNA, is
# produced from the virus RNA.
# For this reason we need to create a complementary copy for each read
# map both strands to the reference genome.
# The the *wrong* strand is discarded.

### Download and read the reference SARS-CoV-2 genome
orig_genome_file = entrez.fetch(
    "NC_045512", tempfile.gettempdir(), "gb",
    db_name="Nucleotide", ret_type="gb"
)
orig_genome = seqio.load_sequence(orig_genome_file)

### Create complementary reads
compl_reads = list(itertools.chain(
    *[(read, read.reverse(False).complement()) for read in reads]
))

########################################################################

K = 12

alphabet = seq.NucleotideSequence.unambiguous_alphabet()

genome_table = align.KmerTable(alphabet, K)
genome_table.add(orig_genome, 0)

all_diagonals = []
for i, read in enumerate(compl_reads):
    matches = genome_table.match_sequence(read)
    diagonals = matches[:,2] - matches[:,0]
    all_diagonals.append(diagonals)

# k-mer tables use quite a large amount of RAM
# and we do not need those objects anymore
del genome_table

########################################################################

best_diagonals = [None] * len(all_diagonals)
for i, diagonals_for_read in enumerate(all_diagonals):
    diag, counts = np.unique(diagonals_for_read, return_counts=True)
    if len(diag) == 0:
        # If no match is found for this sequence, ignore this sequence
        continue
    best_diagonals[i] = diag[np.argmax(counts)]
del all_diagonals

########################################################################

BAND_WIDTH = 100
THRESHOLD_SCORE = 200

matrix = align.SubstitutionMatrix.std_nucleotide_matrix()

def map_sequence(read, diag):
    if diag is None:
        return None
    else:
        return align.align_banded(
            read, orig_genome, matrix, gap_penalty=-10,
            band = (diag - BAND_WIDTH//2, diag + BAND_WIDTH//2),
            max_number = 1
        )[0]

with ProcessPoolExecutor() as executor:
    alignments = list(executor.map(
        map_sequence, compl_reads, best_diagonals, chunksize=1000
    ))

alignments = [alignment
              if alignment is not None
              and alignment.score > THRESHOLD_SCORE else None
              for alignment in alignments]

########################################################################

for_alignments = [alignments[i] for i in range(0, len(alignments), 2)]
rev_alignments = [alignments[i] for i in range(1, len(alignments), 2)]

scores = np.stack((
    [ali.score if ali is not None else 0 for ali in for_alignments],
    [ali.score if ali is not None else 0 for ali in rev_alignments]
),axis=-1)

correct_sense = np.argmax(scores, axis=-1)
correct_alignments = [for_a if sense == 0 else rev_a for for_a, rev_a, sense
                      in zip(for_alignments, rev_alignments, correct_sense)]
# If we use a reverse complementary read,
# we also need to reverse the scores
correct_score_arrays = [score if sense == 0 else score[::-1] for score, sense
                        in zip(score_arrays, correct_sense)]

########################################################################

starts = np.array(
    [ali.trace[ 0, 1] for ali in correct_alignments if ali is not None]
)
stops = np.array(
    [ali.trace[-1, 1] for ali in correct_alignments if ali is not None]
)
order = np.argsort(starts)
starts = starts[order]
stops = stops[order]
              
fig, ax = plt.subplots(figsize=(8.0, 12.0))
ax.barh(
    np.arange(len(starts)), left=starts, width=stops-starts, height=1,
    color=biotite.colors["dimgreen"], linewidth=0
)
ax.set_ylim(0, len(starts)+1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False, labelleft=False)
ax.set_xlabel("Sequence position")
ax.set_title("Read mappings to reference genome")
fig.tight_layout()

########################################################################
# Variant calling
# ---------------
#
# .. note:: For the sake of brevity possible insertions into the
#    reference genome are not considered in the approach shown here.

phred_sum = np.zeros((len(orig_genome), 4), dtype=int)
sequencing_depth = np.zeros(len(orig_genome), dtype=int)
deletion_number = np.zeros(len(orig_genome), dtype=int)

for alignment, score_array in zip(correct_alignments, correct_score_arrays):
    if alignment is not None:
        trace = alignment.trace
        
        no_gap_trace = trace[(trace[:,0] != -1) & (trace[:,1] != -1)]
        seq_code = alignment.sequences[0].code[no_gap_trace[:,0]]
        phred_sum[no_gap_trace[:,1], seq_code] \
            += score_array[no_gap_trace[:,0]]
        
        no_genome_gap_trace = trace[trace[:,1] != -1]
        sequencing_depth[
            no_genome_gap_trace[0,1] : no_genome_gap_trace[-1,1]
        ] += 1
        
        read_gap_trace = trace[trace[:,0] == -1]
        deletion_number[read_gap_trace[:,1]] += 1

########################################################################

most_probable_symbol_codes = np.argmax(phred_sum, axis=1)
max_phred_sum = phred_sum[np.arange(len(phred_sum)), most_probable_symbol_codes]

def moving_average(data_set, window_size):
    weights = np.full(window_size, 1/window_size)
    return np.convolve(data_set, weights, mode='valid')

fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.plot(
    moving_average(max_phred_sum, 100),
    color="lightgray", linewidth=1.0
)
ax2 = ax.twinx()
ax2.plot(
    moving_average(sequencing_depth, 100),
    color=biotite.colors["dimorange"], linewidth=1.0
)
ax.axhline(0, color="silver", linewidth=0.5)
ax.set_xlim(0, len(orig_genome))
ax.set_xlabel("Genome postion")
ax.set_ylabel("Phred score sum")
ax2.set_ylabel("Sequencing depth")
ax.legend(
    [Line2D([0], [0], color=c)
     for c in ("lightgray", biotite.colors["dimorange"])],
    ["Phred score sum", "Sequencing depth"],
    loc="upper left"
)
fig.tight_layout()

########################################################################
# No region has generally bad Phred score

# At least 60 % of all reads at a position must include a deletion,
# so that the deletion is accepted
DELETION_THRESHOLD = 0.6

var_genome = seq.NucleotideSequence()
var_genome.code = most_probable_symbol_codes
# A deletion is called, if either enough reads include this deletion
# or the sequence position is not covered by any read at all
deletion_mask = (deletion_number > sequencing_depth * DELETION_THRESHOLD) \
                | (sequencing_depth == 0)
var_genome = var_genome[~deletion_mask]
# Write the assembled genome into a FASTA file
out_file = fasta.FastaFile()
fasta.set_sequence(
    out_file, var_genome, header="SARS-CoV-2 B.1.1.7", as_rna=True
)
out_file.write(tempfile.NamedTemporaryFile("w"))

########################################################################
# We have done it, the genome of the B.1.1.7 variant is assembled!
# Now we would like to have a closer look on the difference between the
# original and the B.1.1.7 genome.
#
# Mutations in the B.1.1.7 variant
# --------------------------------

BAND_WIDTH = 200

genome_alignment = align.align_banded(
    var_genome, orig_genome, matrix,
    band=(-BAND_WIDTH//2, BAND_WIDTH//2), max_number=1
)[0]
identity = align.get_sequence_identity(genome_alignment, 'all')
print(f"Sequence identity: {identity * 100:.2f} %")

########################################################################

N_BINS = 50

gb_file = gb.GenBankFile.read(orig_genome_file)
annot_seq = gb.get_annotated_sequence(gb_file, include_only=["gene"])

bin_identities = np.zeros(N_BINS)
edges = np.linspace(0, len(orig_genome), N_BINS+1)
for i, (bin_start, bin_stop) in enumerate(zip(edges[:-1], edges[1:])):
    orig_genome_trace = genome_alignment.trace[:,1]
    excerpt = genome_alignment[(orig_genome_trace >= bin_start) & (orig_genome_trace < bin_stop)]
    bin_identities[i] = align.get_sequence_identity(excerpt, "all")


fig, (deviation_ax, feature_ax) = plt.subplots(nrows=2, figsize=(8.0, 5.0))

deviation_ax.bar(
    edges[:-1], width=(edges[1:]-edges[:-1]),
    height=(1 - bin_identities),
    color=biotite.colors["dimorange"], align="edge"
)
deviation_ax.set_xlim(0, len(orig_genome))
deviation_ax.set_ylabel("1 - Sequence identity")
deviation_ax.set_title("Sequence deviation of SARS-CoV-2 B.1.1.7 variant")
deviation_ax.set_yscale("log")
deviation_ax.set_ylim(1e-3, 1e-1)

for i, feature in enumerate(sorted(
    annot_seq.annotation,
    key=lambda feature: min([loc.first for loc in feature.locs])
)):
    for loc in feature.locs:
        feature_ax.barh(
            left=loc.first, width=loc.last-loc.first, y=i, height=1,
            color=biotite.colors["dimgreen"]
        )
        feature_ax.text(
            loc.last + 100, i, feature.qual["gene"],
            fontsize=8, ha="left", va="center"
        )
feature_ax.set_ylim(i+0.5, -0.5)
feature_ax.set_xlim(0, len(orig_genome))
feature_ax.xaxis.set_visible(False)
feature_ax.yaxis.set_visible(False)
feature_ax.set_frame_on(False)
# sphinx_gallery_thumbnail_number = 5

########################################################################
# The *S* gene codes for the infamous *spike protein*: a membrane
# protein on the surface of SARS-CoV-2 that drives the infiltration of
# the host cell.
# Let's have closer look on it.
#
# Differences in the Spike protein
# --------------------------------
# 

for feature in annot_seq.annotation:
    if feature.qual["gene"] == "S":
        orig_spike_seq = annot_seq[feature]
        
alignment = align.align_optimal(
    var_genome, orig_spike_seq, matrix, local=True, max_number=1
)[0]
var_spike_seq = var_genome[alignment.trace[alignment.trace[:,0] != -1, 0]]

orig_spike_prot_seq = orig_spike_seq.translate(complete=True)
var_spike_prot_seq  =  var_spike_seq.translate(complete=True)


spike_annotation_file = gb.GenBankFile.read(entrez.fetch(
    "P0DTC2", ".", "gp", db_name="Protein", ret_type="gp"
))

with warnings.catch_warnings():
    # Ignore warnings about unsupported bond location identifiers
    warnings.simplefilter("ignore")
    spike_annotation = gb.get_annotation(spike_annotation_file)

spike_features = {}
# Get the sequence locations of the signal peptide
# and the receptor-binding domain (RBD)
for feature in spike_annotation:
    qual = feature.qual
    if "region_name" in qual and "Signal" in qual["region_name"]:
        # Expect only one location, i.e. the feature is continuous
        loc = list(feature.locs)[0]
        spike_features["Signal"] = (loc.first, loc.last)
    if "note" in qual and "RBD" in qual["note"]:
        loc = list(feature.locs)[0]
        spike_features["RBD"] = (loc.first, loc.last)

for feature_name, loc in spike_features.items():
    print(f"{feature_name}: {loc[0]} - {loc[1]}")

########################################################################

SYMBOLS_PER_LINE = 75
SPACING = 3

blosum_matrix = align.SubstitutionMatrix.std_protein_matrix()
alignment = align.align_optimal(
    var_spike_prot_seq, orig_spike_prot_seq, blosum_matrix, max_number=1
)[0]

fig = plt.figure(figsize=(8.0, 10.0))
ax = fig.add_subplot(111)

cmap = LinearSegmentedColormap.from_list(
    "custom", colors=[(1.0, 0.3, 0.3), (1.0, 1.0, 1.0)]
    #                    ^ reddish        ^ white
)
graphics.plot_alignment_similarity_based(
    ax, alignment, matrix=blosum_matrix, symbols_per_line=SYMBOLS_PER_LINE,
    labels=["B.1.1.7", "Reference"], show_numbers=True, label_size=9,
    number_size=9, symbol_size=7, spacing=SPACING, cmap=cmap
)

for row in range(1 + len(alignment) // SYMBOLS_PER_LINE):
    col_start = SYMBOLS_PER_LINE * row
    col_stop  = SYMBOLS_PER_LINE * (row + 1)
    if col_stop > len(alignment):
        # This happens in the last line
        col_stop = len(alignment)
    seq_start = alignment.trace[col_start, 1]
    seq_stop  = alignment.trace[col_stop-1,  1] + 1
    n_sequences = len(alignment.sequences)
    y_base = (n_sequences + SPACING) * row + n_sequences
    
    for feature_name, (start, stop) in spike_features.items():
        # Zero based sequence indexing
        start -= 1
        # Stop needs not to be adjusted, since we use exclusive stops
        if start < seq_stop and stop > seq_start:
            # The feature is found in this line
            x_begin = np.clip(start - seq_start,    0, SYMBOLS_PER_LINE+1)
            x_end   = np.clip(stop - seq_start + 1, 0, SYMBOLS_PER_LINE+1)
            x_mean = (x_begin + x_end) / 2
            y_line = y_base + 0.3
            y_text = y_base + 0.6
            ax.plot(
                [x_begin, x_end], [y_line, y_line],
                color="black", linewidth=2
            )
            ax.text(
                x_mean, y_text, feature_name, 
                fontsize=8, va="top"
            )

fig.tight_layout()

plt.show()

########################################################################
# [3]_
#
# References
# ----------
#
# .. [1] https://www.illumina.com/science/technology/next-generation-sequencing/plan-experiments/paired-end-vs-single-read.html
# .. [2] PJA Cock, CJ Fields, N Goto, ML Heuer, PM Rice,
#    "The Sanger FASTQ file format for sequences with quality scores, 
#    and the Solexa/Illumina FASTQ variants."
#    Nucleic Acids Res, 38, 1767-1771 (2010).
# .. [3] https://asm.org/Articles/2021/January/B-1-1-7-What-We-Know-About-the-Novel-SARS-CoV-2-Va