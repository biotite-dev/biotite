r"""
Plot epitope mapping data from peptide arrays
onto protein sequence alignments
============================================

Peptide arrays of overlapping sequences can be used to identify
the epitope of antibodies on a protein antigen at amino acid level.
Scannings for molecular recognition using peptide arrays, 
are particlularly potent for epitope identification on monoclonal 
antibodies. This script visualizes the data from epitope mapping 
screenings, using a color-coded sequence alignment representation 
of the antigens screened. The scanning interrogated a monoclonal 
antibody against the extracellular domain of two alleles of 
the *Plasmodiun falciparum* virulence factor VAR2CSA. Arbritary 
units(AU) of fluorescence intensity quantified the molecular
recognition for each peptide on the peptide arrays.


the peptide array scanning data can be downloaded
:download:`here </examples/download/FCR3_10ug.csv>`
and 
:download:`here </examples/download/NF54_10ug.csv>`
the sequence file of the two VAR2CSA alleles can be downloaded
:download:`here </examples/download/waterbox_md.pdb>`.

This example normalizes the flourescence data using the power
law with cubic exponent. Several data transformations are 
available in :mod:`biotite.sequence.align` through the 
function :func:`data_transform()`.

"""

# Code source: Daniel Ferrer-Vinals
# License: BSD 3 clause


import matplotlib.pyplot as plt
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.io.fasta as fasta
import biotite.sequence.graphics as graphics

# Path to the data files
array_seq_path = "../../download/Array_Seq.txt"
FCR3_file_path  = "../../download/FCR3_10ug.csv"
NF54_file_path  = "../../download/NF54_10ug.csv"

fasta_file = fasta.FastaFile.read(array_seq_path)

# Parse protein sequences of FCR3 and NF54
for name, sequence in fasta_file.items():
    if "AAQ73926" in name:
        FCR3_seq = seq.ProteinSequence(sequence)
    elif "EWC87419" in name:
        NF54_seq = seq.ProteinSequence(sequence)

# Get BLOSUM62 matrix
matrix = align.SubstitutionMatrix.std_protein_matrix()
# Perform pairwise sequence alignment with affine gap penalty
# Terminal gaps are not penalized
alignments = align.align_optimal(FCR3_seq, NF54_seq, matrix,
                                 gap_penalty = (-10, -1), 
                                 terminal_penalty = False)

# Load epitope scan data

files = [FCR3_file_path, NF54_file_path]

d = 0
for f in files:
    if f == files[0]:
        ag1_scan = sa.read_scan(files[d], 20, 20)
    elif f == files[1]:
        ag2_scan = sa.read_scan(files[d], 20, 20)
    d = d + 1

# Compute the statistics of the data    
dfa = graphics.compute_params(ag1_scan, combine = 'max',
                              flag_noisy = True)
dfb = graphics.compute_params(ag2_scan, combine = 'max', 
                              flag_noisy = True)

# Inspect the data and define a threshold
graphics.data_describe(dfa) 
graphics.data_describe(dfb) 

# Normalize and apply the threshold
graphics.data_transform(dfa, method ='cubic', threshold = 0)
graphics.data_transform(dfb, method ='cubic', threshold = 0)

# Convert a list of score residues from the epitope
# scan data into a aligment-like gapped sequences

A = alignments[0]
traceA = align.get_symbols(A)[0]
traceB = align.get_symbols(A)[1]

"""
In this example, screened peptides were 20 amino acids
in length with an overlap of 19 and 18 amino acids for the
FCR3 or NF54 arrays, respectively.
"""
# FCR3 array, overlap_step: 1 (pep = 20-mer with 19 overlap)
gapd_s1 = graphics.gapped_seq(dfa, traceA, 20, 1)

# NF54 array, overlap_step: 2 (pep = 20-mer with 18 overlap)
gapd_s2 = graphics.gapped_seq(dfb, traceB, 20, 2) 

# Checkpoint
len(gapd_s1) == len(gapd_s2)

# Create a signal_map (ndarray)
score = sa.signal_map(gapd_s1, gapd_s2)

# Plot:
fig = plt.figure(figsize=(20, 16))
ax = fig.add_subplot(111)
graphics.plot_alignment_array(
    ax, alignments[0], fl_score = score, labels = ["FCR3", "NF54"],
    show_numbers = True, symbols_per_line = 120,
    show_line_position = True) 

# add a 2nd axes and a colorbar
ax2 = fig.add_axes([0.1,-0.005, 0.8, 0.03])
ax2.set_frame_on(False)
cmp = graphics.get_cmap(ax2, score)
cbar = graphics.get_colorbar(ax2, dfa, dfb, cmp, transform = 'cubic', 
                       orient = 'horizontal', 
                       title = 'Fluorescence Intensity [AU]')

plt.show()
