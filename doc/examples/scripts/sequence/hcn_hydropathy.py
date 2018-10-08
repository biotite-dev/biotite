"""
Positional hydropathy of HCN channels
=====================================

"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import biotite
import biotite.database.entrez as entrez
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import biotite.sequence.io.fasta as fasta
import biotite.sequence.io.genbank as gb
import biotite.application.mafft as mafft

hydropathy_dict = {
    "I" :  4.5,
    "V" :  4.2,
    "L" :  3.8,
    "F" :  2.8,
    "C" :  2.5,
    "M" :  1.9,
    "A" :  1.8,
    "G" : -0.4,
    "T" : -0.7,
    "S" : -0.8,
    "W" : -0.9,
    "Y" : -1.3,
    "P" : -1.6,
    "H" : -3.2,
    "E" : -3.5,
    "Q" : -3.5,
    "D" : -3.5,
    "N" : -3.5,
    "K" : -3.9,
    "R" : -4.5
}

# Look for the Swiss-Prot entry contaning the human HCN1 channel
query =   entrez.SimpleQuery("HCN1", "Gene Name") \
        & entrez.SimpleQuery("homo sapiens", "Organism") \
        & entrez.SimpleQuery("srcdb_swiss-prot", "Properties")
uids = entrez.search(query, db_name="protein")
file_name = entrez.fetch(
    uids[0], biotite.temp_dir(), "gb", db_name="protein", ret_type="gp"
)

gp_file = gb.GenPeptFile()
gp_file.read(file_name)
hcn1 = seq.ProteinSequence(gp_file.get_sequence())
print(hcn1)

########################################################################

hydropathies = np.array([hydropathy_dict[symbol] for symbol in hcn1])

def moving_average(data_set, window_size):
    weights = np.full(window_size, 1/window_size)
    return np.convolve(data_set, weights, mode='valid')

# Apply moving average over 15 amino acids for clearer visualization
ma_radius = 7
hydropathies = moving_average(hydropathies, 2*ma_radius+1)

figure = plt.figure(figsize=(8.0, 4.0))
ax = figure.add_subplot(111)
ax.plot(
    np.arange(1+ma_radius, len(hcn1)-ma_radius+1), hydropathies,
    color=biotite.colors["dimorange"]
)
ax.axhline(0, color="gray")
ax.set_xlim(1, len(hcn1)+1)
ax.set_xlabel("Sequence position")
ax.set_ylabel("Hydropathy (15 residues moving average)")

# Draw boxes for annotated transmembrane helices for comparison
# with hydropathy plot
annotation = gp_file.get_annotation(include_only=["Region"])
transmembrane_annotation = seq.Annotation(
    [feature for feature in annotation
     if feature.qual["region_name"] == "Transmembrane region"]
)
for feature in transmembrane_annotation:
    loc = feature.locs[0]
    ax.axvspan(loc.first, loc.last, color=(0.0, 0.0, 0.0, 0.2), linewidth=0)

########################################################################
#
#
#
#
#
#

names = ["HCN1", "HCN2", "HCN3", "HCN4"]

uids = []
for name in names:
    # Look for the Swiss-Prot entry contaning the human HCN<x> channel
    query =   entrez.SimpleQuery(name, "Gene Name") \
            & entrez.SimpleQuery("homo sapiens", "Organism") \
            & entrez.SimpleQuery("srcdb_swiss-prot", "Properties")
    uids += entrez.search(query, db_name="protein")
file_name = entrez.fetch_single_file(
    uids, biotite.temp_file("fasta"), db_name="protein", ret_type="fasta"
)

fasta_file = fasta.FastaFile()
fasta_file.read(file_name)

for header, seq_str in fasta_file:
    print(header)

########################################################################

sequences = []
for header, seq_str in fasta_file:
    sequences.append(seq.ProteinSequence(seq_str))

alignment = mafft.MafftApp.align(sequences)

def hydropathy_to_color(hydropathy, colormap):
    # Normalize hydropathy to range between 0 and 1
    # (orginally between -4.5 and 4.5)
    norm_hydropathy = (hydropathy - (-4.5)) / (4.5 - (-4.5))
    return colormap(norm_hydropathy)

# Create a color scheme highlighting the hydropathy 
colormap = plt.get_cmap("plasma_r")
colorscheme = [
    hydropathy_to_color(hydropathy_dict[symbol], colormap)
    if symbol in hydropathy_dict else None
    for symbol in sequences[0].get_alphabet()
]

# Show only the first 400 alignment columns for the sake of brevity
# This part contains all transmembrane helices
visualizer = graphics.AlignmentSymbolVisualizer(alignment[:400])
visualizer.set_color_scheme(colorscheme)
visualizer.add_labels(names, size=75)
visualizer.add_location_numbers(size=75)
# Color the symbols instead of the background
visualizer.set_alignment_properties(symbols_per_line=40, color_symbols=True)
figure = visualizer.generate()

plt.show()
