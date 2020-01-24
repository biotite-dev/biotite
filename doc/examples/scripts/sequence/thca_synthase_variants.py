"""
Sequence analysis of THCA synthase variants
===========================================

"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import biotite.sequence as seq
import biotite.sequence.io.genbank as gb
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import biotite.database.entrez as entrez
import biotite.application.clustalo as clustalo


# Search for DNA sequences that belong to the aforementioned article
query =   entrez.SimpleQuery("Forensic Sci. Int.", "Journal") \
        & entrez.SimpleQuery("159", "Volume") \
        & entrez.SimpleQuery("132-140", "Page Number")
uids = entrez.search(query, db_name="nuccore")

multi_file = gb.MultiFile()
multi_file.read(entrez.fetch_single_file(
    uids, file_name=None, db_name="nuccore", ret_type="gb"
))


# This dictionary maps the strain ID to the protein sequence
sequences = {}

for gb_file in multi_file:
    annotation = gb.get_annotation(gb_file)
    
    # Find ID of strain in 'source' feature
    strain = None
    for feature in annotation:
        if feature.key == "source":
            strain = int(feature.qual["strain"])
    assert strain is not None
    
    # Find corresponding protein sequence in 'CDS' feature
    sequence = None
    for feature in annotation:
        if feature.key == "CDS":
            sequence = seq.ProteinSequence(
                # Remove whitespace in sequence
                # resulting from line breaks
                feature.qual["translation"].replace(" ", "")
            )
    assert sequence is not None

    sequences[strain] = sequence


# Perform a multiple sequence alignment the THCA sequences
ordered_strains = (1, 10, 13, 20, 53, 54,   9, 5, 11, 45, 66, 68, 78)
orderes_seqs = [sequences[strain] for strain in ordered_strains]
alignment = clustalo.ClustalOmegaApp.align(orderes_seqs)


# A colormap for hightlighting sequence dissimilarity
cmap = LinearSegmentedColormap.from_list(
    "custom", colors=[(1.0, 0.3, 0.3), (1.0, 1.0, 1.0)]
    #                    ^ reddish        ^ white
)

fig = plt.figure(figsize=(8.0, 30.0))
ax = fig.add_subplot(111)

labels = [f"#{strain:02d}" for strain in ordered_strains]
# Type of strains following the respective indicator
labels[0] = "Drug-type  "  + labels[0]
labels[6] = "Fiber-type  " + labels[6]

symbols_per_lines
graphics.plot_alignment_similarity_based(
    ax, alignment, labels=labels, show_numbers=True, spacing=2.0, cmap=cmap
)

fig.tight_layout()

plt.show()