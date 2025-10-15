"""
Plasmid map of a vector
=======================

This script downloads the GenBank file for a *pET15* plasmid from
*AddGene* and draws a plasmid map using a custom feature formatter.

- **Promoters** - green arrow
- **Terminators** - red arrow
- **Protein binding sites** - light green rectangle
- **RBS** - light orange rectangle
- **CDS** - orange arrow
- **Ori** - gray arrow
- **Primer** - blue arrow
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import io
import matplotlib.pyplot as plt
import requests
import biotite
import biotite.sequence.graphics as graphics
import biotite.sequence.io.genbank as gb

PLASMID_URL = (
    "https://media.addgene.org/snapgene-media/v3.0.0/sequences/466943/"
    "17c6ce2c-cf6d-46e8-a4c9-58cd4a4760b6/addgene-plasmid-26092-sequence-466943.gbk"
)


response = requests.get(PLASMID_URL)
gb_file = gb.GenBankFile.read(io.StringIO(response.text))
annotation = gb.get_annotation(
    gb_file,
    include_only=[
        "promoter",
        "terminator",
        "protein_bind",
        "RBS",
        "CDS",
        "rep_origin",
        "primer_bind",
    ],
)
_, seq_length, _, _, _, _ = gb.get_locus(gb_file)


def custom_feature_formatter(feature):
    # AddGene stores the feature label in the '\label' qualifier
    label = feature.qual.get("label")
    if feature.key == "promoter":
        return True, biotite.colors["dimgreen"], "black", label
    elif feature.key == "terminator":
        return True, "firebrick", "black", label
    elif feature.key == "protein_bind":
        return False, biotite.colors["lightgreen"], "black", label
    elif feature.key == "RBS":
        return False, biotite.colors["brightorange"], "black", label
    elif feature.key == "CDS":
        return True, biotite.colors["orange"], "black", label
    elif feature.key == "rep_origin":
        return True, "lightgray", "black", label
    elif feature.key == "primer_bind":
        return True, "blue", "black", label


fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.add_subplot(111, projection="polar")
graphics.plot_plasmid_map(
    ax,
    annotation,
    plasmid_size=seq_length,
    label="pET15-MHL",
    feature_formatter=custom_feature_formatter,
)
fig.tight_layout()
plt.show()
