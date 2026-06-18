"""
Plasmid map of a vector
=======================

This script downloads the GenBank file for the *pET-28a* expression vector
from the *NCBI Nucleotide* database and draws a plasmid map using a custom
feature formatter.

- **Promoters** - green arrow
- **Terminators** - red arrow
- **CDS** - orange arrow
- **Ori** - gray arrow
- **RBS** - light orange rectangle
- **Other features** - light green rectangle
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import matplotlib.pyplot as plt
import biotite
import biotite.database.entrez as entrez
import biotite.sequence.graphics as graphics
import biotite.sequence.io.genbank as gb

# The NCBI Nucleotide accession of the pET-28a vector
PLASMID_ID = "EF442785"


gb_file = gb.GenBankFile.read(entrez.fetch(PLASMID_ID, None, "gb", "nuccore", "gb"))
annotation = gb.get_annotation(
    gb_file,
    include_only=[
        "regulatory",
        "CDS",
        "rep_origin",
        "misc_feature",
    ],
)
_, seq_length, _, _, _, _ = gb.get_locus(gb_file)


def custom_feature_formatter(feature):
    # Use the gene name, product or note as label
    label = feature.qual.get(
        "gene", feature.qual.get("product", feature.qual.get("note"))
    )
    if feature.key == "regulatory":
        # 'promoter', 'terminator', 'ribosome_binding_site', etc.
        regulatory_class = feature.qual.get("regulatory_class")
        if regulatory_class == "promoter":
            return True, biotite.colors["dimgreen"], "black", "promoter"
        elif regulatory_class == "terminator":
            return True, "firebrick", "black", "terminator"
        elif regulatory_class == "ribosome_binding_site":
            return False, biotite.colors["brightorange"], "black", "RBS"
    elif feature.key == "CDS":
        return True, biotite.colors["orange"], "black", label
    elif feature.key == "rep_origin":
        return True, "lightgray", "black", "ori"
    elif feature.key == "misc_feature":
        return False, biotite.colors["lightgreen"], "black", label
    # Fallback for any other feature
    return True, "lightgray", "black", label


fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.add_subplot(111, projection="polar")
graphics.plot_plasmid_map(
    ax,
    annotation,
    plasmid_size=seq_length,
    label="pET-28a",
    feature_formatter=custom_feature_formatter,
)
fig.tight_layout()
plt.show()
