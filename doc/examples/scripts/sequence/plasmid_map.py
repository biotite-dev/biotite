"""
Plasmid map of a modified pSB1C3 vector
=======================================


"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import io
import requests
import matplotlib.pyplot as plt
import numpy as np
import biotite.sequence.io.genbank as gb
import biotite.sequence.graphics as graphics
import biotite.database.entrez as entrez


PLASMID_URL = "https://media.addgene.org/snapgene-media/" \
              "v1.6.2-0-g4b4ed87/sequences/67/17/246717/" \
              "addgene-plasmid-26094-sequence-246717.gbk"


#file_name = entrez.fetch(
#    "KX986172", target_path=biotite.temp_dir(),
#    suffix="gb", db_name="nuccore", ret_type="gb"
#)
#response = requests.get(PLASMID_URL)
#file = gb.GenBankFile()
#file.read(io.StringIO(response.text))
#annot_seq = gb.get_annotated_sequence(file)

response = requests.get(PLASMID_URL)
with open("test.gb", "w") as file:
    file.write(response.text)
file = gb.GenBankFile()
file.read("test.gb")
annotation = gb.get_annotation(file, include_only=[
    "promoter", "terminator", "protein_bind", "RBS", "CDS", "rep_origin"
])
primters = gb.get_annotation(file, include_only=["primer_bind"])
_, seq_length, _, _, _, _ = gb.get_locus(file)
plasmid_name = file.get_fields("KEYWORDS")[0][0][0]

def feature_formatter(feature):
    label = feature.qual.get("label")
    return True, "green", "black", label

fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.add_subplot(111, projection="polar")
graphics.plot_plasmid_map(
    ax, annotation, loc_range=(1, seq_length+1),
    label=plasmid_name, feature_formatter=feature_formatter
)
fig.tight_layout()
plt.show()
