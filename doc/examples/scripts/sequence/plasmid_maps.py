"""
Plasmid map of an modified pSB1C3 vector
========================================


"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import biotite
import biotite.sequence as sequence
import biotite.sequence.io.genbank as gb
import biotite.sequence.graphics as graphics
import biotite.database.entrez as entrez
"""
file_name = entrez.fetch(
    "L37382", target_path=biotite.temp_dir(),
    suffix="gb", db_name="nuccore", ret_type="gb"
)
file = gb.GenBankFile()
file.read(file_name)
annot_seq = file.get_annotated_sequence(ignore=["source"])

fig = plt.figure(figsize=(8.0, 0.8))
ax = fig.add_subplot(111)
graphics.plot_feature_map(
    ax, annot_seq.annotation,
    multi_line=False, loc_range=(1, len(annot_seq.sequence)),
)
fig.tight_layout()
plt.show()
"""