"""
Custom visualization of a custom plasmid
========================================


"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import biotite
import biotite.sequence as seq
import biotite.sequence.io.genbank as gb
import biotite.sequence.graphics as graphics
import biotite.database.entrez as entrez


NAME = "This is   a A-Test_bla_bla______________________--------------"
#NAME = "T_"
#NAME = "araC promoter"
#NAME = "Test"

annotation = seq.Annotation([
    seq.Feature(
        "CDS", [seq.Location(190,310)]
    ),
    seq.Feature(
        "gene", [seq.Location(190,310)]
    ),
    seq.Feature(
        "CDS", [seq.Location(200,300)], {"product": NAME}
    )
])

fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.add_subplot(111, projection="polar")
graphics.plot_plasmid_map(
    ax, annotation, plasmid_size=700,
    label="My plasmid"
)

ticks = ax.get_xticks()
labels = ax.get_xticklabels()
print(ticks)
print(np.pi/2)
ax.set_xticks(list(ticks) + [np.pi/2])
print(ax.get_xticks())
ax.set_xticklabels(labels + ["EcoRI"])

fig.tight_layout()
plt.show()
