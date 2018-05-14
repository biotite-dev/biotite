"""
Feature map of a synthetic operon
=================================

This script shows how to create a picture of an synthetic operon for
publication purposes.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import matplotlib.pyplot as plt
import biotite
from biotite.sequence import Annotation, Feature, Location
import biotite.sequence.graphics as graphics

strand = Location.Strand.FORWARD
prom  = Feature("regulatory", [Location(10, 50, strand)],
                {"regulatory_class" : "promoter",
                 "note"             : "T7"})
rbs1  = Feature("regulatory", [Location(60, 75, strand)],
                {"regulatory_class" : "ribosome_binding_site",
                 "note"             : "RBS1"})
gene1 = Feature("gene", [Location(81, 380, strand)],
                {"gene" : "gene1"})
rbs2  = Feature("regulatory", [Location(400, 415, strand)],
                {"regulatory_class" : "ribosome_binding_site",
                 "note"             : "RBS2"})
gene2 = Feature("gene", [Location(421, 1020, strand)],
                {"gene" : "gene2"})
term = Feature("regulatory", [Location(1050, 1080, strand)],
                {"regulatory_class" : "terminator"})
annotation = Annotation([prom, rbs1, gene1, rbs2, gene2, term])
feature_map = graphics.FeatureMap(
    annotation, multi_line=False, show_numbers=False
)
figure = feature_map.generate()
plt.show()
