# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from ..features import *
from .features import *
from ..annotation import Location
from matplotlib.patches import Rectangle

__all__ = ["RegionMap"]


base_syle = {CDSFeature: draw_cds}

class RegionMap():
    
    def __init__(self, axes):
        self._ax = axes
        self._ax.set_aspect("equal")
        self._ax.axis('off')
        self.set_size(10)
        self.set_lim(0,10000)
    
    def draw(self, annotation, style=base_syle, **kwargs):
        self._ax.add_patch(Rectangle((0, self._size/2 - 25), 5000000, 50,
                                     edgecolor="None", facecolor="gray"))
        for feature in annotation:
            try:
                draw_func = style[type(feature)]
            except KeyError:
                raise ValueError("Invalid feature type '{:}'"
                                 .format(type(feature).__name__))
            for loc in feature.get_location():
                if loc.strand == Location.Strand.FORWARD:
                    dir = "right"
                else:
                    dir = "left"
                draw_func(self._ax, feature, x=loc.first-0.5, y=self._size/2,
                          width=loc.last-loc.first+1, height=self._size,
                          defect=loc.defect, dir = dir, **kwargs)
    
    def set_lim(self, min, max):
        self._ax.set_xlim(min, max)
        
    def set_size(self, size):
        self._size = size
        self._ax.set_ylim(0, size)
