# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from .annotation import Feature

__all__ = ["CDSFeature"]


class CDSFeature(Feature):
    
    def __init__(self, product, gene, locs):
        super().__init__(product, locs)
        self._gene = gene
        if product == "hypothetical protein":
            self._hypothetical = True
        else:
            self._hypothetical = False
    
    def is_hypothetical(self):
        return self._hypothetical
    
    def get_product(self):
        return self.get_name()
    
    def get_gene(self):
        return self._gene