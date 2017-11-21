# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

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
    
    def __copy_create__(self):
        return CDSFeature(
            self.get_product(), self.get_gene(), self.get_location() )
    
    def is_hypothetical(self):
        return self._hypothetical
    
    def get_product(self):
        return self.get_name()
    
    def get_gene(self):
        return self._gene