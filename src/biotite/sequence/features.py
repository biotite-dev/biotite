# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from .annotation import Feature
import copy

__all__ = ["CDSFeature", "PromoterFeature", "TerminatorFeature"]


class CDSFeature(Feature):
    
    def __init__(self, product, gene,
                 locs, note=""):
        super().__init__(product, locs, note)
        self._gene = gene
        if product == "hypothetical protein":
            self._hypothetical = True
        else:
            self._hypothetical = False
    
    def __copy_create__(self):
        return CDSFeature(
            self.get_product(), self.get_gene(),
            self.get_location(), self.get_note() )
    
    def is_hypothetical(self):
        return self._hypothetical
    
    def get_product(self):
        return self.get_name()
    
    def get_gene(self):
        return self._gene


class PromoterFeature(Feature):
    
    def __init__(self, name, locs, strength=-1, inducers=[], silencers=[],
                 note=""):
        super().__init__(name, locs, note)
        self._strength = strength
        self._inducers = inducers
        self._silencers = silencers
    
    def __copy_create__(self):
        return PromoterFeature(
            self.get_name(), self.get_location(),
            self.get_strength(), self.get_inducers(), self.get_silencers(),
            self.get_note() )
    
    def get_strength(self):
        return self._strength
    
    def get_inducers(self):
        return copy.copy(self._inducers)
    
    def get_silencers(self):
        return copy.copy(self._silencers)


class TerminatorFeature(Feature):
    
    def __init__(self, name, locs, for_efficiency=1, rev_efficiency=0,
                 note=""):
        super().__init__(name, locs, note)
        self._for_efficiency = for_efficiency
        self._rev_efficiency = rev_efficiency
    
    def __copy_create__(self):
        return TerminatorFeature(
            self.get_name(), self.get_location(),
            self._for_efficiency, self._rev_efficiency,
            self.get_note() )
    
    def get_strength(self):
        return self._strength
    
    def get_efficiency(self):
        return (self._for_efficiency, self._rev_efficiency)
