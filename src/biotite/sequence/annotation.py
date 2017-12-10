# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from .sequence import Sequence
from ..copyable import Copyable
import copy
import sys
from enum import IntEnum

__all__ = ["Location", "Feature", "Annotation", "AnnotatedSequence"]



class Location():
    
    class Defect(IntEnum):
        NONE         = 0
        MISS_LEFT    = 1
        MISS_RIGHT   = 2
        BEYOND_LEFT  = 4
        BEYOND_RIGHT = 8
        UNK_LOC      = 16
        BETWEEN      = 32

    class Strand(IntEnum):
        FORWARD = 1
        REVERSE = -1
    
    def __init__(self, first, last, strand=Strand.FORWARD,
                 defect=Defect.NONE):
        self.first = first
        self.last = last
        self.strand = strand
        self.defect = defect
    
    def __str__(self):
        string = "{:d}-{:d}".format(self.first, self.last)
        if self.strand == Location.Strand.FORWARD:
            string = string + " >"
        else:
            string = "< " + string
        if self.defect != Location.Defect(0):
            string += " " + str(self.defect)
        return string
    

class Feature(Copyable):
    
    def __init__(self, key, locs, qual={}):
        self._key = key
        self._locs = copy.deepcopy(locs)
        self._qual = copy.deepcopy(qual)
        self._subfeatures = []
    
    def __copy_create__(self):
        return Feature(self._key, self._locs, self._qual)
        
    def __copy_fill__(self, clone):
        super().__copy_fill__(clone)
        for feature in self._subfeatures:
            clone._subfeatures.append(feature.copy())
    
    def add_subfeature(self, feature):
        self._subfeatures.append(feature.copy())
    
    def del_subfeature(self, feature):
        self._subfeatures.remove(feature)
    
    def del_all_subfeatures(self):
        self._subfeatures = []
    
    def get_subfeatures(self):
        return copy.copy(self._subfeatures)
    
    @property
    def key(self):
        return self._key
    
    @property
    def locs(self):
        return self._locs
    
    @property
    def qual(self):
        return self._qual


class Annotation(object):
    
    def __init__(self, features=[]):
        self._features = copy.copy(features)
        
    def get_features(self):
        return copy.copy(self._features)
    
    def add_feature(self, feature):
        self._features.append(feature.copy())
    
    def del_feature(self, feature):
        if not feature in self._features:
            raise KeyError("Feature is not in annotation")
        self._features.remove(feature)
    
    def release_from(self, feature):
        if not feature in self._features:
            raise KeyError("Feature is not in annotation")
        subfeature_list = feature.get_subfeatures()
        feature.del_all_subfeatures()
        for f in subfeature_list:
            self.add_feature(f)
    
    def __iter__(self):
        i = 0
        while i < len(self._features):
            yield self._features[i]
            i += 1
    
    def __contains__(self, item):
        if not isinstance(item, Feature):
            raise TypeError("Annotation instances only contain features")
        return item in self._features
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            i_first = index.start
            # If no start or stop index is given, include all
            if i_first is None:
                i_first = -sys.maxsize
            i_last = index.stop -1
            if i_last is None:
                i_last = sys.maxsize
            sub_annot = Annotation()
            for feature in self:
                in_scope = False
                for loc in feature.locs:
                    # Always true for maxsize values
                    # in case no start or stop index is given
                    if loc.first <= i_last and loc.last >= i_first:
                        in_scope = True
                if in_scope:
                    # Create copy for new annotation
                    new_feature = feature.copy()
                    for loc in new_feature.locs:
                        # Handle defects
                        if loc.first < i_first:
                            print("test1")
                            loc.defect |= Location.Defect.MISS_LEFT
                            loc.first = i_first
                        if loc.last > i_last:
                            print("test2")
                            loc.defect |= Location.Defect.MISS_RIGHT
                            loc.last = i_last
                    sub_annot.add_feature(new_feature)
            return sub_annot            
        else:
            raise TypeError("{:} instances are invalid Annotation indices"
                            .format(type(index).__name__))


class AnnotatedSequence(object):
    
    def __init__(self, annotation, sequence, sequence_start=1):
        self._annotation = annotation
        self._sequence = sequence
        self._seqstart = sequence_start
    
    @property
    def sequence(self):
        return self._sequence
    
    @property
    def annotation(self):
        return self._annotation
    
    def __getitem__(self, index):
        if isinstance(index, Annotation):
            # Concatenate subsequences for each location of the feature
            # Start by creating an empty sequence
            sub_seq = self._sequence.copy(new_seq_code=np.array([]))
            locs = index._get_location()
            for loc in locs:
                slice_start = loc.first - self._seqstart
                # +1 due to exclusive stop
                slice_stop = loc.last - self._seqstart +1
                slice_index = slice(slice_start, slice_stop)
                add_seq = self._sequence.__getitem__(slice_index)
                if loc.strand == Location.Strand.REVERSE:
                    add_seq = add_seq.reverse().complement()
                sub_seq += add_seq
                return sub_seq


