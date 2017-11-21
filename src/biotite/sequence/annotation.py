# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from .sequence import Sequence
from ..copyable import Copyable
import copy
import sys
from enum import Flag, Enum, auto

__all__ = ["Feature", "Annotation", "AnnotatedSequence"]



class Location():
    
    class Defect(Flag):
        MISS_LEFT    = auto()
        MISS_RIGHT   = auto()
        MISS_INTERN  = auto()
        BEYOND_LEFT  = auto()
        BEYOND_RIGHT = auto()
        UNK_LOC      = auto()
        BETWEEN      = auto()

    class Strand(Enum):
        FORWARD = 1
        REVERSE = -1
    
    def __init__(self, first, last, strand=Strand.FORWARD,
                 defect=Defect(0)):
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
    
    def __init__(self, name, locs):
        self._name = name
        self._locs = copy.deepcopy(locs)
        self._subfeatures = []
    
    def __copy_create__(self):
        return Feature(self._name, self._locs)
    
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
    
    def get_location(self):
        return copy.deepcopy(self._locs)
    
    def get_name(self):
        return self._name


class Annotation(object):
    
    def __init__(self, features=[]):
        self._features = copy.copy(features)
        self._feature_dict = {}
        for feature in features:
            self._feature_dict[feature.get_name()] = feature
        
    def get_features(pos=None):
        return copy.copy(self._features)
    
    def add_feature(self, feature):
        self._features.append(feature.copy())
        self._feature_dict[feature.get_name()] = feature
    
    def del_feature(self, feature):
        self._features.remove(feature)
        del self._feature_dict[feature.get_name()]
    
    def release_from(self, feature):
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
        if isinstance(index, str):
            # Usage as a dictionary
            return self._feature_dict[index]
        elif isinstance(index, slice):
            i_first = index.start
            # If no start or stop index is given, include all
            if i_first is None:
                i_first = -sys.maxsize
            i_last = index.stop -1
            if i_last is None:
                i_last = sys.maxsize
            sub_annot = Annotation()
            for feature in self:
                locs = feature.get_location()
                in_scope = False
                #defect = LocDefect(0)
                for loc in locs:
                    first = loc.first
                    last = loc.last
                    # Always true for maxsize values
                    # in case no start or stop index is given
                    if first <= i_last and last >= i_first:
                        in_scope = True
                    ### Handle defects ###
                if in_scope:
                    sub_annot.add_feature(feature.copy())
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


