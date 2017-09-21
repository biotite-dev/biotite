# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package

from .sequence import Sequence
from ..copyable import Copyable
import copy
from enum import Flag, Enum, auto

__all__ = ["Feature", "Annotation", "AnnotatedSequence"]


class LocDefect(Flag):
    MISS_LEFT    = auto()
    MISS_RIGHT   = auto()
    MISS_INTERN  = auto()
    BEYOND_LEFT  = auto()
    BEYOND_RIGHT = auto()
    UNK_LOC      = auto()


def Strand(Enum):
    FORWARD = 1
    REVERSE = -1
    

class Feature(Copyable):
    
    def __init__(self, name, loc, loc_defects=LocDefect(0)):
        self._name = name
        self._loc = copy.copy(loc)
        self._loc_defects = loc_defects
        self._subfeatures = []
        # True, if feature is already a subfeature or in an annotation
        self._in_use = False
    
    def __copy_create__(self):
        return Feature(self._name, self._loc, self._loc_defects)
    
    def __copy_fill__(self, clone):
        for feature in self._subfeatures:
            clone._subfeatures.append(feature.copy())
    
    def add_subfeature(self, feature):
        feature.use(True)
        self._subfeatures.append(feature)
    
    def del_subfeature(self, feature):
        self._subfeatures.remove(feature)
        feature.use(False)
    
    def del_all_subfeatures(self):
        for feature in self._subfeatures:
            feature.use(False)
        self._subfeatures = []
    
    def get_subfeatures(self):
        return copy.copy(self._subfeatures)
    
    def add_defect(self, loc_defect):
        self._loc_defects |= loc_defect
    
    def get_location(self):
        return copy.copy(self._loc)
    
    def get_name(self):
        return self._name
    
    def use(self, set_in_use):
        if self._in_use and set_in_use:
            raise ValueError("Feature is already in use")
        self._in_use = set_in_use


class Annotation(object):
    
    def __init__(self, features=[]):
        self._features = copy.copy(features)
        self._feature_dict = {}
        for feature in features:
            self._feature_dict[feature.get_name()] = feature
        
    def get_features(pos=None):
        return copy.copy(self._features)
    
    def add_feature(feature):
        feature.use(True)
        self._features.append(feature)
        self._feature_dict[feature.get_name()] = feature
    
    def del_feature(feature):
        self._features.remove(feature)
        del self._feature_dict[feature.get_name()]
        feature.use(False)
    
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
        elif isinstance(index, slice_index):
            i_first = index.start
            i_last = index.stop -1
            sub_annot = Annotation()
            for feature in self:
                locs = feature.get_location()
                in_scope = False
                #defect = LocDefect(0)
                for loc in locs:
                    first = loc[0]
                    last = loc[1]
                    if first <= i_last and last >= i_first:
                        in_scope = True
                    ### Handle defects ###
                if in_scope:
                    sub_annot.add_feature(feature)
            return sub_annot            
        else:
            raise TypeError("{:} instances are invalid Annotation indices"
                            .format(type(index).__name__))

"""
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
                # loc[1] is first base of location,
                # loc[2] is last base of location
                slice_start = loc[1] - self._seqstart
                # +1 due to exclusive stop
                slice_stop = loc[2] - self._seqstart +1
                slice_index = slice(slice_start, slice_stop)
                add_seq = self._sequence.__getitem__(slice_index)
                if loc[0] == Strand.REVERSE:
                    add_seq = add_seq.reverse().complement()
                sub_seq += add_seq
                return sub_seq
"""

