# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package

from .sequence import Sequence
import copy
from enum import Flag, auto

__all__ = ["Feature", "Annotation", "AnnotatedSequence"]


class LocDefect(Flag):
    MISS_LEFT    = auto()
    MISS_RIGHT   = auto()
    MISS_INTERN  = auto()
    BEYOND_LEFT  = auto()
    BEYOND_RIGHT = auto()
    UNK_LOC      = auto()
    

class Feature(object):
    
    def __init__(self, name, loc, loc_defects=0):
        self._name = name
        self._loc = copy.copy(pos)
        self._loc_defects = pos_defects
        self._subfeatures = []
        
    def add_subfeature(self, feature):
        pass
    
    def del_subfeature(self, feature):
        pass


class Annotation(object):
    
    def __init__(self, features=[]):
        self._features = copy.copy(features)
        
    def get_features(pos=None):
        pass
    
    def add_feature(feature):
        pass

    def __getitem__(self, index):
        pass
    
    def __setitem__(self, index, value):
        pass
    
    def __add__(self, annotation):
        pass
    
    
class AnnotatedSequence(object):
    
    def __init__(self, annotation, sequence):
        self._annotation = annotation
        self._sequence = sequence
    
    @property
    def sequence(self):
        return self._sequence
    
    @property
    def annotation(self):
        return self._annotation
    
    def __getitem__(self, index):
        pass
    
    def __setitem__(self, index, value):
        pass
    
    def __add__(self, annotation):
        pass

