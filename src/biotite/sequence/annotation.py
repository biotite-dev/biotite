# Copyright 2017-2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from .sequence import Sequence
from ..copyable import Copyable
import copy
import sys
import numpy as np
from enum import IntEnum

__all__ = ["Location", "Feature", "Annotation", "AnnotatedSequence"]


class Location():
    """
    A `Location` defines at which base(s)/residue(s) a feature is
    located.
    
    A feature can have multiple `Location` instances if multiple
    locations are joined.
    
    Attributes
    ----------
    first : int
        Starting base or residue position of the feature.
    last : int
        Inclusive ending base or residue position of the feature.
    strand : Strand
        The strand direction. Always `Strand.FORWARD` for peptide
        features.
    defect : Defect
        A possible defect of the location.
    """
    
    class Defect(IntEnum):
        """
        This enum type describes location defects.
        
        A location has a defect, when the feature itself is not directly
        located in the range of the first to the last base.
        
           - **NONE** - No location defect
           - **MISS_LEFT** - A part of the feature has been truncated
             before the first base/residue of the `Location`
             (probably by indexing an `Annotation` object)
           - **MISS_RIGHT** - A part of the feature has been truncated
             after the last base/residue of the `Location`
             (probably by indexing an `Annotation` object)
           - **BEYOND_LEFT** - The feature starts at an unknown position
             before the first base/residue of the `Location`
           - **BEYOND_RIGHT** - The feature ends at an unknown position
             after the last base/residue of the `Location`
           - **UNK_LOC** - The exact position is unknown, but it is at a
             single base/residue between the first and last residue of
             the `Location`, inclusive
           - **BETWEEN** - The position is between to consecutive
             bases/residues.
        """
        NONE         = 0
        MISS_LEFT    = 1
        MISS_RIGHT   = 2
        BEYOND_LEFT  = 4
        BEYOND_RIGHT = 8
        UNK_LOC      = 16
        BETWEEN      = 32

    class Strand(IntEnum):
        """
        This enum type describes the strand of the feature location.
        This is not relevant for residue peptide features.
        """
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
    """
    This class represents a single sequence feature, e.g. from a
    GenBank feature table.
    
    Attributes
    ----------
    key : str
        The name of the feature class, e.g. *gene*, *CDS* or
        *regulatory*. 
    locs : list of Location
        A list of feature locations. In most cases this list will only
        contain one location, but multiple ones are also possible for
        example in eukaryotic CDS (due to splicing).
    qual : dict
        Maps GenBank feature qualifiers to their corresponding values.
        The keys and values are always strings.
    """
    
    def __init__(self, key, locs, qual={}):
        self._key = key
        self._locs = copy.deepcopy(locs)
        self._qual = copy.deepcopy(qual)
    
    def __copy_create__(self):
        return Feature(self._key, self._locs, self._qual)
    
    def __eq__(self):
        pass
    
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
    """
    An `Annotation` is a list of features belonging to one sequence.
    
    Its advantage over a simple list is the base/residue position based
    indexing:
    Rather than
    """
    
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
    
    def __add__(self, item):
        if not isinstance(item, Feature):
            raise TypeError("Only can add 'Feature' instances to annotation")
        self.copy().add_feature(item)
    
    def __delitem__(self, item):
        if not isinstance(item, Feature):
            raise TypeError("Only can delete 'Feature' instances "
                            "from annotation")
        self.del_feature(item)
    
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
                            loc.defect |= Location.Defect.MISS_LEFT
                            loc.first = i_first
                        if loc.last > i_last:
                            loc.defect |= Location.Defect.MISS_RIGHT
                            loc.last = i_last
                    sub_annot.add_feature(new_feature)
            return sub_annot
        else:
            raise TypeError("{:} instances are invalid indices"
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
    
    def __copy_create__(self):
        return AnnotatedSequence(
            self._annotation.copy(), self._sequence.copy, self._seqstart)
    
    def __getitem__(self, index):
        if isinstance(index, Feature):
            # Concatenate subsequences for each location of the feature
            # Start by creating an empty sequence
            sub_seq = self._sequence.copy(new_seq_code=np.array([]))
            for loc in index.locs:
                slice_start = loc.first - self._seqstart
                # +1 due to exclusive stop
                slice_stop = loc.last - self._seqstart +1
                add_seq = self._sequence[slice_start:slice_stop]
                if loc.strand == Location.Strand.REVERSE:
                    add_seq = add_seq.reverse().complement()
                sub_seq += add_seq
            return sub_seq
        elif isinstance(index, slice):
            seq_start = index.start - self._seqstart
            seq_stop = index.stop - self._seqstart
            return AnnotatedSequence(self._annotation[index],
                                     self._sequence[seq_start:seq_stop],
                                     self._seqstart)
        elif isinstance(index, int):
            return self._sequence[index - self._seqstart]
        else:
            raise TypeError("{:} instances are invalid indices"
                            .format(type(index).__name__))
    
    def __setitem__(self, index, item):
        if isinstance(index, Feature):
            # Item must be sequence
            # with length equal to sum of location lengths
            sub_seq = item
            sub_seq_i = 0
            for loc in index.locs:
                slice_start = loc.first - self._seqstart
                # +1 due to exclusive stop
                slice_stop = loc.last - self._seqstart +1
                interval_size = slice_stop - slice_start
                self._sequence[slice_start:slice_stop] \
                    = sub_seq[sub_seq_i : sub_seq_i + interval_size]
                sub_seq_i += interval_size
        elif isinstance(index, slice):
            seq_start = index.start - self._seqstart
            seq_stop = index.stop - self._seqstart
            # Item is a Sequence
            self._sequence[seq_start:seq_stop] = item
        elif isinstance(index, int):
            # Item is a symbol
            self._sequence[index - self._seqstart] = item
        else:
            raise TypeError("{:} instances are invalid indices"
                            .format(type(index).__name__))





