# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence"
__author__ = "Patrick Kunzmann"
__all__ = ["Location", "Feature", "Annotation", "AnnotatedSequence"]

import copy
import numbers
import sys
from enum import Enum, Flag, auto
import numpy as np
from biotite.copyable import Copyable


class Location:
    """
    A :class:`Location` defines at which base(s)/residue(s) a feature is
    located.

    A feature can have multiple :class:`Location` instances if multiple
    locations are joined.

    Objects of this class are immutable.

    Parameters
    ----------
    first : int
        Starting base or residue position of the feature.
    last : int
        Inclusive ending base or residue position of the feature.
    strand : Strand
        The strand direction.
        Always :attr:`Strand.FORWARD` for peptide features.
    defect : Defect
        A possible defect of the location.

    Attributes
    ----------
    first, last, strand, defect
        Same as the parameters.
    """

    class Defect(Flag):
        """
        This enum type describes location defects.

        A location has a defect, when the feature itself is not directly
        located in the range of the first to the last base.

           - **NONE** - No location defect
           - **MISS_LEFT** - A part of the feature has been truncated
             before the first base/residue of the :class:`Location`
             (probably by indexing an :class:`Annotation` object)
           - **MISS_RIGHT** - A part of the feature has been truncated
             after the last base/residue of the :class:`Location`
             (probably by indexing an :class:`Annotation` object)
           - **BEYOND_LEFT** - The feature starts at an unknown position
             before the first base/residue of the :class:`Location`
           - **BEYOND_RIGHT** - The feature ends at an unknown position
             after the last base/residue of the :class:`Location`
           - **UNK_LOC** - The exact position is unknown, but it is at a
             single base/residue between the first and last residue of
             the :class:`Location`, inclusive
           - **BETWEEN** - The position is between to consecutive
             bases/residues.
        """

        NONE = 0
        MISS_LEFT = auto()
        MISS_RIGHT = auto()
        BEYOND_LEFT = auto()
        BEYOND_RIGHT = auto()
        UNK_LOC = auto()
        BETWEEN = auto()

    class Strand(Enum):
        """
        This enum type describes the strand of the feature location.
        This is not relevant for protein sequence features.
        """

        FORWARD = auto()
        REVERSE = auto()

    def __init__(self, first, last, strand=Strand.FORWARD, defect=Defect.NONE):
        if first > last:
            raise ValueError(
                "The first position cannot be higher than the last position"
            )
        self._first = first
        self._last = last
        self._strand = strand
        self._defect = defect

    def __repr__(self):
        """Represent Location as a string for debugging."""
        return (
            f"Location({self._first}, {self._last}, strand={'Location.' + str(self._strand)}, "
            f"defect={'Location.' + str(self._defect)})"
        )

    @property
    def first(self):
        return self._first

    @property
    def last(self):
        return self._last

    @property
    def strand(self):
        return self._strand

    @property
    def defect(self):
        return self._defect

    def __str__(self):
        string = "{:d}-{:d}".format(self.first, self.last)
        if self.strand == Location.Strand.FORWARD:
            string = string + " >"
        else:
            string = "< " + string
        return string

    def __eq__(self, item):
        if not isinstance(item, Location):
            return False
        return (
            self.first == item.first
            and self.last == item.last
            and self.strand == item.strand
            and self.defect == item.defect
        )

    def __hash__(self):
        return hash((self._first, self._last, self._strand, self._defect))


class Feature(Copyable):
    """
    This class represents a single sequence feature, for example from a
    GenBank feature table.
    A feature describes a functional part of a sequence.
    It consists of a feature key, describing the general class of the
    feature, at least one location, describing its position on the
    reference, and qualifiers, describing the feature in detail.

    Objects of this class are immutable.

    Parameters
    ----------
    key : str
        The name of the feature class, e.g. *gene*, *CDS* or
        *regulatory*.
    locs : iterable object of Location
        A list of feature locations. In most cases this list will only
        contain one location, but multiple ones are also possible for
        example in eukaryotic CDS (due to splicing).
    qual : dict, optional
        Maps feature qualifiers to their corresponding values.
        The keys are always strings. A value is either a string or
        ``None`` if the qualifier key do not has a value.
        If key has multiple values, the values are separated by a
        line break.

    Attributes
    ----------
    key : str
        The name of the feature class, e.g. *gene*, *CDS* or
        *regulatory*.
    locs : iterable object of Location
        A list of feature locations. In most cases this list will only
        contain one location, but multiple ones are also possible for
        example in eukaryotic CDS (due to splicing).
    qual : dict
        Maps feature qualifiers to their corresponding values.
        The keys are always strings. A value is either a string or
        ``None`` if the qualifier key do not has a value.
        If key has multiple values, the values are separated by a
        line break.
    """

    def __init__(self, key, locs, qual=None):
        self._key = key
        if len(locs) == 0:
            raise ValueError("A feature must have at least one location")
        self._locs = frozenset(locs)
        self._qual = copy.deepcopy(qual) if qual is not None else {}

    def __repr__(self):
        """Represent Feature as a string for debugging."""
        return f'Feature("{self._key}", [{", ".join([loc.__repr__() for loc in self.locs])}], qual={self._qual})'

    def get_location_range(self):
        """
        Get the minimum first base/residue and maximum last base/residue
        of all feature locations.

        This can be used to create a location, that spans all of the
        feature's locations.

        Returns
        -------
        first : int
            The minimum first base/residue of all locations.
        last : int
            The maximum last base/residue of all locations.
        """
        first = np.min([loc.first for loc in self._locs])
        last = np.max([loc.last for loc in self._locs])
        return first, last

    def __eq__(self, item):
        if not isinstance(item, Feature):
            return False
        return (
            self._key == item._key
            and self._locs == item._locs
            and self._qual == item._qual
        )

    def __lt__(self, item):
        if not isinstance(item, Feature):
            return False
        first, last = self.get_location_range()
        it_first, it_last = item.get_location_range()
        # The first base/residue is most significant,
        # if it is equal for both features, look at last base/residue
        if first < it_first:
            return True
        elif first > it_first:
            return False
        else:  # First is equal
            return last > it_last

    def __gt__(self, item):
        if not isinstance(item, Feature):
            return False
        first, last = self.get_location_range()
        it_first, it_last = item.get_location_range()
        # The first base/residue is most significant,
        # if it is equal for both features, look at last base/residue
        if first > it_first:
            return True
        elif first < it_first:
            return False
        else:  # First is equal
            return last < it_last

    @property
    def key(self):
        return self._key

    @property
    def locs(self):
        return copy.copy(self._locs)

    @property
    def qual(self):
        return copy.copy(self._qual)

    def __hash__(self):
        return hash((self._key, self._locs, frozenset(self._qual.items())))


class Annotation(Copyable):
    """
    An :class:`Annotation` is a set of features belonging to one
    sequence.

    Its advantage over a simple list is the base/residue position based
    indexing:
    When using slice indices in Annotation objects, a subannotation is
    created, containing copies of all :class:`Feature` objects whose
    first and last base/residue are in range of the slice.
    If the slice starts after the first base/residue or/and the slice
    ends before the last residue, the position out of range is set to
    the boundaries of the slice (the :class:`Feature` is truncated).
    In this case the :class:`Feature` obtains the
    :attr:`Location.Defect.MISS_LEFT` and/or
    :attr:`Location.Defect.MISS_RIGHT` defect.
    The third case occurs when a :class:`Feature` starts after the slice
    ends or a :class:`Feature` ends before the slice starts.
    In this case the :class:`Feature` will not appear in the
    subannotation.

    The start or stop position in the slice indices can be omitted, then
    the subannotation will include all features from the start or up to
    the stop, respectively. Step values are ignored.
    The stop values are still exclusive, i.e. the subannotation will
    contain a not truncated :class:`Feature` only if its last
    base/residue is smaller than the stop value of the slice.

    Integers or other index types are not supported. If you want to
    obtain the :class:`Feature` instances from the :class:`Annotation`
    you need to  iterate over it.
    The iteration has no defined order.
    Alternatively, you can obtain a copy of the internal
    :class:`Feature` set via :func:`get_features()`.

    Multiple :class:`Annotation` objects can be concatenated to one
    :class:`Annotation` object using the '+' operator.
    Single :class:`Feature` instances can be added this way, too.
    If a feature is present in both :class:`Annotation` objects, the
    resulting :class:`Annotation` will contain this feature twice.

    Parameters
    ----------
    features : iterable object of Feature, optional
        The features to create the :class:`Annotation` from. if not
        provided, an empty :class:`Annotation` is created.

    Examples
    --------
    Creating an annotation from a feature list:

    >>> feature1 = Feature("CDS", [Location(-10, 30 )], qual={"gene" : "test1"})
    >>> feature2 = Feature("CDS", [Location(20,  50 )], qual={"gene" : "test2"})
    >>> annotation = Annotation([feature1, feature2])
    >>> for f in sorted(list(annotation)):
    ...     print(f.qual["gene"], "".join([str(loc) for loc in f.locs]))
    test1 -10-30 >
    test2 20-50 >

    Merging two annotations and a feature:

    >>> feature3 = Feature("CDS", [Location(100, 130 )], qual={"gene" : "test3"})
    >>> feature4 = Feature("CDS", [Location(150, 250 )], qual={"gene" : "test4"})
    >>> annotation2 = Annotation([feature3, feature4])
    >>> feature5 = Feature("CDS", [Location(-50, 200 )], qual={"gene" : "test5"})
    >>> annotation = annotation + annotation2 + feature5
    >>> for f in sorted(list(annotation)):
    ...     print(f.qual["gene"], "".join([str(loc) for loc in f.locs]))
    test5 -50-200 >
    test1 -10-30 >
    test2 20-50 >
    test3 100-130 >
    test4 150-250 >

    Location based indexing, note the defects:

    >>> annotation = annotation[40:150]
    >>> for f in sorted(list(annotation)):
    ...     gene = f.qual["gene"]
    ...     loc_str = "".join([f"{loc}    {loc.defect}" for loc in f.locs])
    ...     print(gene, loc_str)
    test5 40-149 >    Defect.MISS_LEFT|MISS_RIGHT
    test2 40-50 >    Defect.MISS_LEFT
    test3 100-130 >    Defect.NONE
    """

    def __init__(self, features=None):
        if features is None:
            self._features = set()
        else:
            self._features = set(features)

    def __repr__(self):
        """Represent Annotation as a string for debugging."""
        return (
            f"Annotation([{', '.join([feat.__repr__() for feat in self._features])}])"
        )

    def __copy_create__(self):
        return Annotation(self._features)

    def get_features(self):
        """
        Get a copy of the internal feature set.

        Returns
        -------
        feature_list : list of Feature
            A copy of the internal feature set.
        """
        return copy.copy(self._features)

    def add_feature(self, feature):
        """
        Add a feature to the annotation.

        Parameters
        ----------
        feature : Feature
            Feature to be added.
        """
        if not isinstance(feature, Feature):
            raise TypeError(
                f"Only 'Feature' objects are supported, not {type(feature).__name__}"
            )
        self._features.add(feature)

    def get_location_range(self):
        """
        Get the range of feature locations,
        i.e. the first and exclusive last base/residue.

        Returns
        -------
        int : start
            Start location.
        int : stop
            Exclusive stop location.
        """
        first = sys.maxsize
        last = -sys.maxsize
        for feature in self._features:
            for loc in feature.locs:
                if loc.first < first:
                    first = loc.first
                if loc.last > last:
                    last = loc.last
        # Exclusive stop -> +1
        return first, last + 1

    def del_feature(self, feature):
        """
        Delete a feature from the annotation.

        Parameters
        ----------
        feature : Feature
            Feature to be removed.

        Raises
        ------
        KeyError
            If the feature is not in the annotation
        """
        self._features.remove(feature)

    def __add__(self, item):
        if isinstance(item, Annotation):
            return Annotation(self._features | item._features)
        elif isinstance(item, Feature):
            return Annotation(self._features | set([item]))
        else:
            raise TypeError(
                f"Only 'Feature' and 'Annotation' objects are supported, "
                f"not {type(item).__name__}"
            )

    def __iadd__(self, item):
        if isinstance(item, Annotation):
            self._features |= item._features
        elif isinstance(item, Feature):
            self._features.add(item)
        else:
            raise TypeError(
                f"Only 'Feature' and 'Annotation' objects are supported, "
                f"not {type(item).__name__}"
            )
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            # If no start or stop index is given, include all
            if index.start is None:
                i_first = -sys.maxsize
            else:
                i_first = index.start
            if index.stop is None:
                i_last = sys.maxsize
            else:
                i_last = index.stop - 1

            sub_annot = Annotation()
            for feature in self:
                locs_in_scope = []
                for loc in feature.locs:
                    # Always true for maxsize values
                    # in case no start or stop index is given
                    if loc.first <= i_last and loc.last >= i_first:
                        # The location is at least partly in the
                        # given location range
                        # Handle defects
                        first = loc.first
                        last = loc.last
                        defect = loc.defect
                        if loc.first < i_first:
                            defect |= Location.Defect.MISS_LEFT
                            first = i_first
                        if loc.last > i_last:
                            defect |= Location.Defect.MISS_RIGHT
                            last = i_last
                        locs_in_scope.append(Location(first, last, loc.strand, defect))
                if len(locs_in_scope) > 0:
                    # The feature is present in the new annotation
                    # if any of the original locations is in the new
                    # scope
                    new_feature = Feature(
                        key=feature.key, locs=locs_in_scope, qual=feature.qual
                    )
                    sub_annot.add_feature(new_feature)
            return sub_annot
        else:
            raise TypeError(f"'{type(index).__name__}' instances are invalid indices")

    def __delitem__(self, item):
        if not isinstance(item, Feature):
            raise TypeError(
                f"Only 'Feature' objects are supported, not {type(item).__name__}"
            )
        self.del_feature(item)

    def __iter__(self):
        return self._features.__iter__()

    def __contains__(self, item):
        return item in self._features

    def __eq__(self, item):
        if not isinstance(item, Annotation):
            return False
        return self._features == item._features

    def __len__(self):
        return len(self._features)


class AnnotatedSequence(Copyable):
    """
    An :class:`AnnotatedSequence` is a combination of a
    :class:`Sequence` and an :class:`Annotation`.

    Indexing an :class:`AnnotatedSequence` with a slice returns another
    :class:`AnnotatedSequence` with the corresponding subannotation and
    a sequence start corrected subsequence, i.e. indexing starts at 1
    with the default sequence start 1.
    The sequence start in the newly created :class:`AnnotatedSequence`
    is the start of the slice.
    Furthermore, integer indices are allowed in which case the
    corresponding symbol of the sequence is returned (also sequence
    start corrected).
    In both cases the index must be in range of the sequence, e.g. if
    sequence start is 1, index 0 is not allowed.
    Negative indices do not mean indexing from the end of the sequence,
    in contrast to the behavior in :class:`Sequence` objects.
    Both index types can also be used to modify the sequence.

    Another option is indexing with a :class:`Feature` (preferably from the
    :class:`Annotation` in the same :class:`AnnotatedSequence`).
    In this case a sequence, described by the location(s) of the
    :class:`Feature`, is returned.
    When using a :class:`Feature` for setting an
    :class:`AnnotatedSequence` with a sequence, the new sequence is
    replacing the locations of the
    :class:`Feature`.
    Note the the replacing sequence must have the same length as the
    sequence of the :class:`Feature` index.

    Parameters
    ----------
    annotation : Annotation
        The annotation corresponding to `sequence`.
    sequence : Sequence
        The sequence.
        Usually a :class:`NucleotideSequence` or
        :class:`ProteinSequence`.
    sequence_start : int, optional
        By default, the first symbol of the sequence is corresponding
        to location 1 of the features in the annotation. The location
        of the first symbol can be changed by setting this parameter.
        Negative values are not supported yet.

    Attributes
    ----------
    annotation : Annotation
        The annotation corresponding to `sequence`.
    sequence : Sequence
        The represented sequence.
    sequence_start : int
        The location of the first symbol in the sequence.

    See Also
    --------
    Annotation : An annotation separated from a sequence.
    Sequence : A sequence separated from an annotation.

    Examples
    --------
    Creating an annotated sequence

    >>> sequence = NucleotideSequence("ATGGCGTACGATTAGAAAAAAA")
    >>> feature1 = Feature("misc_feature", [Location(1,2), Location(11,12)],
    ...                    {"note" : "walker"})
    >>> feature2 = Feature("misc_feature", [Location(16,22)], {"note" : "poly-A"})
    >>> annotation = Annotation([feature1, feature2])
    >>> annot_seq = AnnotatedSequence(annotation, sequence)
    >>> print(annot_seq.sequence)
    ATGGCGTACGATTAGAAAAAAA
    >>> for f in sorted(list(annot_seq.annotation)):
    ...     print(f.qual["note"])
    walker
    poly-A

    Indexing with integers, note the sequence start correction

    >>> print(annot_seq[2])
    T
    >>> print(annot_seq.sequence[2])
    G

    indexing with slices

    >>> annot_seq2 = annot_seq[:16]
    >>> print(annot_seq2.sequence)
    ATGGCGTACGATTAG
    >>> for f in annot_seq2.annotation:
    ...     print(f.qual["note"])
    walker

    Indexing with features

    >>> print(annot_seq[feature1])
    ATAT
    >>> print(annot_seq[feature2])
    AAAAAAA
    >>> print(annot_seq.sequence)
    ATGGCGTACGATTAGAAAAAAA
    >>> annot_seq[feature1] = NucleotideSequence("CCCC")
    >>> print(annot_seq.sequence)
    CCGGCGTACGCCTAGAAAAAAA
    """

    def __init__(self, annotation, sequence, sequence_start=1):
        self._annotation = annotation
        self._sequence = sequence
        self._seqstart = sequence_start

    def __repr__(self):
        """Represent AnnotatedSequence as a string for debugging."""
        return (
            f"AnnotatedSequence({self._annotation.__repr__()}, {self._sequence.__repr__()}, "
            f"sequence_start={self._seqstart})"
        )

    @property
    def sequence_start(self):
        return self._seqstart

    @property
    def sequence(self):
        return self._sequence

    @property
    def annotation(self):
        return self._annotation

    def __copy_create__(self):
        return AnnotatedSequence(
            self._annotation.copy(), self._sequence.copy, self._seqstart
        )

    def reverse_complement(self, sequence_start=1):
        """
        Create the reverse complement of the annotated sequence.

        This method accurately converts the position and the strand of
        the annotation.
        The information on the sequence start is lost.

        Parameters
        ----------
        sequence_start : int, optional
            The location of the first symbol in the reverse complement
            sequence.

        Returns
        -------
        rev_sequence : Sequence
            The reverse complement of the annotated sequence.
        """
        rev_seqstart = sequence_start

        rev_sequence = self._sequence.reverse().complement()

        seq_len = len(self._sequence)
        rev_features = []
        for feature in self._annotation:
            rev_locs = []
            for loc in feature.locs:
                # Transform location to the reverse complement strand
                # (seq_len-1) -> last sequence index
                # (loc.last-self._seqstart) -> location to index
                # ... + rev_seqstart -> index to location
                rev_loc_first = (
                    (seq_len - 1) - (loc.last - self._seqstart) + rev_seqstart
                )
                rev_loc_last = (
                    (seq_len - 1) - (loc.first - self._seqstart) + rev_seqstart
                )

                if loc.strand == Location.Strand.FORWARD:
                    rev_loc_strand = Location.Strand.REVERSE
                else:
                    rev_loc_strand = Location.Strand.FORWARD

                rev_loc_defect = Location.Defect.NONE
                if loc.defect & Location.Defect.MISS_LEFT:
                    rev_loc_defect |= Location.Defect.MISS_RIGHT
                if loc.defect & Location.Defect.MISS_RIGHT:
                    rev_loc_defect |= Location.Defect.MISS_LEFT
                if loc.defect & Location.Defect.BEYOND_RIGHT:
                    rev_loc_defect |= Location.Defect.BEYOND_LEFT
                if loc.defect & Location.Defect.BEYOND_LEFT:
                    rev_loc_defect |= Location.Defect.BEYOND_RIGHT
                if loc.defect & Location.Defect.UNK_LOC:
                    rev_loc_defect |= Location.Defect.UNK_LOC
                if loc.defect & Location.Defect.BETWEEN:
                    rev_loc_defect |= Location.Defect.BETWEEN

                rev_locs.append(
                    Location(
                        rev_loc_first, rev_loc_last, rev_loc_strand, rev_loc_defect
                    )
                )
            rev_features.append(Feature(feature.key, rev_locs, feature.qual))

        return AnnotatedSequence(Annotation(rev_features), rev_sequence, rev_seqstart)

    def __getitem__(self, index):
        if isinstance(index, Feature):
            # Concatenate subsequences for each location of the feature
            locs = index.locs
            if len(locs) == 0:
                raise ValueError("Feature does not contain any locations")
            # Start by creating an empty sequence
            sub_seq = self._sequence.copy(new_seq_code=np.array([]))
            # Locations need to be sorted, as otherwise the locations
            # chunks would be merged in the wrong order
            # The order depends on whether the locs are on the forward
            # or reverse strand
            strand = None
            for loc in locs:
                if loc.strand == strand:
                    pass
                elif strand is None:
                    strand = loc.strand
                else:  # loc.strand != strand
                    raise ValueError(
                        "All locations of the feature must have the same "
                        "strand direction"
                    )
            if strand == Location.Strand.FORWARD:
                sorted_locs = sorted(locs, key=lambda loc: loc.first)
            else:
                sorted_locs = sorted(locs, key=lambda loc: loc.last, reverse=True)
            # Merge the sequences corresponding to the ordered locations
            for loc in sorted_locs:
                slice_start = loc.first - self._seqstart
                # +1 due to exclusive stop
                slice_stop = loc.last - self._seqstart + 1
                add_seq = self._sequence[slice_start:slice_stop]
                if loc.strand == Location.Strand.REVERSE:
                    add_seq = add_seq.reverse().complement()
                sub_seq += add_seq
            return sub_seq

        elif isinstance(index, slice):
            # Sequence start correction
            if index.start is None:
                seq_start = 0
            else:
                if index.start < self._seqstart:
                    raise IndexError(
                        f"The start of the index ({index.start}) is lower "
                        f"than the start of the sequence ({self._seqstart})"
                    )
                seq_start = index.start - self._seqstart
            if index.stop is None:
                seq_stop = len(self._sequence)
                index = slice(index.start, seq_stop, index.step)
            else:
                seq_stop = index.stop - self._seqstart
            # New value for the sequence start, value is base position
            if index.start is None:
                rel_seq_start = self._seqstart
            else:
                rel_seq_start = index.start
            return AnnotatedSequence(
                self._annotation[index],
                self._sequence[seq_start:seq_stop],
                rel_seq_start,
            )

        elif isinstance(index, numbers.Integral):
            return self._sequence[index - self._seqstart]

        else:
            raise TypeError(f"'{type(index).__name__}' instances are invalid indices")

    def __setitem__(self, index, item):
        if isinstance(index, Feature):
            # Item must be sequence
            # with length equal to sum of location lengths
            sub_seq = item
            sub_seq_i = 0
            for loc in index.locs:
                slice_start = loc.first - self._seqstart
                # +1 due to exclusive stop
                slice_stop = loc.last - self._seqstart + 1
                interval_size = slice_stop - slice_start
                self._sequence[slice_start:slice_stop] = sub_seq[
                    sub_seq_i : sub_seq_i + interval_size
                ]
                sub_seq_i += interval_size
        elif isinstance(index, slice):
            # Sequence start correction
            if index.start is None:
                seq_start = 0
            else:
                seq_start = index.start - self._seqstart
            if index.stop is None:
                seq_stop = len(self._sequence)
            else:
                seq_stop = index.stop - self._seqstart
            # Item is a Sequence
            self._sequence[seq_start:seq_stop] = item
        elif isinstance(index, numbers.Integral):
            # Item is a symbol
            self._sequence[index - self._seqstart] = item
        else:
            raise TypeError(f"'{type(index).__name__}' instances are invalid indices")

    def __eq__(self, item):
        if not isinstance(item, AnnotatedSequence):
            return False
        return (
            self.annotation == item.annotation
            and self.sequence == item.sequence
            and self._seqstart == item._seqstart
        )
