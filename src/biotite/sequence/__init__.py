# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for handling sequences.

A `Sequence` can be seen as a succession of symbols. The set of symbols,
that can occur in a sequence, is defined by an `Alphabet`. For
example, an unambiguous DNA sequence has an `Alphabet`, that includes
the 4 letters (strings) 'A', 'C', 'G' and 'T'. But furthermore, an
`Alphabet` can also contain any arbitrary Python object. If a `Sequence`
is created with at least a symbol, that is not in the given `Alphabet`,
an `AlphabetError` is raised.

Internally, a `Sequence` is saved as a `NumPy` `ndarray` of integer
values, where each integer represents a symbol in the `Alphabet`.
For example, 'A', 'C', 'G' and 'T' would be encoded into 0, 1, 2 and 3.
These integer values are called *symbol code*, the encoding of an entire
sequence of symbols is called *sequence code*.

The size of the *symbol code* type in the array is determined by the 
size of the `Alphabet`: If the `Alphabet` contains 256 symbols or less,
one byte is used per array element; if the `Alphabet` contains between
257 and 65536 symbols, two bytes are used, and so on.

This approach has multiple advantages:

    - Wider spectrum of what kind of objects can be represented by
      `Sequence` objects
    - Efficient memory usage and faster calculations due to
      alphabet-tailored *symbol code* type size
    - Partial C-acceleration due to usage of `ndarrays`
    - Most functions applied on `Sequence` objects are indifferent to
      the actual type of sequence.
    - *Symbol codes* are directly indices for substitution matrices in
      alignments

Besides the `Sequence` superclass, this subpackage contains the classes
`NucleotideSequence` and `ProteinSequence` in order to include the most
important biological sequence types. The class `GeneralSequence` allows
the usage of a custom `Alphabet` without the need to subclass 
`Sequence`.

Additionally, this subpackage provides support for sequence feature,
as for example used in GenBank files. A `Feature` stores its class
name, its qualifiers and locations. An `Annotation` is a froup of multiple
`Feataure` objects and offers convenient location based indexing.
An `AnnotatedSequence` combines an `Annotation` and a `Sequence`
"""

__author__ = "Patrick Kunzmann"

from .alphabet import *
from .search import *
from .seqtypes import *
from .sequence import *
from .codon import *
from .annotation import *