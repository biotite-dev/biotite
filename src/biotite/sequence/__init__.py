# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for handling sequences.

A :class:`Sequence` can be seen as a succession of symbols.
The set of symbols, that can occur in a sequence, is defined by an
:class:`Alphabet`.
For example, an unambiguous DNA sequence has an :class:`Alphabet`, that
includes the 4 letters (strings) ``'A'``, ``'C'``, ``'G'`` and ``'T'``.
But furthermore, an :class:`Alphabet` can also contain any immutable and
hashable Python object like :class:`int`, :class:`tuple`, etc.
If a :class:`Sequence` is created with at least a symbol,
that is not in the given :class:`Alphabet`, an :class:`AlphabetError` is
raised.

Internally, a :class:`Sequence` is saved as a *NumPy* :class:`ndarray`
of integer values, where each integer represents a symbol in the
:class:`Alphabet`.
For example, ``'A'``, ``'C'``, ``'G'`` and ``'T'`` would be encoded into
0, 1, 2 and 3, respectively.
These integer values are called *symbol code*, the encoding of an entire
sequence of symbols is called *sequence code*.

The size of the symbol code type in the array is determined by the 
size of the :class:`Alphabet`:
If the :class:`Alphabet` contains 256 symbols or less, one byte is used
per array element, between 257 and 65536 symbols, two bytes are used,
and so on.

This approach has multiple advantages:

    - Wider spectrum of what kind of objects can be represented by
      :class:`Sequence` objects
    - Efficient memory usage and faster calculations due to
      alphabet-tailored *symbol code* type size
    - C-acceleration due to usage of :class:`ndarray` objects
    - Most functions applied on :class:`Sequence` objects are
      indifferent to the actual type of sequence.
    - Symbol codes are directly indices for substitution matrices in
      alignments

The abstract :class:`Sequence` superclass cannot be instantiated
directly, as it does not define an :class:`Alphabet` by itself.
Instead usually the concrete subclasses :class:`NucleotideSequence`
(for DNA and RNA sequences) and :class:`ProteinSequence`
(for amino acid sequences) are used.
These classes have defined alphabets and provide additional sequence
type specific methods.
The class :class:`GeneralSequence` allows the usage of a custom
:class:`Alphabet` without the need to subclass :class:`Sequence`.

Additionally, this subpackage provides support for sequence features,
as used in e.g. GenBank or GFF files.
A :class:`Feature` stores its key name, its qualifiers and locations.
An :class:`Annotation` is a group of multiple :class:`Feataure` objects
and offers convenient location based indexing.
An :class:`AnnotatedSequence` combines an :class:`Annotation` and a
:class:`Sequence`.
"""

__name__ = "biotite.sequence"
__author__ = "Patrick Kunzmann"

from .alphabet import *
from .search import *
from .seqtypes import *
from .sequence import *
from .codon import *
from .annotation import *
from .profile import *
