From A to T - The Sequence subpackage
-------------------------------------

``sequence`` is a Biopython subpackage concerning maybe the most popular
in computational biology: sequences. The instantiation can be quite simple as:

.. code-block:: python

   >>> import biopython.sequence as seq
   >>> dna = seq.NucleotideSequence("AACTGCTA")
   >>> print(dna)
   AACTGCTA

This example shows `NucleotideSequence` which is a subclass of the abstract
base class `Sequence`. A `NucleotideSequence` accepts a list of strings,
where each string can be 'A', 'C', 'G' or 'T'. Each of this letters is called
a *symbol*.

In general the sequence implementation in Biopython allows for
*Sequences of anything*. This means any Python object can be used as a symbol
in a `Sequence`, as long as the object is part of the `Alphabet` of the
particular `Sequence`. An `Alphabet` object simply represents a list of objects
that are allowed to occur in a `Sequence`. The following figure shows how the
symbols are stored in a sequence.

.. image:: /static/assets/figures/symbol_encoding.svg

When setting the `Sequence` object with a sequence, the `Alphabet` of the
`Sequence` encodes each symbol in the input sequence into a so called
*symbol code*. The encoding process is quite simple: A symbol *s* is at index
*i* in the list of allowed symbols in the alphabet, so the symbol code for *s*
is *i*. If *s* is not in the alphabet, an `AlphabetError` is raised.
The array of symbol codes, that arises from encoding the input sequence, is
called *sequence code*. This sequence code is now stored in an internal
integer `ndarray` in the `Sequence` object.
The sequence code is now accessed via the `code` attribute, the corresponding
symbols via the `symbols` attribute.

This approach has multiple advantages:

   - Ability to create *sequences of anything*
   - Sequence utility functions (searches, alignments,...) usually do not
     care for specific sequence type, since they work with the internal
     sequence code
   - Integer type for sequence code is only as large as the alphabet requests
   - Sequence codes can be directly used as substitution matrix indices in
     alignments

Effectively, this means a potential `Sequence` subclass could work the
following way:

.. code-block:: python

   >>> sequence = NewSequence(["Foo", "Bar", 42, "Foo", "Foo", 42])
   >>> print(sequence.get_alphabet())
   ['Foo', 'Bar', 42]
   >>> print(sequence.symbols)
   ["Foo", "Bar", 42, "Foo", "Foo", 42]
   >>> print(sequence.code)
   [0 1 2 0 0 2]


From DNA to Protein
^^^^^^^^^^^^^^^^^^^

Sequence search
^^^^^^^^^^^^^^^

Sequence alignments
^^^^^^^^^^^^^^^^^^^
