.. include:: /tutorial/preamble.rst

Excursion: Symbol encoding
==========================

.. currentmodule:: biotite.sequence

As you have seen in the previous chapter, :class:`Sequence` objects may support
a wide variety of *Python* objects as symbols.
To still ensure a *NumPy*-boosted performance of functions acting upon a
:class:`Sequence`, an underlying :class:`Alphabet` encodes each *symbol* into
an integer, the so called *code*.

.. figure:: /static/assets/figures/symbol_encoding.png
   :alt: Symbol encoding in Biotite
   :scale: 50%

   Taken from
   `Kunzmann & Hamacher 2018 <https://doi.org/10.1186/s12859-018-2367-z>`_
   licensed under `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.

In short, the :class:`Alphabet` maps a symbol to the index of that symbol in
the alphabet.
Encoding and decoding is done by the the :meth:`Alphabet.encode` and
:meth:`Alphabet.decode` methods, respectively.

.. jupyter-execute::

    import biotite.sequence as seq

    alph = seq.NucleotideSequence.unambiguous_alphabet()
    print("Allowed symbols:", alph.get_symbols())
    print("G is encoded to", alph.encode("G"))
    print("2 is decoded to", alph.decode(2))

A sequence actually does not store the symbols themselves, but only the code
in a *Numpy* array.
The code is only decoded into symbols when required, for example when the
sequence is converted into a string.

.. jupyter-execute::

    dna = seq.NucleotideSequence("AACTGCTA")
    print("Actually stored:", dna.code)
    print("Calculated on-the-fly:", dna.symbols)

As most functions throughout :mod:`biotite.sequence` work directly on the code,
they usually work on any type of sequence.

Most users will never need to work with the code directly.
However, if you want to implement a new function, the recommended approach is
to use the code, as this ensures compatibility with all types of sequences
and enables harnessing the power of *NumPy*.