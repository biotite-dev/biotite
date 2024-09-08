:sd_hide_title: true

.. include:: /tutorial/preamble.rst

#######################
``sequence`` subpackage
#######################

From A to T - The ``sequence`` subpackage
=========================================

.. currentmodule:: biotite.sequence

:mod:`biotite.sequence` is a *Biotite* subpackage concerning maybe the
most prevalent data type in bioinformatics: sequences.

Sequences are represented by :class:`Sequence` objects, with different
subclasses for different types of sequences.
For example, to present DNA one would use a :class:`NucleotideSequence`.

.. jupyter-execute::

    import biotite.sequence as seq

    dna = seq.NucleotideSequence("AACTGCTA")
    print(dna)

Each type of sequence allows only for a certain set of symbols, which is
controlled by the :class:`Alphabet` of a sequence.
For an unambiguous DNA sequence, the alphabet comprises the four nucleobases.

.. jupyter-execute::

    print(dna.alphabet)

.. toctree::
    :maxdepth: 1
    :hidden:

    types
    encoding
    io
    align_optimal
    align_heuristic
    align_multiple
    profiles
    annotations