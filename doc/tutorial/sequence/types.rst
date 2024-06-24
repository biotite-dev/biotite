.. include:: /tutorial/preamble.rst

Types of sequences
==================

.. currentmodule:: biotite.sequence

*Biotite* provides predefined classes for the most common types of sequences:
nucleotide and amino acid sequences.
In the end of this chapter we will also have a look on how to create a sequence
with a custom alphabet.

Nucleotide sequences
--------------------
The class :class:`NucleotideSequence` represents DNA.
A :class:`NucleotideSequence` may use two different alphabets - an unambiguous
alphabet containing the letters ``'A'``, ``'C'``, ``'G'`` and ``'T'`` and an
ambiguous alphabet containing additionally the standard letters for
ambiguous nucleic bases.
A :class:`NucleotideSequence` determines automatically which alphabet
is required, unless an alphabet is specified. If you want to work with
RNA sequences you can use this class, too, you just need to replace
the ``'U'`` with ``'T'``.

.. jupyter-execute::

    import biotite.sequence as seq

    # Create a nucleotide sequence using a string
    # The constructor can take any iterable object (e.g. a list of symbols)
    seq1 = seq.NucleotideSequence("ACCGTATCAAG")
    print(seq1.get_alphabet())
    # Constructing a sequence with ambiguous nucleic bases
    seq2 = seq.NucleotideSequence("TANNCGNGG")
    print(seq2.get_alphabet())

The reverse complement of a DNA sequence is created by chaining the
:func:`Sequence.reverse()` and the
:func:`NucleotideSequence.complement()` method.

.. jupyter-execute::

    # Lower case characters are automatically capitalized
    seq1 = seq.NucleotideSequence("tacagtt")
    print("Original:", seq1)
    seq2 = seq1.reverse().complement()
    print("Reverse complement:", seq2)

Protein sequences
-----------------
The other central :class:`Sequence` type is :class:`ProteinSequence`.
It supports the letters for the 20 standard amino acids plus some
letters for ambiguous amino acids and a letter for a stop signal.
Furthermore, this class provides some utilities like
3-letter to 1-letter translation (and vice versa).

.. jupyter-execute::

    prot_seq = seq.ProteinSequence("BIQTITE")
    print("-".join([seq.ProteinSequence.convert_letter_1to3(symbol)
                    for symbol in prot_seq]))

A :class:`NucleotideSequence` can be translated into a
:class:`ProteinSequence` via the
:func:`NucleotideSequence.translate()` method.
By default, the method searches for open reading frames (ORFs) in the
3 frames of the sequence.
A 6-frame ORF search requires an
additional call of :func:`NucleotideSequence.translate()` with the
reverse complement of the sequence.
If you want to conduct a complete 1-frame translation of the sequence,
irrespective of any start and stop codons, set the parameter
:obj:`complete` to true.

.. jupyter-execute::

    dna = seq.NucleotideSequence("CATATGATGTATGCAATAGGGTGAATG")
    proteins, pos = dna.translate()
    for i in range(len(proteins)):
        print(
            f"Protein sequence {str(proteins[i])} "
            f"from base {pos[i][0]+1} to base {pos[i][1]}"
        )
    protein = dna.translate(complete=True)
    print("Complete translation:", str(protein))

The upper example uses the default :class:`CodonTable` instance.
This can be changed with the :obj:`codon_table` parameter.
A :class:`CodonTable` maps codons to amino acids and defines start
codons (both in symbol and code form).
A :class:`CodonTable` is mainly used in the
:func:`NucleotideSequence.translate()` method,
but can also be used to find the corresponding amino acid for a codon
and vice versa.

.. jupyter-execute::

    table = seq.CodonTable.default_table()
    # Find the amino acid encoded by a given codon
    print(table["TAC"])
    # Find the codons encoding a given amino acid
    print(table["Y"])
    # Works also for codes instead of symbols
    print(table[(1,2,3)])
    print(table[14])

The default :class:`CodonTable` is equal to the NCBI "Standard" table,
with the small difference that only ``'ATG'`` qualifies as start
codon.
You can also use any other official NCBI table via
:func:`CodonTable.load()`.

.. jupyter-execute::

    # Use the official NCBI table name
    table = seq.CodonTable.load("Yeast Mitochondrial")
    print("Yeast Mitochondrial:")
    print(table)
    print()
    # Use the official NCBI table ID
    table = seq.CodonTable.load(11)
    print("Bacterial:")
    print(table)

Feel free to define your own custom codon table via the
:class:`CodonTable` constructor.

Custom sequence types
---------------------

We can also define a sequence type wit a custom alphabet on our own.
First we need to define the symbols that are allowed in the alphabet.
Previously, we have encountered only alphabets that contain characters.
Hence, the specialized :class:`LetterAlphabet` class is used there
(more explanation of the merits of it in the :doc:`next chapter <encoding>`).
However, *Biotite* allows almost every *Python* object to be used as a symbol
in a sequence.

.. jupyter-execute::

    custom_alphabet = seq.Alphabet(["foo", "bar", 42])

There are two ways to define a custom sequence type.
If we require some type-specific methods, we can subclass :class:`Sequence`.
Otherwise, we can use the generic :class:`GeneralSequence` class and pass the
custom alphabet as a parameter.

.. jupyter-execute::

    class MySequence(seq.Sequence):
        def get_alphabet(self):
            return custom_alphabet

    custom_seq = MySequence(["foo", "bar", 42, 42, "foo"])
    print(custom_seq)

    custom_seq = seq.GeneralSequence(
        custom_alphabet, ["foo", "bar", 42, 42, "foo"]
    )
    print(custom_seq)