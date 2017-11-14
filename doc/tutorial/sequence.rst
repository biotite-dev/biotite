From A to T - The Sequence subpackage
-------------------------------------

``sequence`` is a *Biotite* subpackage concerning maybe the most popular
in computational biology: sequences. The instantiation can be quite simple as:

.. code-block:: python

   >>> import biotite.sequence as seq
   >>> dna = seq.NucleotideSequence("AACTGCTA")
   >>> print(dna)
   AACTGCTA

This example shows `NucleotideSequence` which is a subclass of the abstract
base class `Sequence`. A `NucleotideSequence` accepts a list of strings,
where each string can be 'A', 'C', 'G' or 'T'. Each of these letters is called
a *symbol*.

In general the sequence implementation in *Biotite* allows for
*Sequences of anything*. This means any (immutable an hashable) *Python* object
can be used as a symbol in a `Sequence`, as long as the object is part of the
`Alphabet` of the particular `Sequence`. An `Alphabet` object simply represents
a list of objects that are allowed to occur in a `Sequence`. The following
figure shows how the symbols are stored in a sequence.

.. image:: /static/assets/figures/symbol_encoding_path.svg

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

The ``sequence`` subpackage offers two prominent `Sequence` sublasses:

The `NucleotideSequence` represents DNA. It may use two different alphabets -
an unambiguous alphabet containing the letters 'A', 'C', 'G' and 'T' and an
ambiguous alphabet containing additionally the standard letters for
ambiguous nucleic bases. A `NucleotideSequence` determines automatically which
alphabet is required, unless an alphabet is specified. If you want to work with
RNA sequences you can use this class, too, you just need to replace the 'U'
with 'T'.

.. code-block:: python

   import biotite.sequence as seq
   # Create a nucleotide sequence using a string
   # The constructor can take any iterable object (e.g. a list of symbols)
   seq1 = seq.NucleotideSequence("ACCGTATCAAG")
   print(seq1.get_alphabet())
   # Constructing a sequence with ambiguous nucleic bases
   seq2 = seq.NucleotideSequence("TANNCGNGG")
   print(seq2.get_alphabet())

Output:

.. code-block:: none

   ['A', 'C', 'G', 'T']
   ['A', 'C', 'G', 'T', 'R', 'Y', 'W', 'S', 'M', 'K', 'H', 'B', 'V', 'D', 'N', 'X']

The reverse complement of a DNA sequence is created by chaining the
`reverse()` and the `complement()` method.

.. code-block:: python

   seq1 = seq.NucleotideSequence("TACAGTT")
   print(seq1)
   seq2 = seq1.reverse().complement()
   print(seq2)

Output:

.. code-block:: none

   TACAGTT
   AACTGTA

The other `Sequence` type is `ProteinSequence`. It supports the letters for
the 20 standard amino acids plus some letters for ambiguous amino acids and a
letter for a stop signal. Furthermore this class provides some utilities like
codon tables, 3 letter to single letter translation (and vice versa) and more.

.. code-block:: python

   seq1 = seq.ProteinSequence("BIQTITE")
   print("-".join([seq.ProteinSequence.convert_letter_1to3(symbol)
                   for symbol in seq1]))

Output:

.. code-block:: none

   ASX-ILE-GLN-THR-ILE-THR-GLU

A `NucleotideSequence` can be translated into a `ProteinSequence` via the
`translate()` method. By default, the method searches for open reading frames
(ORFs) in the 3 frames of the sequence. A 6 frame ORF search requires an
additional call of the `translate()` method with the reverse complement
sequence. If you want to conduct a complete translation of the sequence,
irrespective of any start and stop codons, set the parameter `complete` to
`True`.

.. code-block:: python

   dna = seq.NucleotideSequence("CATATGATGTATGCAATAGGGTGAATG")
   proteins, pos = dna.translate()
   for i in range(len(proteins)):
       print("Protein sequence {:} from base {:d} to base {:d}"
             .format(str(proteins[i]), pos[i][0]+1, pos[i][1]))
   protein = dna.translate(complete=True)
   print("Complete translation:", str(protein))

Output:

.. code-block:: none

   Protein sequence MMYAIG* from base 4 to base 24
   Protein sequence MYAIG* from base 7 to base 24
   Protein sequence MQ* from base 11 to base 19
   Protein sequence M from base 25 to base 27
   Complete translation: HMMYAIG*M

Other paramters in this powerful `translate()` method allow for a custom
codon table or custom start codons.

Loading sequences from file
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Biotite* enables the user to load and save sequences from/to the popular
FASTA format. A FASTA file may contain multiple sequences. Each sequence entry
starts with a line with a leading '>' and a trailing header name. The
corresponding sequence is specified in the following lines until the next
header or end of file. Since every sequence has its obligatory header, a FASTA
file is predestinated to be implemented as some kind of dictionary. This is
exactly what has been done in *Biotite*: The header strings (without the '>')
are used as keys to access the sequence string. Actually you can cast the
`FastaFile` object into a `dict`.
Let's demonstrate this on the genome of *Escherichia coli* BL21(DE3)
(Accession: CP001509.3). After downloading the FASTA file from the NCBI Entrez
database, we can load the contents in the following way:

.. code-block:: python

   import biotite.sequence as seq
   import biotite.sequence.io.fasta as fasta
   file = FastaFile()
   file.read("tests/sequence/data/ec_bl21.fasta")
   for header, string in file:
       print(header)
       print(len(string))
       print(string[:50])

Output:

.. code-block:: none

   CP001509.3 Escherichia coli BL21(DE3), complete genome
   4558953
   AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAA

Since there is only a single sequence in the file, the loop is run only one
time. Since the sequence string is very long, only the first 50 bp are
printed.
Now this string could be used as input parameter for creation of a
`NucleotideSequence`. But I want to spare you some unnecessary work, there
is already a convenience function for that:

.. code-block:: python

   dna_seq = fasta.get_sequence(file)
   print(type(dna_seq).__name__)
   print(dna_seq[:50])

Output:

.. code-block:: none

   NucleotideSequence
   AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAA

In this form `get_sequence()` returns the first sequence in the file, which is
also the only sequence in most cases. If you want the sequence corresponding
to a specific header, you have to specifiy the `header` parameter.
The function even automatically recognizes if the file contains a DNA or
protein sequence and returns a `NucleotideSequence` or `ProteinSequence`,
instance respectively. Actually it just tries to create a `NucleotideSequence`,
and if this fails, a `ProteinSequence` is created instead.

Sequences can be written into FASTA files in a similar way: either via
dictionary-like access or, as show below, using the `set_sequence()`
convenience function.

.. code-block:: python

   # PROTIP: Let your cat walk over the keyboard
   dna_seq2 = seq.NucleotideSequence("ATCGGATCTATCGATGCTAGCTACAGCTAT")
   fasta.set_sequence(file, dna_seq2, header="gibberish")
   print(file["gibberish"])

Output:

.. code-block:: none

   ATCGGATCTATCGATGCTAGCTACAGCTAT

As you see, our file contains our new 'gibberish' sequence now, additionally
to the original sequence.

Sequence search
^^^^^^^^^^^^^^^

A sequence can be searched for the indices of a subsequence or a specific
symbol:

.. code-block:: python

   import biotite.sequence as seq
   main_seq = seq.NucleotideSequence("ACCGTATCAAGTATTG")
   sub_seq = seq.NucleotideSequence("TAT")
   print(seq.find_subsequence(main_seq, sub_seq))
   print(seq.find_symbol(main_seq, "C"))

Output:

.. code-block:: none

   [ 4 11]
   [1 2 7]

Sequence alignments
^^^^^^^^^^^^^^^^^^^

When comparing two (or more) sequences, usually an alignment needs to be
performed. Two kinds of algorithms need to be distinguished here:
Heuristic algorithms do not guarantee to yield the optimal alignment, but
instead they are very fast. On the other hand, there are algorithms that
calculate the optimal (maximum similarity score) alignment, but are quite slow.

The ``sequence.align`` package provides the function `align_optimal()`, which
either performs an optimal global alignment, using the *Needleman-Wunsch*
algorithm, or an optimal local alignment, using the *Smith-Waterman*
algorithm. By default it uses a general gap penalty, but an affine gap penalty
can be used, too.

Most functions in ``sequence.align`` can align any two `Sequence` objects with
each other. In fact the `Sequence` objects can be instances from different
`Sequence` subclasses and therefore may have different alphabets. The only
condition that must be satisfied, is that the `SubstitutionMatrix` alphabets
matches the alphabets of the sequences to be aligned.

But wait, what's a `SubstitutionMatrix`? This class maps a similarity score
to two symbols, one from the first sequence the other from the second sequence.
A `SubstitutionMatrix` object contains two alphabets with length *n* or *m*,
respectively, and an *(n,m)*-shaped `ndarray` storing the similarity scores.
You can choose one of many predefined matrices from an internal database
or you can create a custom matrix on your own.

So much for theory, Let's start by showing different ways to construct
a `SubstitutionMatrix`, in our case for protein sequence alignments:

.. code-block:: python

   import biotite.sequence as seq
   import biotite.sequence.align as align
   import numpy as np
   alph = seq.ProteinSequence.alphabet
   # Load the standard protein substitution matrix, which is BLOSUM62
   matrix = align.SubstitutionMatrix.std_protein_matrix()
   # Load another matrix from internal database
   matrix = align.SubstitutionMatrix(alph, alph, "BLOSUM50")
   # Load a matrix dictionary representation,
   # modify it, and create the SubstitutionMatrix
   # (Dictionary could be loaded from matrix string in NCBI format, too)
   matrix_dict = align.SubstitutionMatrix.dict_from_db("BLOSUM62")
   matrix_dict[("P","Y")] = 100
   matrix = align.SubstitutionMatrix(alph, alph, matrix_dict)
   # And now create a matrix by directly provding the ndarray
   # containing the similarity scores
   # (identity matrix in our case)
   scores = np.identity(len(alph), dtype=int)
   matrix = align.SubstitutionMatrix(alph, alph, scores)

For our protein alignment we will use the standard *BLOSUM62* matrix.

.. code-block:: python

   seq1 = seq.ProteinSequence("BIQTITE")
   seq2 = seq.ProteinSequence("IQLITE")
   matrix = align.SubstitutionMatrix.std_protein_matrix()
   print("Global alignment")
   alignments = align.align_optimal(seq1, seq2, matrix, local=False)
   for ali in alignments:
       print(ali)
   print("Local alignment")
   alignments = align.align_optimal(seq1, seq2, matrix, local=True)
   for ali in alignments:
       print(ali)

Output:

.. code-block:: none

   Global alignment
   BIQTITE
   -IQLITE
   Local alignment
   IQTITE
   IQLITE

The alignment functions return a list of `Alignment` objects. This object saves
the input sequences together with the indices (so called trace) in these
sequences that are aligned to each other (*-1* for a gap). Additionally the
alignment score is stored in this object. Furthermore, this object can
prettyprint the alignment into a human readable form.



