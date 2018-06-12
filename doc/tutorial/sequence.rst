.. This source code is part of the Biotite package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

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
3-letter to 1-letter translation (and vice versa).

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

The upper example uses the default `CodonTable` instance. This can be
changed with the `codon_table` parameter.
A `CodonTable` maps codons to amino acid and defines start codons (both
in symbol and code form). A `CodonTable` is mainly used in the
`translate()` method, but can also to find the corresponding amino acid
for a codon and vice versa.

.. code-block:: python

   table = seq.CodonTable.default_table()
   # Find the amino acid encoded by a given codon
   print(table["TAC"])
   # Find the codons encoding a given amino acid
   print(table["Y"])
   # Works also for codes instead of symbols
   print(table[(1,2,3)])
   print(table[14])

Output:

.. code-block:: none

   Y
   ('TAT', 'TAC')
   14
   ((1, 2, 3), (1, 2, 1), (1, 2, 0), (1, 2, 2), (0, 2, 0), (0, 2, 2))

The default `CodonTable` is equal to the NCBI "Standard" table, with the
small difference that only "ATG" qualifies as start codon. You can also
use any other official NCBI table via `CodonTable.load()`.

.. code-block:: python

   # Use the official NCBI table name
   table = seq.CodonTable.load("Yeast Mitochondrial")
   print("Yeast Mitochondrial:")
   print(table)
   print()
   # Use the official NCBI table ID
   table = seq.CodonTable.load(11)
   print("Bacterial:")
   print(table)

Output:

.. code-block:: none

   Yeast Mitochondrial:
   AAA K      AAC N      AAG K      AAT N
   ACA T      ACC T      ACG T      ACT T
   AGA R      AGC S      AGG R      AGT S
   ATA M i    ATC I      ATG M i    ATT I
   
   CAA Q      CAC H      CAG Q      CAT H
   CCA P      CCC P      CCG P      CCT P
   CGA R      CGC R      CGG R      CGT R
   CTA T      CTC T      CTG T      CTT T
   
   GAA E      GAC D      GAG E      GAT D
   GCA A      GCC A      GCG A      GCT A
   GGA G      GGC G      GGG G      GGT G
   GTA V      GTC V      GTG V      GTT V
   
   TAA *      TAC Y      TAG *      TAT Y
   TCA S      TCC S      TCG S      TCT S
   TGA W      TGC C      TGG W      TGT C
   TTA L      TTC F      TTG L      TTT F
   
   
   Bacterial:
   AAA K      AAC N      AAG K      AAT N
   ACA T      ACC T      ACG T      ACT T
   AGA R      AGC S      AGG R      AGT S
   ATA I i    ATC I i    ATG M i    ATT I
   
   CAA Q      CAC H      CAG Q      CAT H
   CCA P      CCC P      CCG P      CCT P
   CGA R      CGC R      CGG R      CGT R
   CTA L      CTC L      CTG L i    CTT L
   
   GAA E      GAC D      GAG E      GAT D
   GCA A      GCC A      GCG A      GCT A
   GGA G      GGC G      GGG G      GGT G
   GTA V      GTC V      GTG V i    GTT V
   
   TAA *      TAC Y      TAG *      TAT Y
   TCA S      TCC S      TCG S      TCT S
   TGA *      TGC C      TGG W      TGT C
   TTA L      TTC F      TTG L i    TTT F

Feel free to define your own custom codon table via the `CodonTable`
constructor.

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
   file.read("path/to/ec_bl21.fasta")
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

Alternatively, a sequence can also be loaded from GenBank or GenPept files,
using the `GenBankFile` and `GenPeptFile` class (more on this later).

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

Sequence features
^^^^^^^^^^^^^^^^^

Sequence features describe functional parts of a sequence, like coding or
regulatory parts. One popular source to obtain information about sequence
features are GenBank (for DNA and RNA) and GenPept (for peptides) files.
As example for sequence features we will work with the GenBank file for the
DNA sequence of the avidin gene (Accession: AJ311647), that we can download
from the NCBI Entrez database. After downloading we can load the file using
the `GenBankFile` class from ``biotite.sequence.io.genbank``:

.. code-block:: python

   import biotite.sequence.io.genbank as gb
   file = gb.GenBankFile()
   file.read("path/to/gg_avidin.gb")
   print("Accession:", file.get_accession())
   print("Definition:", file.get_definition())

Output:

.. code-block:: none

   Accession: AJ311647
   Definition: Gallus gallus AVD gene for avidin, exons 1-4.

Now that we have loaded the file, we want to have a look at the features.
Therefore, we grab the annotation from the file.
An annotation is the collection of features corresponding to one sequence
(the sequence itself is not included, though).
In case of *Biotite* we can get an `Annotation` object from the `GenBankFile`.
This `Annotation` can be iterated in order to obtain single `Feature` objects.
Each `Feature` contains 3 pieces of information: Its feature key
(e.g. *regulatory* or *CDS*), a dictionary of qualifiers and one or multiple
locations on the corresponding sequence.
A `Location` in turn, contains its starting and its ending base/residue
position, the strand it is on (only for DNA) and possible *location defects*
(defects will be discussed later).
In the next example we will print the keys of the features and their locations:

.. code-block:: python

   annotation = file.get_annotation()
   for feature in annotation:
       # Convert the feature locations in better readable format
       locs = [str(loc) for loc in feature.locs]
       print("{:12}   {:}".format(feature.key, locs))

Output:

.. code-block:: none

   source         ['1-1224 >']
   regulatory     ['26-33 >']
   gene           ['98-1152 >']
   mRNA           ['98-178 >', '263-473 >', '899-1019 >', '1107-1152 >']
   CDS            ['98-178 >', '263-473 >', '899-1019 >', '1107-1152 >']
   sig_peptide    ['98-169 >']
   exon           ['98-178 >']
   intron         ['179-262 >']
   exon           ['263-473 >']
   intron         ['474-898 >']
   exon           ['899-1019 >']
   intron         ['1020-1106 >']
   exon           ['1107-1152 >']
   regulatory     ['1215-1220 >']

The '>' characters in the string representations of a location indicate
that the location is on the forward strand. Most of the features have only
one location, except the *mRNA* and *CDS* feature, which have 4 locations
joined. When we look at the rest of the features, this makes sense: The gene
has 4 exons. Therefore, the mRNA (and consequently the CDS) is composed of
these exons.

The two *regulatory* features are the TATA box and the poly-A signal as the
feature qualifiers make clear:

.. code-block:: python

   for feature in annotation:
       if feature.key == "regulatory":
           print(feature.qual["regulatory_class"])

Output:

.. code-block:: none

   TATA_box
   polyA_signal_sequence

`Annotation` objects can be indexed with slices, that represent the start and
the stop base/residue of the annotation from which the subannotation is
created. All features, that are not in this range, are not included in the
subannotation.
In order to demonstrate this indexing method, we create a subannotation that
includes only features in range of the gene itself (without the regulatory
stuff):

.. code-block:: python

   # At first we have the find the feature with the 'gene' key
   for feature in annotation:
       if feature.key == "gene":
           gene_feature = feature
   # Then we create a subannotation from the feature's location
   # Since the stop value of the slice is still exclusive,
   # the stop value is the position of the last base +1
   loc = gene_feature.locs[0]
   sub_annot = annotation[loc.first : loc.last +1]
   # Print the remaining features and their locations
   for feature in sub_annot:
       locs = [str(loc) for loc in feature.locs]
       print("{:12}   {:}".format(feature.key, locs))

Output:

.. code-block:: none

   source         ['98-1152 >']
   gene           ['98-1152 >']
   mRNA           ['98-178 >', '263-473 >', '899-1019 >', '1107-1152 >']
   CDS            ['98-178 >', '263-473 >', '899-1019 >', '1107-1152 >']
   sig_peptide    ['98-169 >']
   exon           ['98-178 >']
   intron         ['179-262 >']
   exon           ['263-473 >']
   intron         ['474-898 >']
   exon           ['899-1019 >']
   intron         ['1020-1106 >']
   exon           ['1107-1152 >']

The regulatory sequences have disappeared in the subannotation. Another
interesting thing happened: the location of the *source* feature narrowed and
is in range of the slice now. This happened, because the feature was
*truncated*: The bases that were not in range of the slice were removed.

Let's have a closer look into location defects now: A `Location` instance
has a defect, when the feature itself is not directly located in the range of
the first to the last base, for example when the exact postion is not known or,
as in our case, a part of the feature was truncated. Let's have a look at the
location defects of our subannotation:

.. code-block:: python

   for feature in sub_annot:
       defects = [int(location.defect) for location in feature.locs]
       print("{:12}   {:}".format(feature.key, defects))

Output:

.. code-block:: none

   source         [3]
   gene           [12]
   mRNA           [4, 0, 0, 8]
   CDS            [0, 0, 0, 0]
   sig_peptide    [0]
   exon           [0]
   intron         [0]
   exon           [0]
   intron         [0]
   exon           [0]
   intron         [0]
   exon           [0]

The class `Location.Defect` is an `IntEnum` that behaves like a flag
(unfortunately the `Flag` class is only available since Python 3.6). This means
that multiple defects can be combined to one value.
`0` means that the location has no defect, which is true for most of the
features.
The *source* feature has a defect with a value of `3`, which is a combination
of `1` (*MISS_LEFT*) and `2` (*MISS_RIGHT*). *MISS_LEFT* is applied, if a
feature was truncated before the first base and *MISS_RIGHT* is applied if
feature was truncated after the last base. Since *source* was truncated from
both sides, the combinated is applied.
*gene* has the defect value `12`, combining `4` (*BEYOND_LEFT*) and
`8` (*BEYOND_RIGHT*). These defects already appear in the GenBank file, since
the gene is defined as the unit that is transcribed into one (pre-)mRNA. As
the transcription starts somewhere before the start of the coding region, but
the exact location is not known, *BEYOND_LEFT* is applied. In an analogous way,
the transription does stop somewhere after the coding region (at the terminator
signal), hence *BEYOND_RIGHT* is applied. These two defects are also reflected
in the *mRNA* feature.

Now, that you have understood what annotations are, we proceed to the next
topic: annotated sequences. An `AnnotatedSequence` is like an annotation, but
the sequence is included this time. Since our GenBank file contains the
sequence corresponding to the feature table, we can directly obtain the
`AnnotatedSequence`:

.. code-block:: python

   annot_seq = file.get_annotated_sequence()
   print("Same annotation as before?", (annotation == annot_seq.annotation))
   print(annot_seq.sequence[:60], "...")

Output:

.. code-block:: none

   Same annotation as before? True
   ACTGGGCAGAGTCAGTGCTGGAAGCAATMAAAAGGCGAGGGAGCAGGCAGGGGTGAGTCC ...

When indexing an `AnnotatedSequence` with a slice, the index is applied to the
`Annotation` and the `Sequence`. While the `Annotation` handles the index as
shown before, the `Sequence` is indexed, based on the sequence start value.

.. code-block:: python

   print("Sequence start before indexing:", annot_seq.sequence_start)
   for feature in annot_seq.annotation:
       if feature.key == "regulatory" \
           and feature.qual["regulatory_class"] == "polyA_signal_sequence":
               polya_feature = feature
   loc = feature.locs[0]
   # Get annotated sequence containing only the poly-A signal region
   poly_a = annot_seq[loc.first : loc.last +1]
   print("Sequence start after indexing:", poly_a.sequence_start)
   print(poly_a.sequence)

Output:

.. code-block:: none

   Sequence start before indexing: 1
   Sequence start after indexing: 1215
   AATAAA

Here we get the poly-A signal Sequence ``AATAAA``. As you might have noticed,
the sequence start has shifted to the start of the slice index (the first base
of the *regulatory* feature).

.. warning:: Since `AnnoatedSequence` objects use base position indices and
   `Sequence` objects use array position indices, you will get different
   results for ``annot_seq[n:m].sequence`` and ``annot_seq.sequence[n:m]``.

There is also a convenient way to obtain the sequence of a feature, even if
the feature contains multiple locations or a location is on the reverse strand:
Simply use a `Feature` object as index:

.. code-block:: python

   for feature in annot_seq.annotation:
       if feature.key == "CDS":
           cds_feature = feature
   dna_seq = annot_seq[cds_feature]
   print(dna_seq[:60], "...")

Output:

.. code-block:: none

   ATGGTGCACGCAACCTCCCCGCTGCTGCTGCTGCTGCTGCTCAGCCTGGCTCTGGTGGCT ...

Awesome. Now we can translate the sequence and compare it with the translation
given by the CDS feature. But before we can do that, we have to prepare the
data: The DNA sequence uses currently an ambiguous alphabet due to the nasty
`M` at position 28 of the original sequence, we have to remove the stop symbol
after translation and we need to remove the space characters in the translation
given by the CDS feature.

.. code-block:: python

   import biotite.sequence as seq
   # This step make the alphabet unambiguous
   dna_seq = seq.NucleotideSequence(dna_seq)
   prot_seq = dna_seq.translate(complete=True)
   print("Are the translated sequences equal?",
         (str(prot_seq.remove_stops()) == \
          cds_feature.qual["translation"].replace(" ", "")))

Output:

.. code-block:: none

   Are the translated sequences equal? True