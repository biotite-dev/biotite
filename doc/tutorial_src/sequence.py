"""
From A to T - The Sequence subpackage
=====================================

.. currentmodule:: biotite.sequence

:mod:`biotite.sequence` is a *Biotite* subpackage concerning maybe the
most popular data type in computational molecular biology: sequences.
The instantiation can be quite simple as
"""

import biotite.sequence as seq
dna = seq.NucleotideSequence("AACTGCTA")
print(dna)

########################################################################
# This example shows :class:`NucleotideSequence` which is a subclass of
# the abstract base class :class:`Sequence`.
# A :class:`NucleotideSequence` accepts a list of strings,
# where each string can be ``'A'``, ``'C'``, ``'G'`` or ``'T'``.
# Each of these letters is called a *symbol*.
# 
# In general the sequence implementation in *Biotite* allows for
# *sequences of anything*.
# This means any (immutable an hashable) *Python* object can be used as
# a symbol in a sequence, as long as the object is part of the
# :class:`Alphabet` of the particular :class:`Sequence`.
# An :class:`Alphabet` object simply represents a list of objects that
# are allowed to occur in a :class:`Sequence`.
# The following figure shows how the symbols are stored in a
# :class:`Sequence` object.
# 
# .. image:: /static/assets/figures/symbol_encoding_path.svg
# 
# When setting the :class:`Sequence` object with a sequence of symbols,
# the :class:`Alphabet` of the :class:`Sequence` encodes each symbol in
# the input sequence into a so called *symbol code*.
# The encoding process is quite simple:
# A symbol *s* is at index *i* in the list of allowed symbols in the
# alphabet, so the symbol code for *s* is *i*
# If *s* is not in the alphabet, an :class:`AlphabetError` is raised.
# The array of symbol codes, that arises from encoding the input
# sequence, is called *sequence code*.
# This sequence code is now stored in an internal integer
# :class:`ndarray` in the :class:`Sequence` object.
# The sequence code is now accessed via the :attr:`code` attribute,
# the corresponding symbols via the :attr:`symbols` attribute.
# 
# This approach has multiple advantages:
# 
#    - Ability to create *sequences of anything*
#    - Sequence utility functions (searches, alignments,...) usually do
#      not care about the specific sequence type, since they work with
#      the internal sequence code
#    - Integer type for sequence code is only as large as the alphabet 
#      requests
#    - Sequence codes can be directly used as substitution matrix
#      indices in alignments
# 
# Effectively, this means a potential :class:`Sequence` subclass could look
# like following:

class NonsenseSequence(seq.Sequence):
    
    alphabet = seq.Alphabet([42, "foo", b"bar"])
    
    def get_alphabet(self):
        return NonsenseSequence.alphabet

sequence = NonsenseSequence(["foo", b"bar", 42, "foo", "foo", 42])
print("Alphabet: ", sequence.get_alphabet())
print("Symbols: ", sequence.symbols)
print("Code: ", sequence.code)

########################################################################
# From DNA to Protein
# -------------------
# 
# Biotite offers two prominent `Sequence` sublasses:
# 
# The :class:`NucleotideSequence` represents DNA.
# It may use two different alphabets - an unambiguous alphabet
# containing the letters ``'A'``, ``'C'``, ``'G'`` and ``'T'`` and an
# ambiguous alphabet containing additionally the standard letters for
# ambiguous nucleic bases.
# A :class:`NucleotideSequence` determines automatically which alphabet
# is required, unless an alphabet is specified. If you want to work with
# RNA sequences you can use this class, too, you just need to replace
# the ``'U'`` with ``'T'``.

import biotite.sequence as seq
# Create a nucleotide sequence using a string
# The constructor can take any iterable object (e.g. a list of symbols)
seq1 = seq.NucleotideSequence("ACCGTATCAAG")
print(seq1.get_alphabet())
# Constructing a sequence with ambiguous nucleic bases
seq2 = seq.NucleotideSequence("TANNCGNGG")
print(seq2.get_alphabet())

########################################################################
# The reverse complement of a DNA sequence is created by chaining the
# :func:`Sequence.reverse()` and the
# :func:`NucleotideSequence.complement()` method.

# Lower case characters are automatically capitalized
seq1 = seq.NucleotideSequence("tacagtt")
print("Original: ", seq1)
seq2 = seq1.reverse().complement()
print("Reverse complement: ", seq2)

########################################################################
# The other :class:`Sequence` type is :class:`ProteinSequence`.
# It supports the letters for the 20 standard amino acids plus some
# letters for ambiguous amino acids and a letter for a stop signal.
# Furthermore, this class provides some utilities like
# 3-letter to 1-letter translation (and vice versa).

prot_seq = seq.ProteinSequence("BIQTITE")
print("-".join([seq.ProteinSequence.convert_letter_1to3(symbol)
                for symbol in prot_seq]))

########################################################################
# A :class:`NucleotideSequence` can be translated into a
# :class:`ProteinSequence` via the
# :func:`NucleotideSequence.translate()` method.
# By default, the method searches for open reading frames (ORFs) in the
# 3 frames of the sequence.
# A 6 frame ORF search requires an
# additional call of :func:`NucleotideSequence.translate()` with the
# reverse complement of the sequence.
# If you want to conduct a complete translation of the sequence,
# irrespective of any start and stop codons, set the parameter
# :obj:`complete` to true.

dna = seq.NucleotideSequence("CATATGATGTATGCAATAGGGTGAATG")
proteins, pos = dna.translate()
for i in range(len(proteins)):
    print("Protein sequence {:} from base {:d} to base {:d}"
            .format(str(proteins[i]), pos[i][0]+1, pos[i][1]))
protein = dna.translate(complete=True)
print("Complete translation:", str(protein))

########################################################################
# The upper example uses the default :class:`CodonTable` instance.
# This can be changed with the :obj:`codon_table` parameter.
# A :class:`CodonTable` maps codons to amino acid and defines start
# codons (both in symbol and code form).
# A :class:`CodonTable` is mainly used in the
# :func:`NucleotideSequence.translate()` method,
# but can also be used to find the corresponding amino acid for a codon
# and vice versa.

table = seq.CodonTable.default_table()
# Find the amino acid encoded by a given codon
print(table["TAC"])
# Find the codons encoding a given amino acid
print(table["Y"])
# Works also for codes instead of symbols
print(table[(1,2,3)])
print(table[14])

########################################################################
# The default :class:`CodonTable` is equal to the NCBI "Standard" table,
# with the small difference that only ``'ATG'`` qualifies as start
# codon.
# You can also use any other official NCBI table via
# :func:`CodonTable.load()`.

# Use the official NCBI table name
table = seq.CodonTable.load("Yeast Mitochondrial")
print("Yeast Mitochondrial:")
print(table)
print()
# Use the official NCBI table ID
table = seq.CodonTable.load(11)
print("Bacterial:")
print(table)

########################################################################
# Feel free to define your own custom codon table via the
# :class:`CodonTable` constructor.
#
# Loading sequences from file
# ---------------------------
#
# .. currentmodule:: biotite.sequence.io.fasta
# 
# *Biotite* enables the user to load and save sequences from/to the
# popular FASTA format via the :class:`FastaFile` class.
# A FASTA file may contain multiple sequences.
# Each sequence entry starts with a line with a leading ``'>'`` and a
# trailing header name.
# The corresponding sequence is specified in the following lines until
# the next header or end of file.
# Since every sequence has its obligatory header, a FASTA file is
# predestinated to be implemented as some kind of dictionary.
# This is exactly what has been done in *Biotite*:
# The header strings (without the ``'>'``) are used as keys to access
# the sequence string.
# Actually you can cast the  `FastaFile` object into a `dict`.
# Let's demonstrate this on the genome of the *lambda* phage
# (Accession: NC_001416).
# After downloading the FASTA file from the NCBI Entrez database,
# we can load the contents in the following way:

import biotite
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import biotite.database.entrez as entrez
file_path = entrez.fetch(
    "NC_001416", biotite.temp_dir(), suffix="fa",
    db_name="nuccore", ret_type="fasta"
)
file = fasta.FastaFile()
file.read(file_path)
for header, string in file:
    print("Header: ", header)
    print(len(string))
    print("Sequence: ", string[:50], "...")
    print("Sequence length: ", len(string))

########################################################################
# Since there is only a single sequence in the file, the loop is run
# only one time.
# As the sequence string is very long, only the first 50 bp are printed.
# Now this string could be used as input parameter for creation of a
# :class:`NucleotideSequence`.
# But I want to spare you some unnecessary work, there is already a
# convenience function for that:

dna_seq = fasta.get_sequence(file)
print(type(dna_seq).__name__)
print(dna_seq[:50])

########################################################################
# In this form :func:`get_sequence()` returns the first sequence in the
# file, which is also the only sequence in most cases.
# If you want the sequence corresponding to a specific header, you have
# to specifiy the :obj:`header` parameter.
# The function even automatically recognizes, if the file contains a
# DNA or protein sequence and returns a :class:`NucleotideSequence` or
# :class:`ProteinSequence`, instance respectively.
# Actually, it just tries to create a :class:`NucleotideSequence`,
# and if this fails, a :class:`ProteinSequence` is created instead.
# 
# Sequences can be written into FASTA files in a similar way: either via
# dictionary-like access or using the  :func:`set_sequence()`
# convenience function.

# Create new empty FASTA file
file = fasta.FastaFile()
# PROTIP: Let your cat walk over the keyboard
dna_seq1 = seq.NucleotideSequence("ATCGGATCTATCGATGCTAGCTACAGCTAT")
dna_seq2 = seq.NucleotideSequence("ACGATCTACTAGCTGATGTCGTGCATGTACG")
# Append entries to file...
# ... via set_sequence()
fasta.set_sequence(file, dna_seq1, header="gibberish")
# .. or dictionary style
file["more gibberish"] = str(dna_seq2)
print(file)
file.write(biotite.temp_file("fa"))

########################################################################
# As you see, our file contains our new ``'gibberish'`` and
# ``'more gibberish'`` sequences now.
# 
# Alternatively, a sequence can also be loaded from GenBank or GenPept
# files,
# using the :class:`GenBankFile` and :class:`GenPeptFile` class
# (more on this later).
# 
# Sequence search
# ---------------
# 
# A sequence can be searched for the position of a subsequence or a
# specific symbol:

import biotite.sequence as seq
main_seq = seq.NucleotideSequence("ACCGTATCAAGTATTG")
sub_seq = seq.NucleotideSequence("TAT")
print("Occurences of 'TAT': ", seq.find_subsequence(main_seq, sub_seq))
print("Occurences of 'C': ", seq.find_symbol(main_seq, "C"))

########################################################################
# Sequence alignments
# -------------------
# 
# .. currentmodule:: biotite.sequence.align 
#
# When comparing two (or more) sequences, usually an alignment needs
# to be performed. Two kinds of algorithms need to be distinguished
# here:
# Heuristic algorithms do not guarantee to yield the optimal alignment,
# but instead they are very fast. On the other hand, there are
# algorithms that calculate the optimal (maximum similarity score)
# alignment, but are quite slow.
# 
# The :mod:`biotite.sequence.align` package provides the function
# :func:`align_optimal()`, which either performs an optimal global
# alignment, using the *Needleman-Wunsch* algorithm, or an optimal local
# alignment, using the *Smith-Waterman* algorithm.
# By default it uses a general gap penalty, but an affine gap penalty
# can be used, too.
# 
# Most functions in :mod:`biotite.sequence.align` can align any two
# :class:`Sequence` objects with each other.
# In fact the `Sequence` objects can be instances from different
# `Sequence` subclasses and therefore may have different alphabets.
# The only condition that must be satisfied, is that the
# class:`SubstitutionMatrix` alphabets matches the alphabets of the sequences
# to be aligned.
# 
# But wait, what's a :class:`SubstitutionMatrix`?
# This class maps a similarity score to two symbols, one from the first
# sequence the other from the second sequence.
# A :class:`SubstitutionMatrix` object contains two alphabets with
# length *n* or *m*, respectively, and an *(n,m)*-shaped :class:`ndarray`
# storing the similarity scores.
# You can choose one of many predefined matrices from an internal
# database or you can create a custom matrix on your own.
# 
# So much for theory.
# Let's start by showing different ways to construct a
# :class:`SubstitutionMatrix`, in our case for protein sequence
# alignments:

import biotite.sequence as seq
import biotite.sequence.align as align
import numpy as np
alph = seq.ProteinSequence.alphabet
# Load the standard protein substitution matrix, which is BLOSUM62
matrix = align.SubstitutionMatrix.std_protein_matrix()
print("\nBLOSUM62\n")
print(matrix)
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
print("\n\nIdentity matrix\n")
print(matrix)

########################################################################
# For our protein alignment we will use the standard *BLOSUM62* matrix.

seq1 = seq.ProteinSequence("BIQTITE")
seq2 = seq.ProteinSequence("IQLITE")
matrix = align.SubstitutionMatrix.std_protein_matrix()
print("\nLocal alignment")
alignments = align.align_optimal(seq1, seq2, matrix, local=True)
for ali in alignments:
    print(ali)
print("Global alignment")
alignments = align.align_optimal(seq1, seq2, matrix, local=False)
for ali in alignments:
    print(ali)

########################################################################
# The alignment functions return a list of `Alignment` objects.
# This object saves the input sequences together with a so called trace
# - the indices to symbols in these sequences that are aligned to each
# other (*-1* for a gap).
# Additionally the alignment score is stored in this object.
# Furthermore, this object can prettyprint the alignment into a human
# readable form.
#
# You can also do some simple analysis on these objects, like
# determining the sequence identity or calculating the score.
# For further custom analysis, it can be convenient to have directly the
# aligned symbos codes instead of the trace.

alignment = alignments[0]
print("Score: ", alignment.score)
print("Recalculated score: ", align.score(alignment, matrix=matrix))
print("Sequence identity: ", align.get_sequence_identity(alignment))
print("Symbols:")
print(align.get_symbols(alignment))
print("symbols codes:")
print(align.get_codes(alignment))

########################################################################
#
# .. currentmodule:: biotite.sequence.io.fasta
#
# You may ask, why should you recalculate the score, when the score has
# already been directly calculated via :func:`align_optimal()`.
# The answer is, that you might load an alignment from an external
# alignment program as FASTA file using :func:`get_alignment()`.
#
# Sequence features
# -----------------
# 
# .. currentmodule:: biotite.sequence.io.genbank
#
# Sequence features describe functional parts of a sequence,
# like coding regions or regulatory parts.
# One popular source to obtain information about sequence features are
# GenBank (for DNA and RNA) and GenPept (for peptides) files.
# As example for sequence features we will work with the GenBank file
# for the DNA sequence of the avidin gene (Accession: AJ311647),
# that we can download from the NCBI Entrez database.
# After downloading we can load the file using the :class:`GenBankFile`
# class from :mod:`biotite.sequence.io.genbank`.

import biotite.sequence.io.genbank as gb
file_path = entrez.fetch(
    "AJ311647", biotite.temp_dir(), suffix="gb",
    db_name="nuccore", ret_type="gb"
)
file = gb.GenBankFile()
file.read(file_path)
print("Accession:", file.get_accession())
print("Definition:", file.get_definition())

########################################################################
# 
# .. currentmodule:: biotite.sequence
# 
# Now that we have loaded the file, we want to have a look at the
# sequence features.
# Therefore, we grab the annotation from the file.
# An annotation is the collection of features corresponding to one
# sequence (the sequence itself is not included, though).
# In case of *Biotite* we can get an :class:`Annotation` object from the
# :class:`GenBankFile`.
# This :class:`Annotation` can be iterated in order to obtain single
# `Feature` objects.
# Each `Feature` contains 3 pieces of information: Its feature key
# (e.g. *regulatory* or *CDS*), a dictionary of qualifiers and one or
# multiple locations on the corresponding sequence.
# A :class:`Location` in turn, contains its starting and its ending
# base/residue position, the strand it is on (only for DNA) and possible
# *location defects* (defects will be discussed later).
# In the next example we will print the keys of the features and their
# locations:

annotation = file.get_annotation()
for feature in annotation:
    # Convert the feature locations in better readable format
    locs = [str(loc) for loc in feature.locs]
    print(f"{feature.key:12}   {locs}")

########################################################################
# The ``'>'`` characters in the string representations of a location
# indicate that the location is on the forward strand.
# Most of the features have only one location, except the *mRNA* and
# *CDS* feature, which have 4 locations joined.
# When we look at the rest of the features, this makes sense: The gene
# has 4 exons.
# Therefore, the mRNA (and consequently the CDS) is composed of
# these exons.
#
# The two *regulatory* features are the TATA box and the poly-A signal as the
# feature qualifiers make clear:

for feature in annotation:
    if feature.key == "regulatory":
        print(feature.qual["regulatory_class"])

########################################################################
# :class:`Annotation` objects can be indexed with slices, that represent
# the start and the stop base/residue of the annotation from which the
# subannotation is created.
# All features, that are not in this range, are not included in the
# subannotation.
# In order to demonstrate this indexing method, we create a
# subannotation that includes only features in range of the gene itself
# (without the regulatory stuff).

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
    print(f"{feature.key:12}   {locs}")

########################################################################
# The regulatory sequences have disappeared in the subannotation.
# Another interesting thing happened:
# The location of the *source* feature narrowed and
# is in range of the slice now. This happened, because the feature was
# *truncated*:
# The bases that were not in range of the slice were removed.
# 
# Let's have a closer look into location defects now:
# A :class:`Location` instance has a defect, when the feature itself is
# not directly located in the range of the first to the last base,
# for example when the exact postion is not known or, as in our case, a
# part of the feature was truncated.
# Let's have a closer look at the location defects of our subannotation:

for feature in sub_annot:
    defects = [str(location.defect) for location in feature.locs]
    print(f"{feature.key:12}   {defects}")

########################################################################
# The class `Location.Defect` is a :class:`Flag`.
# This means that multiple defects can be combined to one value.
# ``NONE`` means that the location has no defect, which is true for most
# of the features.
# The *source* feature has a defect has a combination of ``MISS_LEFT``
# and ``MISS_RIGHT``. ``MISS_LEFT`` is applied, if a feature was
# truncated before the first base, and ``MISS_RIGHT`` is applied, if
# a feature was truncated after the last base.
# Since *source* was truncated from both sides, the combinated is
# applied.
# *gene* has the defect values ``BEYOND_LEFT`` and ``BEYOND_RIGHT``.
# These defects already appear in the GenBank file, since
# the gene is defined as the unit that is transcribed into one
# (pre-)mRNA.
# As the transcription starts somewhere before the start of the coding
# region, and the exact location is not known, ``BEYOND_LEFT`` is
# applied.
# In an analogous way, the transription does stop somewhere after the
# coding region (at the terminator signal),
# hence ``BEYOND_RIGHT`` is applied.
# These two defects are also reflected in the *mRNA* feature.
# 
# Now, that you have understood what annotations are, we proceed to the
# next topic: annotated sequences.
# An :class:`AnnotatedSequence` is like an annotation, but the sequence
# is included this time.
# Since our GenBank file contains the
# sequence corresponding to the feature table, we can directly obtain the
# :class:`AnnotatedSequence`.

annot_seq = file.get_annotated_sequence()
print("Same annotation as before?", (annotation == annot_seq.annotation))
print(annot_seq.sequence[:60], "...")

########################################################################
# When indexing an :class:`AnnotatedSequence` with a slice,
# the index is applied to the :class:`Annotation` and the
# :class:`Sequence`.
# While the `Annotation` handles the index as shown before,
# the :class:`Sequence` is indexed based on the sequence start
# value (usually *1*).

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

########################################################################
# Here we get the poly-A signal Sequence ``'AATAAA'``.
# As you might have noticed, the sequence start has shifted to the start
# of the slice index (the first base of the *regulatory* feature).
# 
# .. warning:: Since :class:`AnnotatedSequence` objects use base position
#    indices and :class:`Sequence` objects use array position indices,
#    you will get different results for ``annot_seq[n:m].sequence`` and
#    ``annot_seq.sequence[n:m]``.
# 
# There is also a convenient way to obtain the sequence corresponding to
# a feature, even if the feature contains multiple locations or a
# location is on the reverse strand:
# Simply use a `Feature` object as index.

for feature in annot_seq.annotation:
    if feature.key == "CDS":
        cds_feature = feature
dna_seq = annot_seq[cds_feature]
print(dna_seq[:60], "...")

########################################################################
# Awesome.
# Now we can translate the sequence and compare it with the translation
# given by the CDS feature.
# But before we can do that, we have to prepare the data:
# The DNA sequence uses currently an ambiguous alphabet due to the nasty
# ``'M'`` at position 28 of the original sequence, we have to remove the
# stop symbol after translation and we need to remove the space
# characters in the translation given by the CDS feature.

import biotite.sequence as seq
# This step make the alphabet unambiguous
dna_seq = seq.NucleotideSequence(dna_seq)
prot_seq = dna_seq.translate(complete=True)
print("Are the translated sequences equal?",
        (str(prot_seq.remove_stops()) == \
        cds_feature.qual["translation"].replace(" ", "")))