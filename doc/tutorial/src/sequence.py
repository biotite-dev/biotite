"""
From A to T - The Sequence subpackage
=====================================

.. currentmodule:: biotite.sequence

:mod:`biotite.sequence` is a *Biotite* subpackage concerning maybe the
most popular type of data in bioinformatics: sequences.
The instantiation can be quite simple as
"""

import biotite.sequence as seq

dna = seq.NucleotideSequence("AACTGCTA")
print(dna)

########################################################################
# This example shows :class:`NucleotideSequence` which is a subclass of
# the abstract base class :class:`Sequence`.
# A :class:`NucleotideSequence` accepts an iterable object of strings,
# where each string can be ``'A'``, ``'C'``, ``'G'`` or ``'T'``.
# Each of these letters is called a *symbol*.
# 
# In general the sequence implementation in *Biotite* allows for
# *sequences of anything*.
# This means any immutable and hashable *Python* object can be used as
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
# alphabet, so the symbol code for *s* is *i*.
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
# Effectively, this means a potential :class:`Sequence` subclass could
# look like following:

class NonsenseSequence(seq.Sequence):
    
    alphabet = seq.Alphabet([42, "foo", b"bar"])
    
    def get_alphabet(self):
        return NonsenseSequence.alphabet

sequence = NonsenseSequence(["foo", b"bar", 42, "foo", "foo", 42])
print("Alphabet:", sequence.alphabet)
print("Symbols:", sequence.symbols)
print("Code:", sequence.code)

########################################################################
# From DNA to Protein
# -------------------
# 
# Biotite offers two prominent :class:`Sequence` sublasses:
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
print("Original:", seq1)
seq2 = seq1.reverse().complement()
print("Reverse complement:", seq2)

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
# A 6-frame ORF search requires an
# additional call of :func:`NucleotideSequence.translate()` with the
# reverse complement of the sequence.
# If you want to conduct a complete 1-frame translation of the sequence,
# irrespective of any start and stop codons, set the parameter
# :obj:`complete` to true.

dna = seq.NucleotideSequence("CATATGATGTATGCAATAGGGTGAATG")
proteins, pos = dna.translate()
for i in range(len(proteins)):
    print(
        f"Protein sequence {str(proteins[i])} "
        f"from base {pos[i][0]+1} to base {pos[i][1]}"
    )
protein = dna.translate(complete=True)
print("Complete translation:", str(protein))

########################################################################
# The upper example uses the default :class:`CodonTable` instance.
# This can be changed with the :obj:`codon_table` parameter.
# A :class:`CodonTable` maps codons to amino acids and defines start
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
# the respective sequence strings.
# Actually you can cast the  :class:`FastaFile` object into a
# :class:`dict`.
# Let's demonstrate this on the genome of the *lambda* phage
# (Accession: ``NC_001416``).
# After downloading the FASTA file from the NCBI Entrez database,
# we can load its contents in the following way:

from tempfile import gettempdir, NamedTemporaryFile
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import biotite.database.entrez as entrez

file_path = entrez.fetch(
    "NC_001416", gettempdir(), suffix="fa",
    db_name="nuccore", ret_type="fasta"
)
fasta_file = fasta.FastaFile.read(file_path)
for header, string in fasta_file.items():
    print("Header:", header)
    print(len(string))
    print("Sequence:", string[:50], "...")
    print("Sequence length:", len(string))

########################################################################
# Since there is only a single sequence in the file, the loop is run
# only one time.
# As the sequence string is very long, only the first 50 bp are printed.
# Now this string could be used as input parameter for creation of a
# :class:`NucleotideSequence`.
# But we want to spare ourselves some unnecessary work, there is already
# a convenience function for that:

dna_seq = fasta.get_sequence(fasta_file)
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
fasta_file = fasta.FastaFile()
# PROTIP: Let your cat walk over the keyboard
dna_seq1 = seq.NucleotideSequence("ATCGGATCTATCGATGCTAGCTACAGCTAT")
dna_seq2 = seq.NucleotideSequence("ACGATCTACTAGCTGATGTCGTGCATGTACG")
# Append entries to file...
# ... via set_sequence()
fasta.set_sequence(fasta_file, dna_seq1, header="gibberish")
# .. or dictionary style
fasta_file["more gibberish"] = str(dna_seq2)
print(fasta_file)
temp_file = NamedTemporaryFile(suffix=".fasta", delete=False)
fasta_file.write(temp_file.name)
temp_file.close()

########################################################################
# As you see, our file contains our new ``'gibberish'`` and
# ``'more gibberish'`` sequences now.
#
# In a similar manner sequences and sequence quality scores can be read
# from FASTQ files. For further reference, have a look at the
# :mod:`biotite.sequence.io.fastq` subpackage.
#
# Alternatively, a sequence can also be loaded from GenBank or GenPept
# files, using the :class:`GenBankFile` class (more on this later).
# 
# Sequence search
# ---------------
# 
# A sequence can be searched for the position of a subsequence or a
# specific symbol:

import biotite.sequence as seq

main_seq = seq.NucleotideSequence("ACCGTATCAAGTATTG")
sub_seq = seq.NucleotideSequence("TAT")
print("Occurences of 'TAT':", seq.find_subsequence(main_seq, sub_seq))
print("Occurences of 'C':", seq.find_symbol(main_seq, "C"))

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
# In fact the :class:`Sequence` objects can be instances from different
# :class:`Sequence` subclasses and therefore may have different
# alphabets.
# The only condition that must be satisfied, is that the
# :class:`SubstitutionMatrix` alphabets matches the alphabets of the
# sequences to be aligned.
# 
# But wait, what's a :class:`SubstitutionMatrix`?
# This class maps a similarity score to two symbols, one from the first
# sequence the other from the second sequence.
# A :class:`SubstitutionMatrix` object contains two alphabets with
# length *n* or *m*, respectively, and an *(n,m)*-shaped
# :class:`ndarray` storing the similarity scores.
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
# (The dictionary could be alternatively loaded from a string containing
# the matrix in NCBI format)
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
# The alignment functions return a list of :class:`Alignment` objects.
# This object saves the input sequences together with a so called trace
# - the indices to symbols in these sequences that are aligned to each
# other (*-1* for a gap).
# Additionally the alignment score is stored in this object.
# Furthermore, this object can prettyprint the alignment into a human
# readable form.
#
# For publication purposes you can create an actual figure based
# on *Matplotlib*.
# You can either decide to color the symbols based on the symbol type 
# or based on the similarity within the alignment columns.
# In this case we will go with the similarity visualization.

import matplotlib.pyplot as plt
import biotite.sequence.graphics as graphics

fig, ax = plt.subplots(figsize=(2.0, 0.8))
graphics.plot_alignment_similarity_based(
    ax, alignments[0], matrix=matrix, symbols_per_line=len(alignments[0])
)
fig.tight_layout()

########################################################################
# If you are interested in more advanced visualization examples, have a
# look at the :doc:`example gallery <../../examples/gallery/index>`.
# 
# You can also do some simple analysis on these objects, like
# determining the sequence identity or calculating the score.
# For further custom analysis, it can be convenient to have directly the
# aligned symbos codes instead of the trace.

alignment = alignments[0]
print("Score: ", alignment.score)
print("Recalculated score:", align.score(alignment, matrix=matrix))
print("Sequence identity:", align.get_sequence_identity(alignment))
print("Symbols:")
print(align.get_symbols(alignment))
print("symbols codes:")
print(align.get_codes(alignment))

########################################################################
#
# .. currentmodule:: biotite.sequence.io.fasta
#
# You wonder, why you should recalculate the score, when the score has
# already been directly calculated via :func:`align_optimal()`.
# The answer is that you might load an alignment from a FASTA file
# using :func:`get_alignment()`, where the score is not provided.
# 
# .. currentmodule:: biotite.sequence.align
#
# If you want to perform a multiple sequence alignment, have a look at
# the :func:`align_multiple()` function or the interfaces to external
# MSA software in the :mod:`biotite.application` subpackage.
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
# for the avidin gene (Accession: ``AJ311647``),
# that we can download from the NCBI Entrez database.
# After downloading we can load the file using the :class:`GenBankFile`
# class from :mod:`biotite.sequence.io.genbank`.
# Similar to the other file classes we have encountered, a
# :class:`GenBankFile` provides a low-level interface.
# In contrast, the :mod:`biotite.sequence.io.genbank` module contains
# high-level functions to directly obtain useful objects from a
# :class:`GenBankFile` object.

import biotite.sequence.io.genbank as gb

file_path = entrez.fetch(
    "AJ311647", gettempdir(), suffix="gb",
    db_name="nuccore", ret_type="gb"
)
file = gb.GenBankFile.read(file_path)
print("Accession:", gb.get_accession(file))
print("Definition:", gb.get_definition(file))

########################################################################
# 
# .. currentmodule:: biotite.sequence
# 
# Now that we have loaded the file, we want to have a look at the
# sequence features.
# Therefore, we grab the :class:`Annotation` from the file.
# An annotation is the collection of features corresponding to one
# sequence (the sequence itself is not included, though).
# This :class:`Annotation` can be iterated in order to obtain single
# :class:`Feature` objects.
# Each :class:`Feature` contains 3 pieces of information: Its feature
# key (e.g. ``regulatory`` or ``CDS``), a dictionary of qualifiers and
# one or multiple locations on the corresponding sequence.
# A :class:`Location` in turn, contains its starting and its ending
# base/residue position, the strand it is on (only for DNA) and possible
# *location defects* (defects will be discussed later).
# In the next example we will print the keys of the features and their
# locations:

annotation = gb.get_annotation(file)
for feature in annotation:
    # Convert the feature locations in better readable format
    locs = [str(loc) for loc in sorted(feature.locs, key=lambda l: l.first)]
    print(f"{feature.key:12}   {locs}")

########################################################################
# The ``'>'`` characters in the string representations of a location
# indicate that the location is on the forward strand.
# Most of the features have only one location, except the ``mRNA`` and
# ``CDS`` feature, which have 4 locations joined.
# When we look at the rest of the features, this makes sense: The gene
# has 4 exons.
# Therefore, the mRNA (and consequently the CDS) is composed of
# these exons.
#
# The two ``regulatory`` features are the TATA box and the
# poly-A signal, as the feature qualifiers make clear:

for feature in annotation:
    if feature.key == "regulatory":
        print(feature.qual["regulatory_class"])


########################################################################
# Similarily to :class:`Alignment` objects, we can visualize an
# Annotation using the :mod:`biotite.sequence.graphics` subpackage, in
# a so called *feature map*.
# In order to avoid overlaping features, we draw only the *CDS* feature.

# Get the range of the entire annotation via the *source* feature
for feature in annotation:
    if feature.key == "source":
        # loc_range has exclusive stop
        loc = list(feature.locs)[0]
        loc_range = (loc.first, loc.last+1)
fig, ax = plt.subplots(figsize=(8.0, 1.0))
graphics.plot_feature_map(
    ax,
    seq.Annotation(
        [feature for feature in annotation if feature.key == "CDS"]
    ),
    multi_line=False,
    loc_range=loc_range,
    show_line_position=True
)
fig.tight_layout()

########################################################################
# :class:`Annotation` objects can be indexed with slices, that represent
# the start and the exclusive stop base/residue of the annotation from
# which the subannotation is created.
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
loc = list(gene_feature.locs)[0]
sub_annot = annotation[loc.first : loc.last +1]
# Print the remaining features and their locations
for feature in sub_annot:
    locs = [str(loc) for loc in sorted(feature.locs, key=lambda l: l.first)]
    print(f"{feature.key:12}   {locs}")

########################################################################
# The regulatory sequences have disappeared in the subannotation.
# Another interesting thing happened:
# The location of the ``source``` feature narrowed and
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
    defects = [str(location.defect) for location
               in sorted(feature.locs, key=lambda l: l.first)]
    print(f"{feature.key:12}   {defects}")

########################################################################
# The class :class:`Location.Defect` is a :class:`Flag`.
# This means that multiple defects can be combined to one value.
# ``NONE`` means that the location has no defect, which is true for most
# of the features.
# The ``source`` feature has a defect - a combination of ``MISS_LEFT``
# and ``MISS_RIGHT``. ``MISS_LEFT`` is applied, if a feature was
# truncated before the first base, and ``MISS_RIGHT`` is applied, if
# a feature was truncated after the last base.
# Since ``source``` was truncated from both sides, the combination is
# applied.
# ``gene`` has the defect values ``BEYOND_LEFT`` and ``BEYOND_RIGHT``.
# These defects already appear in the GenBank file, since
# the gene is defined as the unit that is transcribed into one
# (pre-)mRNA.
# As the transcription starts somewhere before the start of the coding
# region and the exact start location is not known, ``BEYOND_LEFT`` is
# applied.
# In an analogous way, the transcription does stop somewhere after the
# coding region (at the terminator signal).
# Hence, ``BEYOND_RIGHT`` is applied.
# These two defects are also reflected in the ``mRNA`` feature.
# 
# Annotated sequences
# ^^^^^^^^^^^^^^^^^^^
#
# Now, that you have understood what annotations are, we proceed to the
# next topic: annotated sequences.
# An :class:`AnnotatedSequence` is like an annotation, but the sequence
# is included this time.
# Since our GenBank file contains the
# sequence corresponding to the feature table, we can directly obtain the
# :class:`AnnotatedSequence`.

annot_seq = gb.get_annotated_sequence(file)
print("Same annotation as before?", (annotation == annot_seq.annotation))
print(annot_seq.sequence[:60], "...")

########################################################################
# When indexing an :class:`AnnotatedSequence` with a slice,
# the index is applied to the :class:`Annotation` and the
# :class:`Sequence`.
# While the :class:`Annotation` handles the index as shown before,
# the :class:`Sequence` is indexed based on the sequence start
# value (usually *1*).

print("Sequence start before indexing:", annot_seq.sequence_start)
for feature in annot_seq.annotation:
    if feature.key == "regulatory" \
        and feature.qual["regulatory_class"] == "polyA_signal_sequence":
            polya_feature = feature
loc = list(polya_feature.locs)[0]
# Get annotated sequence containing only the poly-A signal region
poly_a = annot_seq[loc.first : loc.last +1]
print("Sequence start after indexing:", poly_a.sequence_start)
print(poly_a.sequence)

########################################################################
# Here we get the poly-A signal Sequence ``'AATAAA'``.
# As you might have noticed, the sequence start has shifted to the start
# of the slice index (the first base of the ``regulatory`` feature).
# 
# .. warning:: Since :class:`AnnotatedSequence` objects use base position
#    indices and :class:`Sequence` objects use array position indices,
#    you will get different results for ``annot_seq[n:m].sequence`` and
#    ``annot_seq.sequence[n:m]``.
# 
# There is also a convenient way to obtain the sequence corresponding to
# a feature, even if the feature contains multiple locations or a
# location is on the reverse strand:
# Simply use a :class:`Feature` object (in this case the CDS feature)
# as index.

for feature in annot_seq.annotation:
    if feature.key == "CDS":
        cds_feature = feature
cds_seq = annot_seq[cds_feature]
print(cds_seq[:60], "...")

########################################################################
# Awesome.
# Now we can translate the sequence and compare it with the translation
# given by the CDS feature.
# But before we can do that, we have to prepare the data:
# The DNA sequence uses an ambiguous alphabet due to the nasty
# ``'M'`` at position 28 of the original sequence, we have to remove the
# stop symbol after translation and we need to remove the whitespace
# characters in the translation given by the CDS feature.

# To make alphabet unambiguous we create a new NucleotideSequence
# containing only the CDS portion, which is unambiguous
# Thus, the resulting NucleotideSequence has an unambiguous alphabet
cds_seq = seq.NucleotideSequence(cds_seq)
# Now we can translate the unambiguous sequence.
prot_seq = cds_seq.translate(complete=True)
print(prot_seq[:60], "...")
print(
    "Are the translated sequences equal?",
    # Remove stops of our translation
    (str(prot_seq.remove_stops()) == 
    # Remove whitespace characters from translation given by CDS feature
    cds_feature.qual["translation"].replace(" ", ""))
)

########################################################################
# Phylogenetic and guide trees
# ----------------------------
# 
# .. currentmodule:: biotite.sequence.phylo
#
# Trees have an important role in bioinformatics, as they are used to
# guide multiple sequence alignments or to create phylogenies.
#
# In *Biotite* such a tree is represented by the :class:`Tree` class in
# the :mod:`biotite.sequence.phylo` package.
# A tree is rooted, that means each tree node has at least one child,
# or none in case of leaf nodes.
# Each node in a tree is represented by a :class:`TreeNode`.
# When a :class:`TreeNode` is created, you have to provide either child
# nodes and their distances to this node (intermediate node) or a
# reference index (leaf node).
# This reference index is dependent on the context and can refer to
# anything: sequences, organisms, etc.
#
# The childs and the reference index cannot be changed after object
# creation.
# Also the parent can only be set once - when the node is used as child
# in the creation of a new node.

import biotite.sequence.phylo as phylo

# The reference objects
fruits = ["Apple", "Pear", "Orange", "Lemon", "Banana"]
# Create nodes
apple  = phylo.TreeNode(index=fruits.index("Apple"))
pear   = phylo.TreeNode(index=fruits.index("Pear"))
orange = phylo.TreeNode(index=fruits.index("Orange"))
lemon  = phylo.TreeNode(index=fruits.index("Lemon"))
banana = phylo.TreeNode(index=fruits.index("Banana"))
intermediate1 = phylo.TreeNode(
    children=(apple, pear), distances=(2.0, 2.0)
)
intermediate2 = phylo.TreeNode((orange, lemon), (1.0, 1.0))
intermediate3 = phylo.TreeNode((intermediate2, banana), (2.0, 3.0))
root = phylo.TreeNode((intermediate1, intermediate3), (2.0, 1.0))
# Create tree from root node
tree = phylo.Tree(root=root)
# Trees can be converted into Newick notation
print("Tree:", tree.to_newick(labels=fruits))
# Distances can be omitted
print(
    "Tree w/o distances:",
    tree.to_newick(labels=fruits, include_distance=False)
)
# Distances can be measured
distance = tree.get_distance(fruits.index("Apple"), fruits.index("Banana"))
print("Distance Apple-Banana:", distance)

########################################################################
# You can also plot a tree as dendrogram.

fig, ax = plt.subplots(figsize=(6.0, 6.0))
graphics.plot_dendrogram(ax, tree, labels=fruits)
fig.tight_layout()

########################################################################
# From distances to trees
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# When you want to create a :class:`Tree` from distances obtained for
# example from sequence alignments, you can use the *UPGMA* or
# *neighbour joining* algorithm.

distances = np.array([
    [ 0, 17, 21, 31, 23],
    [17, 0,  30, 34, 21],
    [21, 30, 0,  28, 39],
    [31, 34, 28,  0, 43],
    [23, 21, 39, 43,  0]
])
tree = phylo.upgma(distances)
fig, ax = plt.subplots(figsize=(6.0, 3.0))
graphics.plot_dendrogram(ax, tree, orientation="top")
fig.tight_layout()