.. include:: /tutorial/preamble.rst

Annotating sequences with features
==================================

.. currentmodule:: biotite.sequence.io.genbank

Sequence features describe functional parts of a sequence, like coding or
regulatory regions.
One popular source to obtain information about sequence features are GenBank
(for DNA and RNA) and GenPept (for peptides) files.
As example for sequence features we will work with the GenBank file for the
avidin gene (Accession: ``AJ311647``), that we can download from the
*NCBI Entrez* database.
After downloading we can load the file using the :class:`GenBankFile` class
from :mod:`biotite.sequence.io.genbank`.
Similar to the other file classes we have encountered, a :class:`GenBankFile`
provides a low-level interface.
In contrast, the :mod:`biotite.sequence.io.genbank` module contains high-level
functions to directly obtain useful objects from a :class:`GenBankFile` object.

.. jupyter-execute::

    from tempfile import gettempdir
    import biotite.database.entrez as entrez
    import biotite.sequence.io.genbank as gb

    file_path = entrez.fetch(
        "AJ311647", gettempdir(), suffix="gb",
        db_name="nuccore", ret_type="gb"
    )
    file = gb.GenBankFile.read(file_path)
    print("Accession:", gb.get_accession(file))
    print("Definition:", gb.get_definition(file))

.. currentmodule:: biotite.sequence

Now that we have loaded the file, we want to have a look at the sequence
features.
Therefore, we grab the :class:`Annotation` from the file.
An annotation is the collection of features corresponding to one sequence
(the sequence itself is not included, though).
This :class:`Annotation` can be iterated in order to obtain single
:class:`Feature` objects.
Each :class:`Feature` contains 3 pieces of information:
its feature key (e.g. ``regulatory`` or ``CDS``), a dictionary of qualifiers
and one or multiple locations on the corresponding sequence.
A :class:`Location` in turn, contains its starting and its ending base/residue
position, the strand it is on (only for DNA) and possible *location defects*
(defects will be discussed later).
In the next example we will print the keys of the features and their locations:

.. jupyter-execute::

    annotation = gb.get_annotation(file)
    for feature in annotation:
        # Convert the feature locations in better readable format
        locs = [str(loc) for loc in sorted(feature.locs, key=lambda l: l.first)]
        print(f"{feature.key:12}   {locs}")

The ``'>'`` characters in the string representations of a location indicate
that the location is on the forward strand.
Most of the features have only one location, except the ``mRNA`` and ``CDS``
feature, which have 4 locations joined.
When we look at the rest of the features, this makes sense:
The gene has 4 exons.
Therefore, the mRNA (and consequently the CDS) is composed of these exons.

The two ``regulatory`` features are the TATA box and the
poly-A signal, as the feature qualifiers make clear:

.. jupyter-execute::

    for feature in annotation:
        if feature.key == "regulatory":
            print(feature.qual["regulatory_class"])

Similarily to :class:`Alignment` objects, we can visualize an
:class:`Annotation` using the :mod:`biotite.sequence.graphics` subpackage, in
a so called *feature map*.
In order to avoid overlaping features, we draw only the *CDS* feature.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import biotite.sequence as seq
    import biotite.sequence.graphics as graphics

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

:class:`Annotation` objects can be indexed with slices, that represent
the start and the exclusive stop base/residue of the annotation from
which the subannotation is created.
All features, that are not in this range, are not included in the
subannotation.
In order to demonstrate this indexing method, we create a
subannotation that includes only features in range of the gene itself
(without the regulatory parts).

.. jupyter-execute::

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

The regulatory sequences have disappeared in the subannotation.
Another interesting thing happened:
The location of the ``source`` feature narrowed and
is in range of the slice now. This happened, because the feature was
*truncated*:
The bases that were not in range of the slice were removed.

Let's have a closer look into location defects now:
A :class:`Location` instance has a defect, when the feature itself is
not directly located in the range of the first to the last base,
for example when the exact postion is not known or, as in our case, a
part of the feature was truncated.
Let's have a closer look at the location defects of our subannotation:

.. jupyter-execute::

    for feature in sub_annot:
        defects = [str(location.defect) for location
                   in sorted(feature.locs, key=lambda l: l.first)]
        print(f"{feature.key:12}   {defects}")

The class :class:`Location.Defect` is a :class:`Flag`.
This means that multiple defects can be combined to one value.
``NONE`` means that the location has no defect, which is true for most
of the features.
The ``source`` feature has a defect - a combination of ``MISS_LEFT``
and ``MISS_RIGHT``. ``MISS_LEFT`` is applied, if a feature was
truncated before the first base, and ``MISS_RIGHT`` is applied, if
a feature was truncated after the last base.
Since ``source`` was truncated from both sides, the combination is
applied.
``gene`` has the defect values ``BEYOND_LEFT`` and ``BEYOND_RIGHT``.
These defects already appear in the GenBank file, since
the gene is defined as the unit that is transcribed into one
(pre-)mRNA.
As the transcription starts somewhere before the start of the coding
region and the exact start location is not known, ``BEYOND_LEFT`` is
applied.
In an analogous way, the transcription does stop somewhere after the
coding region (at the terminator signal).
Hence, ``BEYOND_RIGHT`` is applied.
These two defects are also reflected in the ``mRNA`` feature.

Annotated sequences
^^^^^^^^^^^^^^^^^^^

An :class:`AnnotatedSequence` is like an annotation, but the sequence
is included this time.
Since our GenBank file contains the
sequence corresponding to the feature table, we can directly obtain the
:class:`AnnotatedSequence`.

.. jupyter-execute::

    annot_seq = gb.get_annotated_sequence(file)
    print("Same annotation as before?", (annotation == annot_seq.annotation))
    print(annot_seq.sequence[:60], "...")

When indexing an :class:`AnnotatedSequence` with a slice,
the index is applied to the :class:`Annotation` and the
:class:`Sequence`.
While the :class:`Annotation` handles the index as shown before,
the :class:`Sequence` is indexed based on the sequence start
value (usually *1*).

.. jupyter-execute::

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

Here we get the poly-A signal Sequence ``'AATAAA'``.
As you might have noticed, the sequence start has shifted to the start
of the slice index (the first base of the ``regulatory`` feature).

.. warning:: Since :class:`AnnotatedSequence` objects use base position
   indices and :class:`Sequence` objects use array position indices,
   you will get different results for ``annot_seq[n:m].sequence`` and
   ``annot_seq.sequence[n:m]``.

There is also a convenient way to obtain the sequence corresponding to
a feature, even if the feature contains multiple locations or a
location is on the reverse strand:
Simply use a :class:`Feature` object (in this case the CDS feature)
as index.

.. jupyter-execute::

    for feature in annot_seq.annotation:
        if feature.key == "CDS":
            cds_feature = feature
    cds_seq = annot_seq[cds_feature]
    print(cds_seq[:60], "...")

Now we can translate the sequence and compare it with the translation
given by the CDS feature.
But before we can do that, we have to prepare the data:
The DNA sequence uses an ambiguous alphabet due to the nasty
``'M'`` at position 28 of the original sequence, we have to remove the
stop symbol after translation and we need to remove the whitespace
characters in the translation given by the CDS feature.

.. jupyter-execute::

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