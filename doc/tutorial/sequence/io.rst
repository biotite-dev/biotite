.. include:: /tutorial/preamble.rst

Reading and writing sequences
=============================

Loading sequences from FASTA files
----------------------------------

.. currentmodule:: biotite.sequence.io.fasta

*Biotite* enables the user to load and save sequences from/to the
popular FASTA format via the :class:`FastaFile` class.
As each sequence in a FASTA file is identified by an unique header,
a :class:`FastaFile` is implemented in a dictionary-like fashion:
The header strings (without the leading ``'>'``) are used as keys to access
the respective sequence strings.
Let's demonstrate this on the hemoglobin example from an earlier chapter.

.. jupyter-execute::

    from tempfile import gettempdir, NamedTemporaryFile
    import biotite.sequence as seq
    import biotite.sequence.io.fasta as fasta
    import biotite.database.entrez as entrez

    temp_file = NamedTemporaryFile(suffix=".fasta", delete=False)
    file_path = entrez.fetch_single_file(
        ["6BB5_A", "6BB5_B"], temp_file.name,
        db_name="protein", ret_type="fasta"
    )
    with open(file_path) as file:
        print(file.read())

After downloading the FASTA file from the NCBI Entrez database,
we can load its contents in the following way:

.. jupyter-execute::

    fasta_file = fasta.FastaFile.read(file_path)
    for header, seq_string in fasta_file.items():
        print("Header:", header)
        # For sake of brevity, print only the start of the sequence
        print("Sequence:", seq_string[:50], "...")
        print()

Now this string could be used as input parameter for creation of a
:class:`NucleotideSequence`.
If we want to spare ourselves some unnecessary work, there is already a
convenience function for that:

.. jupyter-execute::

    dna_sequences = fasta.get_sequences(fasta_file)
    for header, sequence in dna_sequences.items():
        print("Header:", header)
        # Note that this time we have a Sequence object instead of a string
        print("Sequence:", repr(sequence))
        print()

The function even automatically recognizes, if the file contains a
DNA or protein sequence and returns a :class:`NucleotideSequence` or
:class:`ProteinSequence`, instance respectively.
If we would be interested in only a single sequence (or the file contains only
a single sequence anyway) we would use :func:`get_sequence()` instead.

Sequences can be written into FASTA files in a similar way: either via
dictionary-like access or using the  :func:`set_sequence()`
convenience function.

.. jupyter-execute::

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


Assessing sequencing quality using FASTQ files
----------------------------------------------

.. currentmodule:: biotite.sequence.io.fastq

In addition to sequences FASTQ files store the
`Phred quality score <https://en.wikipedia.org/wiki/Phred_quality_score>`_
for each base call from a sequencing run.
Hence, the :class:`FastqFile` class works also like a dictionary, but the value
is tuple containing both, the sequence strings and the corresponding score
array.

.. jupyter-execute::

    from io import StringIO
    from textwrap import dedent
    import biotite.sequence.io.fastq as fastq

    # Sample FASTQ file from https://en.wikipedia.org/wiki/FASTQ_format
    fastq_content = StringIO(dedent(
        """
        @SEQ_ID
        GATTTGGGGTTCAAAGCAGTATCGATCAAATAGTAAATCCATTTGTTCAACTCACAGTTT
        +
        !''*((((***+))%%%++)(%%%%).1***-+*''))**55CCF>>>>>>CCCCCCC65
        """
    ))

    fastq_file = fastq.FastqFile.read(fastq_content, offset="Sanger")
    seq_string, scores = fastq_file["SEQ_ID"]
    print(seq_string)
    print(scores)

Also similar to :mod:`biotite.sequence.io.fasta`, there are convenience
functions for loading :class:`Sequence` objects from FASTQ files.

.. jupyter-execute::

    seq, scores = fastq.get_sequence(fastq_file)
    print(repr(seq))
    print(scores)