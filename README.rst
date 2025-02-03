.. image:: https://img.shields.io/pypi/v/biotite.svg
   :target: https://pypi.python.org/pypi/biotite
   :alt: Biotite at PyPI
.. image:: https://img.shields.io/pypi/pyversions/biotite.svg
   :alt: Python version
.. image:: https://github.com/biotite-dev/biotite/actions/workflows/test_and_deploy.yml/badge.svg
   :target: https://github.com/biotite-dev/biotite/actions/workflows/test_and_deploy.yml
   :alt: Test status

.. image:: https://www.biotite-python.org/_static/assets/general/biotite_logo_m.png
   :alt: The Biotite Project

Biotite project
===============

*Biotite* is your Swiss army knife for bioinformatics.
Whether you want to identify homologous sequence regions in a protein family
or you would like to find disulfide bonds in a protein structure: *Biotite*
has the right tool for you.
This package bundles popular tasks in computational molecular biology
into a uniform *Python* library.
It can handle a major part of the typical workflow
for sequence and biomolecular structure data:

   - Searching and fetching data from biological databases
   - Reading and writing popular sequence/structure file formats
   - Analyzing and editing sequence/structure data
   - Visualizing sequence/structure data
   - Interfacing external applications for further analysis

*Biotite* internally stores most of the data as *NumPy* `ndarray` objects,
enabling

   - fast C-accelerated analysis,
   - intuitive usability through *NumPy*-like indexing syntax,
   - extensibility through direct access of the internal *NumPy* arrays.

As a result the user can skip writing code for basic functionality (like
file parsers) and can focus on what their code makes unique - from
small analysis scripts to entire bioinformatics software packages.

If you use *Biotite* in a scientific publication, please cite:

| Kunzmann, P. & Hamacher, K. BMC Bioinformatics (2018) 19:346.
| `<https://doi.org/10.1186/s12859-018-2367-z>`_


Installation
------------

*Biotite* requires the following packages:

   - **numpy**
   - **requests**
   - **msgpack**
   - **networkx**

Some functions require some extra packages:

   - **matplotlib** - Required for plotting purposes.

*Biotite* can be installed via *Conda*...

.. code-block:: console

   $ conda install -c conda-forge biotite

... or *pip*

.. code-block:: console

   $ pip install biotite


Usage
-----

Here is a small example that downloads two protein sequences from the
*NCBI Entrez* database and aligns them:

.. code-block:: python

   import biotite.sequence.align as align
   import biotite.sequence.io.fasta as fasta
   import biotite.database.entrez as entrez

   # Download FASTA file for the sequences of avidin and streptavidin
   file_name = entrez.fetch_single_file(
       uids=["CAC34569", "ACL82594"], file_name="sequences.fasta",
       db_name="protein", ret_type="fasta"
   )

   # Parse the downloaded FASTA file
   # and create 'ProteinSequence' objects from it
   fasta_file = fasta.FastaFile.read(file_name)
   avidin_seq, streptavidin_seq = fasta.get_sequences(fasta_file).values()

   # Align sequences using the BLOSUM62 matrix with affine gap penalty
   matrix = align.SubstitutionMatrix.std_protein_matrix()
   alignments = align.align_optimal(
       avidin_seq, streptavidin_seq, matrix,
       gap_penalty=(-10, -1), terminal_penalty=False
   )
   print(alignments[0])

.. code-block::

   MVHATSPLLLLLLLSLALVAPGLSAR------KCSLTGKWDNDLGSNMTIGAVNSKGEFTGTYTTAV-TA
   -------------------DPSKESKAQAAVAEAGITGTWYNQLGSTFIVTA-NPDGSLTGTYESAVGNA

   TSNEIKESPLHGTQNTINKRTQPTFGFTVNWKFS----ESTTVFTGQCFIDRNGKEV-LKTMWLLRSSVN
   ESRYVLTGRYDSTPATDGSGT--ALGWTVAWKNNYRNAHSATTWSGQYV---GGAEARINTQWLLTSGTT

   DIGDDWKATRVGINIFTRLRTQKE---------------------
   -AANAWKSTLVGHDTFTKVKPSAASIDAAKKAGVNNGNPLDAVQQ

More documentation, including a tutorial, an example gallery and the API
reference is available at `<https://www.biotite-python.org/>`_.


Contribution
------------

Interested in improving *Biotite*?
Have a look at the
`contribution guidelines <https://www.biotite-python.org/latest/contribution/index.html>`_.
Feel free to join our community chat on `Discord <https://discord.gg/cUjDguF>`_.
