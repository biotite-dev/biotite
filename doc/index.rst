.. This source code is part of the Biotite package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

#####################
Biotite documentation
#####################

.. sidebar:: Name origin

   Biotite is a mineral within the mica group.
   It usually forms brownish pseudohexagonal crystals.

   .. image:: https://upload.wikimedia.org/wikipedia/commons/3/3b/Biotite%2C_Sanidine_and_Nepheline_-_Ochtendung%2C_Eifel%2C_Germany.jpg
      :alt: Biotite image

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

.. bibliography::

   Kunzmann2018

----

**Sequence** subpackage
-----------------------
This subpackage contains functionality for working with sequence information
of any kind.
The package contains by default sequence types for nucleotides and proteins,
but the alphabet-based implementation allows simple integration of own sequence
types, even if they do not rely on letters.
Besides the standard I/O operations, the package includes general purpose
functions for sequence manipulations and global/local alignments.
On top of the actual sequence data, the subpackage can also handle sequence
features, to annotate your sequences with the respective functionality.
Eventually, the data can be visualized in different *Matplotlib* based
representations, ranging from sequence alignments to feature maps.

.. image:: /examples/gallery/sequence/images/sphx_glr_hcn_hydropathy_001.png
   :width: 48 %
   :target: examples/gallery/sequence/hcn_hydropathy.html

.. image:: /examples/gallery/sequence/images/sphx_glr_avidin_alignment_001.png
   :width: 48 %
   :target: examples/gallery/sequence/avidin_alignment.html

----

**Structure** subpackage
------------------------
This subpackage enables handling of 3D structures of biomolecules.
Simplified, a structure is represented by a list of atoms and their properties,
based on `ndarray` objects.
Optionally, this representation can be enriched with chemical bond information.
*Biotite* supports different structure formats, including the ones provided
by the *RCSB* and *Gromacs* trajectory formats.
The subpackage offers a wide range of functions for atom filtering,
coordinate transformations, angle and bond measurements,
accessible surface area calculation, structure superimposition and more.

.. image:: /examples/gallery/structure/images/sphx_glr_ramachandran_001.png
   :width: 48 %
   :target: examples/gallery/structure/ramachandran.html

.. image:: /examples/gallery/structure/images/sphx_glr_sheet_arrangement_001.png
   :width: 48 %
   :target: examples/gallery/structure/sheet_arrangement.html

----

**Application** subpackage
--------------------------
This subpackage provides interfaces for external software, in case *Biotite*'s
integrated functionality is not sufficient for your tasks.
These interfaces range from locally installed software (e.g. MSA software) to
web services (e.g. BLAST).
The speciality is that the interfaces are seamless:
You do not have to write input files and read output files, you only have to
input *Python* objects and you get *Python* objects.
It is basically very similar to using normal *Python* functions.

.. image:: /examples/gallery/sequence/images/sphx_glr_lexa_conservation_002.png
   :width: 48 %
   :target: examples/gallery/sequence/lexa_conservation.html

.. image:: /examples/gallery/structure/images/sphx_glr_transketolase_sse_004.png
   :width: 48 %
   :target: examples/gallery/structure/transketolase_sse.html

----

**Database** subpackage
-----------------------
This subpackage is all about searching and downloading data from biological
databases, including the arguably most important ones: the *RCSB PDB* and the
*NCBI Entrez* database.

.. image:: /examples/gallery/structure/images/sphx_glr_pdb_statistics_001.png
   :width: 48 %
   :target: examples/gallery/structure/pdb_statistics.html



.. toctree::
   :maxdepth: 1
   :hidden:
   
   install
   tutorial/target/index
   apidoc/index
   examples/gallery/index
   extensions
   contribute
   logo

