.. This source code is part of the Biotite package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

#####################
Biotite documentation
#####################

The *Biotite* package bundles popular tasks in computational molecular biology
into a unifying *Python* framework.
It can handle the complete molecular biology workflow
for sequence and macromolecular structure data:

   - Downloading data from biological databases
   - Loading data from structure and sequence files
   - Analyzing and modifying the data
   - Saving the data back to file
   - Interfacing external applications

The internal structure and sequence representations are based on *NumPy*
`ndarray` objects.
Vectorization and *Cython* based C-extensions render most operations
highly efficient.

Additionally, the package aims for simple usability and extensibility:
The objects representing structures and sequences can be indexed and sliced
like a `ndarray`.
Even the actual internal `ndarray` instances are easily accessible allowing
advanced users to implement their own algorithms upon the existing types.

If you use *Biotite* in a scientific publication, please cite:

| Kunzmann, P. & Hamacher, K. BMC Bioinformatics (2018) 19:346.
| `<https://doi.org/10.1186/s12859-018-2367-z>`_


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

.. image:: /examples/gallery/sequence/images/sphx_glr_sw_genome_search_001.png
   :width: 48 %
   :target: examples/gallery/sequence/sw_genome_search.html

.. image:: /examples/gallery/sequence/images/sphx_glr_avidin_alignment_001.png
   :width: 48 %
   :target: examples/gallery/sequence/avidin_alignment.html


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

.. image:: /examples/gallery/structure/images/sphx_glr_adjacency_matrix_001.png
   :width: 48 %
   :target: examples/gallery/structure/adjacency_matrix.html


**Application** subpackage
--------------------------
This subpackage provides interfaces for external software, in case *Biotite*'s
integrated functionality is not sufficient for your tasks.
These interfaces range from locally installed software (e.g. MSA software) to
web apps (e.g. BLAST).
The speciality is that the interfaces are seamless:
You do not have to write input files and read output files, you only have to
input `Python` objects and you get `Python` objects.
It is basically very similar to using normal `Python` functions.

.. image:: /examples/gallery/sequence/images/sphx_glr_lexa_conservation_001.png
   :width: 48 %
   :target: examples/gallery/sequence/lexa_conservation.html

.. image:: /examples/gallery/structure/images/sphx_glr_transketolase_sse_004.png
   :width: 48 %
   :target: examples/gallery/structure/transketolase_sse.html


**Database** subpackage
-----------------------
This subpackage is all about searching in and downloading data from biological
databases, including the probably most important ones: the *RCSB PDB* and the
*NCBI Entrez* database.


.. toctree::
   :maxdepth: 1
   :hidden:
   
   install
   tutorial/index
   apidoc/index
   examples/gallery/index
   extensions
   contribute
   logo

