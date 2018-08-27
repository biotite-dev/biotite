.. This source code is part of the Biotite package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

Biotite documentation
=====================

The *Biotite* package bundles popular tasks in computational molecular biology
into an unifying *Python* framework.
It can handle the complete molecular biology workflow
for sequence and macromolecular sructure data:

   - Downloading data from biological databases
   - Loading data from popular structure and sequence files
   - Analyzing and modificating the data
   - Saving the data back to file
   - Interfacing external applications

The internal structure and sequence representations are based on *NumPy*
`ndarray` objects.
Vectorization and *Cython* based C-extensions accelerate most operations in the
package.

Additionally, the package aims for simple usability and extensibility:
The objects representing structures and sequences can be indexed and sliced
like an `ndarray`.
Even the actual internal `ndarray` instances are easily accessible allowing
advanced users to implement their own algorithms upon the existing types.

**Sequence** subpackage
-----------------------
The ``sequence`` subpackage contains functionality for working with sequence
information of any kind. The package contains by default sequence types for
nucleotides and proteins, but the alphabet-based implementation allows simple
integration of own sequence types, even if they do not rely on letters.
Beside the standard I/O operations, the package includes general purpose
functions for sequence manipulations and global/local alignments.

**Structure** subpackage
------------------------
The ``structure`` subpackage enables handling of 3D structures of biomolecules.
Simplified, a structure is represented by a list of atoms and their properties,
based on `ndarrays`. The subpackage includes read/write functionality for
different formats, structure filters, coordinate transformations, angle and
bond measurements, accessible surface area calculation, structure
superimposition and more.

**Database** subpackage
-----------------------
The ``database`` subpackage is all about downloading data from biological
databases, including the probably most important ones: the `RCSB PDB` and the
`NCBI Entrez` database.

**Application** subpackage
--------------------------
The ``application`` subpackage provides interfaces for external software.
The interfaces range from locally installed software (e.g. MSA software) to
web apps (e.g. BLAST). The speciality is that the interfaces are seamless:
You do not have to write input files and read output files, you only have to
input `Python` objects and you get `Python` objects. It is basically very
similar to using normal `Python` functions.


.. toctree::
   :maxdepth: 1
   :hidden:
   
   install
   tutorial/index
   apidoc/index
   examples/gallery/index
   extensions
   develop
   logo

