# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for reading and writing structure related data.

Macromolecular structure files (PDB, PDBx/mmCIF, BinaryCIF, etc.) and
small molecule files (MOL, SDF, etc.) can be used
to load an :class:`AtomArray` or :class:`AtomArrayStack`.

Since the data model for the :class:`AtomArray` and
:class:`AtomArrayStack` class does not support duplicate atoms,
only one *altloc* can be chosen for each atom. Hence, the amount of
atoms may be lower in the atom array (stack) than in respective
structure file.

The recommended format for reading structure files is *BinaryCIF*.
It has by far the shortest parsing time and file size.

Besides the mentioned structure formats, common trajectory formats can be
loaded as well.
"""

__name__ = "biotite.structure.io"
__author__ = "Patrick Kunzmann"

from .general import *
from .trajfile import *
