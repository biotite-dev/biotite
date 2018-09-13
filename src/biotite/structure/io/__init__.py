# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for reading and writing structure related data.

Macromolecular structure files (PDB, PDBx/mmCIF, MMTF) can be used to
load an `AtomArray` or `AtomArrayStack`.
Since the data model for the `AtomArray` and `AtomArrayStack` class does
not support dublicate atoms, only one *altloc* or *insertion code* can
be chosen for each residue. Hence, the amount of atoms may be lower
in the atom array (stack) than in respective structure file.

The recommended format for reading structure files is MMTF.
It has by far the shortest parsing time and file size.
Furthermore, chemical bond information can be read from MMTF files
as `BondList` instances.

Besides the mentioned structure formats, Gromacs trajectory files can be
loaded, if `mdtraj` is installed.
"""

__author__ = "Patrick Kunzmann"

from .trajfile import *
from .general import *