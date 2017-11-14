# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This subpackage is used for reading and writing an `AtomArray` or
`AtomArrayStack` using the popular PDB format. Since this format has
some limitations, it will be completely replaced someday by the modern
PDBx/mmCIF format. Therefore this subpackage should only be used, if
usage of the PDBx/mmCIF format is not suitable (e.g. when interfacing
other software). In all other cases, usage of the `PDBx` subpackage is
encouraged.
"""

from .file import *