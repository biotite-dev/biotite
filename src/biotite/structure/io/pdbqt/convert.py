# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Some convenience functions for consistency with other ``structure.io``
subpackages.
"""

__name__ = "biotite.structure.io.pdbqt"
__author__ = "Patrick Kunzmann"
__all__ = ["get_structure", "set_structure"]


def get_structure(pdb_file, model=None, altloc="first", extra_fields=[]):
    raise NotImplementedError()


def set_structure(pdb_file, array, hybrid36=False):
    raise NotImplementedError()