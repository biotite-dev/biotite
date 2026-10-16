"""
This subpackage provides an interface to the `RDKit <https://www.rdkit.org/>`_
cheminformatics package.
It allows conversion between :class:`.AtomArray` and :class:`rdkit.Chem.rdchem.Mol`
objects.
"""

__name__ = "biotite.interface.rdkit"
__author__ = "Patrick Kunzmann"

from biotite.interface.version import require_package

require_package("rdkit")

from .mol import *
