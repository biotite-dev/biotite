# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for obtaining all kinds of chemical information of atoms
and residues.

Attributes
----------
atom_masses : dict of str to float
    A dictionary that maps an element name (upper case) to an atom
    mass [1]_.

References
----------

.. [1] J Meija, TB Coplen, M Berglund, WA Brand, P De Bièvre,
    M Gröning, NE Holden, J Irrgeher, RD Loss, T Walczyk
    and T Prohaska
    "Atomic weights of the elements 2013 (IUPAC Technical Report)."
    Pure Appl Chem, 88, 265-291 (2016).
"""

__author__ = "Patrick Kunzmann"

from .masses import *
from .bonds import *