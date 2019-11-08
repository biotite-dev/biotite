# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage is used for reading and writing sequence features in the
*Generic Feature Format 3* (GFF3).

It provides the :class:`GFFFile` class, a low-level line-based
interface to this format, and high-level functions for extracting
:class:`Annotation` objects.

.. note: This package cannot create hierarchical data structures from
   GFF 3 files. This means, that you cannot directly access the the
   parent or child of a feature.
   However, the ``Id`` and ``Name`` attributes are stored in the
   qualifiers of the created :class:`Feature` objects. 
   Hence, it is possible to implement such a data structure from this
   information.
"""

__name__ = "biotite.sequence.io.gff"
__author__ = "Patrick Kunzmann"

from .file import *
from .convert import *