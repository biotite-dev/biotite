# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage provides interfaces to other Python packages in the bioinformatics
ecosystem.
Its purpose is to convert between native Biotite objects, such as :class:`.AtomArray`
and :class:`.Sequence`, and the corresponding objects in the respective interfaced
package.
In contrast to :mod:`biotite.application`, where an entire application run is handled
under the hood, :mod:`biotite.interface` only covers the object conversion, allowing
for more flexibility.
"""

__name__ = "biotite.interface"
__author__ = "Patrick Kunzmann"

from .warning import *
