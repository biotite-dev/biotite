"""
This subpackage provides interfaces to other Python packages in the bioinformatics
ecosystem.
Its purpose is to convert between native Biotite objects, such as :class:`.AtomArray`
and :class:`.Sequence`, and the corresponding objects in the respective interfaced
package.
In contrast to :mod:`biotite.application_v2`, where an entire application run is handled
under the hood, :mod:`biotite.interface` only covers the object conversion, allowing
for more flexibility.
"""

__name__ = "biotite.interface"
__author__ = "Patrick Kunzmann"

from .warning import *
