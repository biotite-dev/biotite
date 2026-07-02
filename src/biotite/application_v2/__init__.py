# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A redesigned subpackage that provides interfaces for external software.

In contrast to :mod:`biotite.application`, an :class:`Application` is a
reusable, stateless handle to the wrapped software.
A run is launched by calling the respective method (e.g. ``align()``)
with all required parameters, which returns a :class:`Future`.
The actual result is obtained by calling :meth:`Future.result()`.
"""

__name__ = "biotite.application_v2"
__author__ = "Patrick Kunzmann"

from .base import *
from .localapp import *
from .msa import *
