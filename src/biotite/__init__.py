# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This is the top-level package of *Biotite*.
Although it does not provide useful functionality for most users,
it does provide utilities and base classes used by a lot of *Biotite*'s
modules.
"""

__version__ = "0.34.0"
__name__ = "biotite"
__author__ = "Patrick Kunzmann"

from .file import *
from .temp import *
from .copyable import * 
from .visualize import * 
