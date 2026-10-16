"""
This is the top-level package of *Biotite*.
Although it does not provide useful functionality for most users,
it does provide utilities and base classes used by a lot of *Biotite*'s
modules.
"""

__name__ = "biotite"
__author__ = "Patrick Kunzmann"

from .copyable import *
from .file import *
from .version import __version__, __version_tuple__  # noqa: F401
from .visualize import *
