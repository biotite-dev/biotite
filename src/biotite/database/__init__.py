# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for fetching data from online databases.

Each subpackage of the database package contains an interface for a
specific database.

All subpackages provide at least two functions:
:func:`search()` is used to search for IDs (e.g. PDB ID) that match the
given search parameters.
The search parameters are usually abstracted by the respective
:class:`Query` objects.
Then the obtained IDs can be given to the :func:`fetch()` function to
download the associated files.
"""

__name__ = "biotite.database"
__author__ = "Patrick Kunzmann"

from .error import *
