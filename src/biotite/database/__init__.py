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
Then IDs can be given to the :func:`fetch()` function to download the
associated files.

.. currentmodule:: biotite

If the file is only needed temporarily, they can be stored in a
temporary directory by using :func:`biotite.temp_dir()` or
:func:`biotite.temp_file()` as path name.
"""

__name__ = "biotite.database"
__author__ = "Patrick Kunzmann"

from .error import *