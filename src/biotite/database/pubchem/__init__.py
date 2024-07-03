# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for searching and downloading files from the *PubChem*
database.
Although *PubChem* is part of *NCBI Entrez*,
:mod:`biotite.database.entrez` is only capable of accessing
meta-information from *PubChem*.
This subpackage, on the other hand, supports searching *PubChem*
compounds based on chemical information and is able to download
structure records.
"""

__name__ = "biotite.database.pubchem"
__author__ = "Patrick Kunzmann"

from .download import *
from .query import *
from .throttle import *
