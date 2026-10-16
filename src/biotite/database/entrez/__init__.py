"""
A subpackage for downloading files from the NCBI Entrez database.
"""

__name__ = "biotite.database.entrez"
__author__ = "Patrick Kunzmann"

from .dbnames import *
from .download import *
from .key import *
from .query import *
