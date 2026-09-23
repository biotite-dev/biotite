"""
This subpackage is used for reading and writing sequencing data
using the popular FASTQ format.

This package contains the :class:`FastqFile`, which provides a
dictionary like interface to FASTQ files, with the sequence identifer
strings being the keys and the sequences and quality scores being the
values.
"""

__name__ = "biotite.sequence.io.fastq"
__author__ = "Patrick Kunzmann"

from .convert import *
from .file import *
