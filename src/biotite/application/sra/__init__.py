# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for obtaining sequencing data from the *NCBI*
*sequence read archive* (SRA).

It comprises two central classes:
:class:`FastqDumpApp` downloads sequence reads in FASTQ format.
If only sequences (and no scores) are required :class:`FastaDumpApp`
writes sequence reads into FASTA format.
"""

__name__ = "biotite.application.sra"
__author__ = "Patrick Kunzmann"

from .app import *
