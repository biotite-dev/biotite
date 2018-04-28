# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

"""
This subpackage is used for reading and writing an `AtomArray` or
`AtomArrayStack` using the binary MMTF format. This format features a
smaller file size and a highly increased I/O operation performance,
than the text based file formats.
"""

from .file import *
from .convertfile import *
from .convertarray import *