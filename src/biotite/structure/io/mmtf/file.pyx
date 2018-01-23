# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import numpy as np
from ...atoms import Atom, AtomArray, AtomArrayStack
from ....file import File
from ...error import BadStructureError
from ...filter import filter_inscode_and_altloc
import msgpack

__all__ = ["MMTFFile"]


class MMTFFile(File):
    """
    This class represents a MMTF file.
    
    This class provides only a parser for MMTF files.
    Writing MMTF files is not possible at this point.
    """
    
    def __init__(self):
        self._content = None
    
    def read(self, file_name):
        """
        Parse a MMTF file.
        
        Parameters
        ----------
        file_name : str
            The name of the file to be read.
        """
        with open(file_name, "rb") as f:
            self._content = msgpack.unpackb(f.read())
        a = [print(e) for e in self._content]
    
    def write(self, file_name):
        """
        Not implemented yet.        
        """
        raise NotImplementedError()