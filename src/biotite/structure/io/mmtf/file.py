# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import numpy as np
import msgpack
import struct
from ...atoms import Atom, AtomArray, AtomArrayStack
from ....file import File
from ...error import BadStructureError
from ...filter import filter_inscode_and_altloc

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
        for key in list(self._content.keys()):
            self._content[key.decode("UTF-8")] = self._content.pop(key)
        for key in list(self._content.keys()):
            print(key)
            self[key]
    
    def write(self, file_name):
        """
        Not implemented yet.        
        """
        raise NotImplementedError()
    
    def __getitem__(self, key):
        data = self._content[key]
        if isinstance(data, bytes) and data[0] == 0:
            # MMTF specific format -> requires decoding
            codec     = struct.unpack(">i", data[0:4 ])[0]
            length    = struct.unpack(">i", data[4:8 ])[0]
            param     = struct.unpack(">i", data[8:12])[0]
            raw_data  = data[12:]
            print(codec)
        else:
            return data