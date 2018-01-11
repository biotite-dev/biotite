# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import numpy as np
from ...atoms import Atom, AtomArray, AtomArrayStack
from ....file import File
from ...error import BadStructureError
import copy

__all__ = ["MMTFFile"]


class MMTFFile(File):
    
    def read(self, file_name):
        pass
    
    def write(self, file_name):
        pass
    
    def get_structure(self, extra_fields=[]):
        pass
        
    def set_structure(self, array):
        pass
                
            