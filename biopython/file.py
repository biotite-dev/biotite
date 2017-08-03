# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import abc

class File(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def read(self, file_path):
        pass
    
    @abc.abstractmethod
    def write(self, file_path):
        pass
    
    @abc.abstractmethod
    def copy():
        pass