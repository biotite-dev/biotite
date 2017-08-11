# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import abc

class File(metaclass=abc.ABCMeta):
    
    def __init__(self):
        self._lines = []
    
    def read(self, file_name):
        with open(file_name, "r") as f:
            str_data = f.read()
        self._lines = str_data.split("\n")
    
    def write(self, file_name):
        with open(file_name, "w") as f:
            f.writelines([line+"\n" for line in self._lines])
    
    @abc.abstractmethod
    def copy():
        pass