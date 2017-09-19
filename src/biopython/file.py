# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import abc

__all__ = ["File", "TextFile"]

class File(metaclass=abc.ABCMeta):
    """
    Base class for all file classes. Every file class is
    instantiated without arguments. In order to fill the instance
    with content, either a file is read using the `read()` method,
    or the instance is directly modified with class specific setter
    methods. In order to write the instance content into a file the
    `write()` method is used.
    """
    
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def read(self, file_name):
        pass
    
    @abc.abstractmethod
    def write(self, file_name):
        pass
    
    @abc.abstractmethod
    def copy(self):
        """
        Copy the contents of the file.
        
        Returns
        -------
        copy : File
            A copy of the file.
        """
        pass
        

class TextFile(File, metaclass=abc.ABCMeta):
    """
    Base class for all text file classes. When reading a file, the text
    content is saved as list of strings. When writing a file, the list
    is written into the file.
    
    Attributes
    ----------
    _lines : list
        List of string representing the lines in the text file.
    """
    
    def __init__(self):
        self._lines = []
    
    def read(self, file_name):
        """
        Read the lines of the given text file.
        
        Parameters
        ----------
        file_name : str
            The name of the file to be read.
        """
        with open(file_name, "r") as f:
            str_data = f.read()
        self._lines = str_data.split("\n")
    
    def write(self, file_name):
        """
        Write the content to a given text file.
        
        Parameters
        ----------
        file_name : str
            The name of the file to be written to.
        """
        with open(file_name, "w") as f:
            f.writelines([line+"\n" for line in self._lines])
