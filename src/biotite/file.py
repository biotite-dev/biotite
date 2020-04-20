# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite"
__author__ = "Patrick Kunzmann"
__all__ = ["File", "TextFile", "InvalidFileError"]

import abc
import warnings
from .copyable import Copyable
import copy


class File(Copyable, metaclass=abc.ABCMeta):
    """
    Base class for all file classes. Every file class is
    instantiated without arguments. In order to fill the instance
    with content, either a file is read using the `read()` method,
    or the instance is directly modified with class specific setter
    methods. In order to write the instance content into a file the
    `write()` method is used.
    """
    
    def __init__(self):
        self.read = self._deprecated_read
    

    #def __getattribute__(self, name):
    #    if name == "read":
    #        return File._deprecated_read
    
    @classmethod
    @abc.abstractmethod
    def read(cls, file):
        """
        Parse a file (or file-like object).
        
        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively a file path can be supplied.
        
        Returns
        -------
        file_object : File
            An instance from the respective :class:`File` subclass
            representing the read file.
        """
        pass
        
            
    def _deprecated_read(self, file, *args, **kwargs):
        warnings.warn(
            "Instance method 'read()' is deprecated, "
            "use static method instead",
            DeprecationWarning
        )
        cls = type(self)
        new_file = cls.read(file, *args, **kwargs)
        self.__dict__.update(new_file.__dict__)
    
    @abc.abstractmethod
    def write(self, file):
        """
        Write the contents of this object into a file
        (or file-like object).
        
        Parameters
        ----------
        file_name : file-like object or str
            The file to be written to.
            Alternatively a file path can be supplied.
        """
        pass
        

class TextFile(File, metaclass=abc.ABCMeta):
    """
    Base class for all line based text file classes.
    When reading a file, the text content is saved as list of strings,
    one for each line.
    When writing a file, the list is written into the file.
    
    Attributes
    ----------
    lines : list
        List of string representing the lines in the text file.
        PROTECTED: Do not modify from outside.
    """
    
    def __init__(self):
        pass

    @classmethod
    def read(cls, file):
        # File name
        if isinstance(file, str):
            with open(file, "r") as f:
                lines = f.read().splitlines()
        # File object
        else:
            lines = file.read().splitlines()
        return cls.parse(lines)
    
    @staticmethod
    @abc.abstractmethod
    def parse(lines):
        """
        Parse the lines of the text file and return a :class:`TextFile`
        object of the corresponding subclass.

        PROTECTED: Overwrite when inheriting. Do not call from outside.
        """
        pass

    def write(self, file):
        """
        Write the contents of this object into a file
        (or file-like object).
        
        Parameters
        ----------
        file_name : file-like object or str
            The file to be written to.
            Alternatively a file path can be supplied.
        """
        if isinstance(file, str):
            with open(file, "w") as f:
                f.write("\n".join(self._lines) + "\n")
        else:
           file.write("\n".join(self._lines) + "\n")
    
    @property
    @abc.abstractmethod
    def lines():
        pass
    
    def __copy_fill__(self, clone):
        super().__copy_fill__(clone)
        clone.lines = copy.copy(self._lines)
    
    def __str__(self):
        return("\n".join(self._lines))


class InvalidFileError(Exception):
    """
    Indicates that the file is not suitable for the requested action.
    """
    pass