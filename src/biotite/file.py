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
        self.read = _deprecated_read
    
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
        
            
    def _deprecated_read(self, file):
        warnings.warn(
            "Instance method 'read()' is deprecated, "
            "use static method instead",
            warnings.DeprecationWarning
        )
        cls = type(self)
        return cls.read(file)
    
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
    Base class for all text file classes. When reading a file, the text
    content is saved as list of strings. When writing a file, the list
    is written into the file.
    
    Attributes
    ----------
    lines : list
        List of string representing the lines in the text file.
        PROTECTED: Do not modify from outside.
    """
    
    def __init__(self):
        self.lines = []

    @classmethod
    def read(self, file):
        # File name
        elif isinstance(args[0], str):
            with open(file, "r") as f:
                lines = file.read().splitlines()
        # File object
        else:
            lines = file.read().splitlines()
        return cls.parse(lines)

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
        def _write(file):
            nonlocal self
            # Include 'newline' at the end of file
            file.write("\n".join(self.lines) + "\n")

        if isinstance(file, str):
            with open(file, "w") as f:
                _write(f)
        else:
            _write(file)
    
    def __copy_fill__(self, clone):
        super().__copy_fill__(clone)
        clone.lines = copy.copy(self.lines)
    
    def __str__(self):
        return("\n".join(self.lines))


class InvalidFileError(Exception):
    """
    Indicates that the file is not suitable for the requested action.
    """
    pass