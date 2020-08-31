# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite"
__author__ = "Patrick Kunzmann"
__all__ = ["File", "TextFile", "InvalidFileError"]

import abc
import io
import warnings
from .copyable import Copyable
import copy


class File(Copyable, metaclass=abc.ABCMeta):
    """
    Base class for all file classes.
    The constructor creates an empty file, that can be filled with data
    using the class specific setter methods.
    Conversely, the class method :func:`read()` reads a file from disk
    (or a file-like object from other sources).
    In order to write the instance content into a file the
    :func:`write()` method is used.
    """
    
    def __init__(self):
        # Support for deprecated instance method 'read()':
        # When creating an instance, the 'read()' class method is
        # replaced by the instance method, so that subsequent
        # 'read()' calls are delegated to the instance method
        self.read = self._deprecated_read
    
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
            representing the parsed file.
        """
        pass
        
            
    def _deprecated_read(self, file, *args, **kwargs):
        """
        Support for deprecated instance method :func:`read()`.

        Internally this calls the :func:`read()` class method and
        replaces the data in `self` with the data from the newly created
        :class:`File` object
        """
        warnings.warn(
            "Instance method 'read()' is deprecated, "
            "use class method instead",
            DeprecationWarning
        )
        cls = type(self)
        new_file = cls.read(file, *args, **kwargs)
        self.__dict__.update(new_file.__dict__)
    
    @abc.abstractmethod
    def write(self, file):
        """
        Write the contents of this :class:`File` object into a file.
        
        Parameters
        ----------
        file_name : file-like object or str
            The file to be written to.
            Alternatively a file path can be supplied.
        """
        pass
        

class TextFile(File, metaclass=abc.ABCMeta):
    """
    Base class for all line based text files.
    When reading a file, the text content is saved as list of strings,
    one for each line.
    When writing a file, this list is written into the file.
    
    Attributes
    ----------
    lines : list
        List of string representing the lines in the text file.
        PROTECTED: Do not modify from outside.
    """
    
    def __init__(self):
        super().__init__()
        self.lines = []

    @classmethod
    def read(cls, file, *args, **kwargs):
        # File name
        if isinstance(file, str):
            with open(file, "r") as f:
                lines = f.read().splitlines()
        # File object
        else:
            if not is_text(file):
                raise TypeError("A file opened in 'text' mode is required")
            lines = file.read().splitlines()
        file_object = cls(*args, **kwargs)
        file_object.lines = lines
        return file_object

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
                f.write("\n".join(self.lines) + "\n")
        else:
            if not is_text(file):
                raise TypeError("A file opened in 'text' mode is required")
            file.write("\n".join(self.lines) + "\n")
    
    def __copy_fill__(self, clone):
        super().__copy_fill__(clone)
        clone.lines = copy.copy(self.lines)
    
    def __str__(self):
        return("\n".join(self.lines))


class InvalidFileError(Exception):
    """
    Indicates that the file is not suitable for the requested action,
    either because the file does not contain the required data or
    because the file is malformed.
    """
    pass


def wrap_string(text, width):
    """
    A much simpler and hence much more efficient version of
    `textwrap.wrap()`.

    This function simply wraps the given `text` after `width`
    characters, ignoring sentences, whitespaces, etc.
    """
    lines = []
    for i in range(0, len(text), width):
        lines.append(text[i : i+width])
    return lines


def is_binary(file):
    if isinstance(file, io.BufferedIOBase):
        return True
    # for file wrappers, e.g. 'TemporaryFile'
    elif hasattr(file, "file") and isinstance(file.file, io.BufferedIOBase):
        return True
    else:
        return False


def is_text(file):
    if isinstance(file, io.TextIOBase):
        return True
    # for file wrappers, e.g. 'TemporaryFile'
    elif hasattr(file, "file") and isinstance(file.file, io.TextIOBase):
        return True
    else:
        return False
