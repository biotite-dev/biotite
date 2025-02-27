# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite"
__author__ = "Patrick Kunzmann"
__all__ = [
    "File",
    "TextFile",
    "InvalidFileError",
    "SerializationError",
    "DeserializationError",
]

import abc
import copy
import io
from os import PathLike
from biotite.copyable import Copyable


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
        file : File
            An instance from the respective :class:`File` subclass
            representing the parsed file.
        """
        pass

    @abc.abstractmethod
    def write(self, file):
        """
        Write the contents of this :class:`File` object into a file.

        Parameters
        ----------
        file : file-like object or str
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
        if is_open_compatible(file):
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

    @staticmethod
    def read_iter(file):
        """
        Create an iterator over each line of the given text file.

        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively a file path can be supplied.

        Yields
        ------
        line : str
            The current line in the file.
        """
        # File name
        if is_open_compatible(file):
            with open(file, "r") as f:
                yield from f
        # File object
        else:
            if not is_text(file):
                raise TypeError("A file opened in 'text' mode is required")
            yield from file

    def write(self, file):
        """
        Write the contents of this object into a file
        (or file-like object).

        Parameters
        ----------
        file : file-like object or str
            The file to be written to.
            Alternatively a file path can be supplied.
        """
        if is_open_compatible(file):
            with open(file, "w") as f:
                f.write("\n".join(self.lines) + "\n")
        else:
            if not is_text(file):
                raise TypeError("A file opened in 'text' mode is required")
            file.write("\n".join(self.lines) + "\n")

    @staticmethod
    def write_iter(file, lines):
        """
        Iterate over the given `lines` of text and write each line into
        the specified `file`.

        In contrast to :meth:`write()`, each line of text is not stored
        in an intermediate :class:`TextFile`, but is directly written
        to the file.
        Hence, this static method may save a large amount of memory if
        a large file should be written, especially if the `lines`
        are provided as generator.

        Parameters
        ----------
        file : file-like object or str
            The file to be written to.
            Alternatively a file path can be supplied.
        lines : generator or array-like of str
            The lines of text to be written.
            Must not include line break characters.
        """
        if is_open_compatible(file):
            with open(file, "w") as f:
                for line in lines:
                    f.write(line + "\n")
        else:
            if not is_text(file):
                raise TypeError("A file opened in 'text' mode is required")
            for line in lines:
                file.write(line + "\n")

    def __copy_fill__(self, clone):
        super().__copy_fill__(clone)
        clone.lines = copy.copy(self.lines)

    def __str__(self):
        return "\n".join(self.lines)


class InvalidFileError(Exception):
    """
    Indicates that the file is not suitable for the requested action,
    either because the file does not contain the required data or
    because the file is malformed.
    """

    pass


class SerializationError(Exception):
    pass


class DeserializationError(Exception):
    pass


def wrap_string(text, width):
    """
    A much simpler and hence much more efficient version of
    `textwrap.wrap()`.

    This function simply wraps the given `text` after `width`
    characters, ignoring sentences, whitespaces, etc.

    Parameters
    ----------
    text : str
        The text to be wrapped.
    width : int
        The maximum number of characters per line.

    Returns
    -------
    lines : list of str
        The wrapped lines.
    """
    lines = []
    for i in range(0, len(text), width):
        lines.append(text[i : i + width])
    return lines


def is_binary(file):
    if isinstance(file, io.BufferedIOBase):
        return True
    # for file wrappers, e.g. 'TemporaryFile'
    return hasattr(file, "file") and isinstance(file.file, io.BufferedIOBase)


def is_text(file):
    if isinstance(file, io.TextIOBase):
        return True
    # for file wrappers, e.g. 'TemporaryFile'
    return hasattr(file, "file") and isinstance(file.file, io.TextIOBase)


def is_open_compatible(file):
    return isinstance(file, (str, bytes, PathLike))
