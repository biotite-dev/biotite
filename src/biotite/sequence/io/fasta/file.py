# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.io.fasta"
__author__ = "Patrick Kunzmann"
__all__ = ["FastaFile"]

from ....file import TextFile, InvalidFileError, wrap_string
from collections import OrderedDict
from collections.abc import MutableMapping


class FastaFile(TextFile, MutableMapping):
    """
    This class represents a file in FASTA format.
    
    A FASTA file contains so called *header* lines, beginning with
    ``>``, that describe following sequence.
    The corresponding sequence starts at the line after the header line
    and ends at the next header line or at the end of file.
    The header along with its sequence forms an entry.
    
    This class is used in a dictionary like manner, implementing the
    :class:`MutableMapping` interface:
    Headers (without the leading ``>``) are used as keys,
    and strings containing the sequences are the corresponding values.
    Entries can be accessed using indexing,
    ``del`` deletes the entry at the given index.

    Parameters
    ----------
    chars_per_line : int, optional
        The number characters in a line containing sequence data
        after which a line break is inserted.
        Only relevant, when adding sequences to a file.
        Default is 80.
    
    Examples
    --------
    
    >>> import os.path
    >>> file = FastaFile()
    >>> file["seq1"] = "ATACT"
    >>> print(file["seq1"])
    ATACT
    >>> file["seq2"] = "AAAATT"
    >>> print(file)
    >seq1
    ATACT
    >seq2
    AAAATT
    >>> print(dict(file.items()))
    {'seq1': 'ATACT', 'seq2': 'AAAATT'}
    >>> for header, seq in file.items():
    ...     print(header, seq)
    seq1 ATACT
    seq2 AAAATT
    >>> del file["seq1"]
    >>> print(dict(file.items()))
    {'seq2': 'AAAATT'}
    >>> file.write(os.path.join(path_to_directory, "test.fasta"))
    """
    
    def __init__(self, chars_per_line=80):
        super().__init__()
        self._chars_per_line = chars_per_line
        self._entries = OrderedDict()
    
    @classmethod
    def read(cls, file, chars_per_line=80):
        """
        Read a FASTA file.
        
        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively a file path can be supplied.
        chars_per_line : int, optional
            The number characters in a line containing sequence data
            after which a line break is inserted.
            Only relevant, when adding sequences to a file.
            Default is 80.
        
        Returns
        -------
        file_object : FastaFile
            The parsed file.
        """
        file = super().read(file, chars_per_line)
        # Filter out empty and comment lines
        file.lines = [line for line in file.lines
                      if len(line.strip()) != 0 and line[0] != ";"]
        if len(file.lines) == 0:
            raise InvalidFileError("File is empty or contains only comments")
        file._find_entries()
        return file
        
    def __setitem__(self, header, seq_str):
        if not isinstance(header, str):
            raise IndexError(
                "'FastaFile' only supports header strings as keys"
            )
        if not isinstance(seq_str, str):
            raise TypeError("'FastaFile' only supports sequence strings "
                             "as values")
        # Create lines for new header and sequence (with line breaks)
        new_lines = [">" + header.replace("\n","").strip()] + \
                    wrap_string(seq_str, width=self._chars_per_line)
        if header in self:
            # Delete lines of entry corresponding to the header,
            # if existing
            del self[header]
            self.lines += new_lines
            self._find_entries()
        else:
            # Simply append lines
            # Add entry in a more efficient way than '_find_entries()'
            # for this simple case
            self._entries[header] = (
                len(self.lines),
                len(self.lines) + len(new_lines)
            )
            self.lines += new_lines
    
    def __getitem__(self, header):
        if not isinstance(header, str):
            raise IndexError(
                "'FastaFile' only supports header strings as keys"
            )
        start, stop = self._entries[header]
        # Concatenate sequence string from following lines
        seq_string = "".join(
            [line.strip() for line in self.lines[start+1 : stop]]
        )
        return seq_string
    
    def __delitem__(self, header):
        start, stop = self._entries[header]
        del self.lines[start:stop]
        del self._entries[header]
        self._find_entries()
    
    def __len__(self):
        return len(self._entries)
    
    def __iter__(self):
        return self._entries.__iter__()
    
    def __contains__(self, identifer):
        return identifer in self._entries
    
    def _find_entries(self):
        if len(self.lines) > 0 and self.lines[0][0] != ">":
            raise InvalidFileError(
                f"File starts with '{self.lines[0][0]}' instead of '>'"
            )
        
        header_i = []
        for i, line in enumerate(self.lines):
            if line[0] == ">":
                header_i.append(i)
        
        self._entries = OrderedDict()
        for j in range(len(header_i)):
            # Remove leading '>' from header
            header = self.lines[header_i[j]].strip()[1:]
            start = header_i[j]
            if j < len(header_i) -1:
                # Header in mid or start of file
                # -> stop is start of next header
                stop = header_i[j+1]
            else:
                # Last header -> entry stops at end of file
                stop = len(self.lines)
            self._entries[header] = (start, stop)


    @staticmethod
    def read_iter(file):
        """
        Create an iterator over each sequence of the given FASTA file.
        
        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively a file path can be supplied.
        
        Yields
        ------
        header : str
            The header of the current sequence.
        seq_str : str
            The current sequence as string.
        
        Notes
        -----
        This approach gives the same results as
        `FastaFile.read(file).items()`, but is slightly faster and much
        more memory efficient.
        """
        header = None
        seq_str_list = []
        for line in TextFile.read_iter(file):
            line = line.strip()
            # Ignore empty and comment lines
            if len(line) == 0 or line[0] == ";":
                continue
            if line[0] == ">":
                # New entry
                # -> yield previous entry
                if header is not None:
                    yield header, "".join(seq_str_list)
                # Track new header and reset sequence
                header = line[1:]
                seq_str_list = []
            else:
                seq_str_list.append(line)
        # Yield final entry
        if header is not None:
            yield header, "".join(seq_str_list)
    

    @staticmethod
    def write_iter(file, items, chars_per_line=80):
        """
        Iterate over the given `items` and write each item into
        the specified `file`.

        In contrast to :meth:`write()`, the lines of text are not stored
        in an intermediate :class:`TextFile`, but are directly written
        to the file.
        Hence, this static method may save a large amount of memory if
        a large file should be written, especially if the `items`
        are provided as generator.
        
        Parameters
        ----------
        file : file-like object or str
            The file to be written to.
            Alternatively a file path can be supplied.
        items : generator or array-like of tuple(str, str)
            The entries to be written into the file.
            Each entry consists of an header string and a sequence
            string.
        chars_per_line : int, optional
            The number characters in a line containing sequence data
            after which a line break is inserted.
            Only relevant, when adding sequences to a file.
            Default is 80.

        Notes
        -----
        This method does not test, whether the given identifiers are
        unambiguous.
        """
        def line_generator():
            for item in items:
                header, seq_str = item
                if not isinstance(header, str):
                    raise IndexError(
                        "'FastaFile' only supports header strings"
                    )
                if not isinstance(seq_str, str):
                    raise TypeError(
                        "'FastaFile' only supports sequence strings"
                    )
                
                # Yield header line
                yield ">" + header.replace("\n","").strip()

                # Yield sequence line(s)
                for line in wrap_string(seq_str, width=chars_per_line):
                    yield line
    
        TextFile.write_iter(file, line_generator())