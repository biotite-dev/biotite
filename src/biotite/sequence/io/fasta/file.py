# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

from ....file import TextFile
import textwrap
from collections import OrderedDict

__all__ = ["FastaFile"]


class FastaFile(TextFile):
    """
    This class represents a file in FASTA format.
    
    A FASTA file contains so called *header* lines, beginning with '>',
    that describe following sequence. The corresponding sequence starts
    at the line after the header line and ends at the next header line
    or at the end of file. The header along with its sequence forms an
    entry.
    
    This class is used in a dictionary like manner,
    headers (without the leading '>') are used as keys,
    and strings containing the sequences are the corresponding values.
    Therefore entries can be accessed using indexing, `del` deletes the
    entry at the given index. In fact objects of this class can be
    casted into actual `dict` instances.
    
    Parameters
    ----------
    chars_per_line : int, optional
        The number characters in a line containing sequence data
        after which a line break is inserted. Default is 80.
    
    Examples
    --------
    
    >>> file = FastaFile()
    >>> file["seq1"] = "ATACT"
    >>> print(file["seq1"])
    ATACT
    >>> file["seq2"] = "AAAATT"
    >>> print(dict(file))
    {'seq1': 'ATACT', 'seq2': 'AAAATT'}
    >>> for header, seq in file:
    ...     print(header, seq)
    seq1 ATACT
    seq2 AAAATT
    >>> del file["seq1"]
    >>> print(dict(file))
    {'seq2': 'AAAATT'}
    >>> file.write("test.fasta")
    """
    
    def __init__(self, chars_per_line=80):
        super().__init__()
        self._chars_per_line = chars_per_line
        self._entries = OrderedDict()
    
    def read(self, file):
        super().read(file)
        # Filter out empty and comment lines
        self.lines = [line for line in self.lines
                       if len(line.strip()) != 0 and line[0] != ";"]
        self._find_entries()
        
    def __setitem__(self, header, seq_str):
        if not isinstance(header, str):
            raise IndexError(
                "'FastaFile' only supports header strings as keys"
            )
        if not isinstance(seq_str, str):
            raise IndexError("'FastaFile' only supports sequence strings "
                             "as values")
        # Delete lines of entry corresponding to the header,
        # if existing
        if header in self._entries:
            start, stop = self._entries[header]
            del self.lines[start:stop]
            del self._entries[header]
        # Append header line
        self.lines += [">" + header.replace(">","").replace("\n","").strip()]
        # Append new lines with sequence string (with line breaks)
        self.lines += textwrap.wrap(seq_str, width=self._chars_per_line)
        self._find_entries()
    
    def __getitem__(self, header):
        if not isinstance(header, str):
            raise IndexError(
                "'FastaFile' only supports header strings as keys"
            )
        start, stop = self._entries[header]
        # Concatenate sequence string from following lines
        seq_string = ""
        i = start +1
        while i < stop:
            seq_string += self.lines[i].strip()
            i += 1
        return seq_string
    
    def __delitem__(self, header):
        start, stop = self._entries[header]
        del self.lines[start:stop]
        del self._entries[header]
        self._find_entries()
    
    def __len__(self):
        return len(self._entries)
    
    def __iter__(self):
        for header in self._entries:
            yield header, self[header]
    
    def _find_entries(self):
        header_i = []
        for i, line in enumerate(self.lines):
            if line[0] == ">":
                header_i.append(i)
        self._entries = OrderedDict()
        for j in range(len(header_i)):
            # Remove '>' from header
            header = self.lines[header_i[j]].replace(">","").strip()
            start = header_i[j]
            if j < len(header_i) -1:
                # Header in mid or start of file
                # -> stop is start of next header
                stop = header_i[j+1]
            else:
                # Last header -> entry stops at end of file
                stop = len(self.lines)
            self._entries[header] = (start, stop)
    