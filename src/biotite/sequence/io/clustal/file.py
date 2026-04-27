# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.io.clustal"
__author__ = "haoyu"
__all__ = ["ClustalFile"]

import re
from collections import OrderedDict
from collections.abc import MutableMapping
from biotite.file import InvalidFileError, TextFile

_CONSENSUS_LINE = re.compile(r"[ *:.]+")


class ClustalFile(TextFile, MutableMapping):
    """
    This class represents a file in ClustalW format (``.aln``).

    A ClustalW file starts with a header line beginning with ``CLUSTAL``,
    followed by blocks of aligned sequences separated by blank lines.
    Each sequence line contains a sequence name followed by a segment of
    the gapped sequence.
    An optional consensus line, containing only ``*``, ``:``, ``.`` and
    space characters, may follow each block.

    This class is used in a dictionary-like manner, implementing the
    :class:`MutableMapping` interface:
    Sequence names are used as keys,
    and strings containing the full gapped sequences are the
    corresponding values.

    Examples
    --------

    >>> import os.path
    >>> file = ClustalFile()
    >>> file["seq1"] = "ADTRCGTARDCGTR-DRTCGRAGD"
    >>> file["seq2"] = "ADTRCGT---CGTRADRTCGRAGD"
    >>> print(file["seq1"])
    ADTRCGTARDCGTR-DRTCGRAGD
    >>> for name, seq in file.items():
    ...     print(name, seq)
    seq1 ADTRCGTARDCGTR-DRTCGRAGD
    seq2 ADTRCGT---CGTRADRTCGRAGD
    """

    def __init__(self):
        super().__init__()
        self._entries = OrderedDict()

    @classmethod
    def read(cls, file):
        """
        Read a ClustalW file.

        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively a file path can be supplied.

        Returns
        -------
        file_object : ClustalFile
            The parsed file.
        """
        file = super().read(file)
        if len(file.lines) == 0:
            raise InvalidFileError("File is empty")
        file._parse()
        return file

    def _parse(self):
        """
        Parse the lines into sequence entries.
        """
        # Validate header
        if not self.lines[0].startswith("CLUSTAL"):
            raise InvalidFileError(
                f"File does not start with 'CLUSTAL' header, "
                f"got '{self.lines[0][:20]}...'"
            )

        self._entries = OrderedDict()
        for line in self.lines[1:]:
            # Skip blank lines
            if len(line.strip()) == 0:
                continue
            # Skip consensus lines (lines containing only the consensus
            # characters ``*``, ``:``, ``.`` and space)
            if _CONSENSUS_LINE.fullmatch(line):
                continue
            # Otherwise this must be a sequence line: name followed by
            # whitespace and a sequence segment.
            parts = line.split()
            if len(parts) < 2:
                raise InvalidFileError(
                    f"Malformed sequence line, expected a name and a "
                    f"sequence segment separated by whitespace, got "
                    f"'{line.rstrip()}'"
                )
            name = parts[0]
            seq_segment = parts[1]
            if name in self._entries:
                self._entries[name] += seq_segment
            else:
                self._entries[name] = seq_segment

    def __setitem__(self, name, seq_str):
        if not isinstance(name, str):
            raise IndexError(
                "'ClustalFile' only supports sequence name strings as keys"
            )
        if not isinstance(seq_str, str):
            raise TypeError("'ClustalFile' only supports sequence strings as values")
        self._entries[name] = seq_str
        self._rebuild_lines()

    def __getitem__(self, name):
        if not isinstance(name, str):
            raise IndexError(
                "'ClustalFile' only supports sequence name strings as keys"
            )
        return self._entries[name]

    def __delitem__(self, name):
        del self._entries[name]
        self._rebuild_lines()

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return self._entries.__iter__()

    def __contains__(self, name):
        return name in self._entries

    def _rebuild_lines(self, line_length=60):
        """
        Rebuild the text lines from the stored entries.
        """
        self.lines = ["CLUSTAL W multiple sequence alignment", ""]

        if len(self._entries) == 0:
            return

        names = list(self._entries.keys())
        sequences = list(self._entries.values())
        total_len = max(len(s) for s in sequences)
        # Pad name column to accommodate the longest name + spacing
        name_width = max(len(n) for n in names)

        for start in range(0, total_len, line_length):
            for name, seq_str in zip(names, sequences):
                segment = seq_str[start : start + line_length]
                if len(segment) > 0:
                    padding = " " * (name_width - len(name) + 6)
                    self.lines.append(f"{name}{padding}{segment}")
            self.lines.append("")
