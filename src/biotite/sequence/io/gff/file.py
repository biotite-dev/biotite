# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["GFFFile"]

import copy
import string
from collections.abc import MutableSequence
from urllib.parse import quote, unquote
import warnings
from ....file import TextFile, InvalidFileError
from ...annotation import Location


# All punctuation characters except
# percent, semicolon, equals, ampersand, comma
_NOT_QUOTED = "".join(
    [char for char in string.punctuation if char not in "%;=&,"]
) + " "


class GFFFile(TextFile, MutableSequence):
    """
    This class represents a file in *Generic Feature Format 3*
    (`GFF3 <https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md>`_)
    format.

    Similar to GenBank files, GFF3 files contain information about
    features of a reference sequence, but in a more concise and better
    parsable way.
    However, it does not provide additional meta information.

    This class serves as low-level API for accessing GFF3 files.
    It is used as a sequence of entries:
    Each entry consists of of values corresponding to the 9 columns of
    GFF3:

    ==============  ===========================  ====================================================================
    **seqid**       ``str``                      The ID of the reference sequence
    **source**      ``str``                      Source of the data (e.g. ``Genbank``)
    **type**        ``str``                      Type of the feature (e.g. ``CDS``)
    **start**       ``int``                      Start coordinate of feature on the reference sequence
    **end**         ``int``                      End coordinate of feature on the reference sequence
    **score**       ``float or None``            Optional score (e.g. an E-value)
    **strand**      ``Location.Strand or None``  Strand of the feature, ``None`` if feature is not stranded
    **phase**       ``int or None``              Shift according top the reading frame, ``None`` for non-CDS features
    **attributes**  ``dict``                     Additional properties of the feature
    ==============  ===========================  ====================================================================

    Note that the entry index is not equal to the line index, because
    GFF3 files

    Notes
    -----
    Although the GFF3 specification allows mixing in reference sequence
    data in FASTA format via the ``##FASTA`` directive, this class does
    not support extracting the sequence information.
    The content after the ``##FASTA`` directive is simply ignored.
    Please provide the sequence via a separate file or read the FASTA
    data directly via the `lines` attribute:
    
    >>> import os.path
    >>> from io import StringIO
    >>> gff_file = GFFFile()
    >>> gff_file.read(os.path.join(path_to_sequences, "indexing_test.gff3"))
    >>> fasta_start_index = None
    >>> for directive, line_index in gff_file.directives():
    ...     if directive == "FASTA":
    ...         fasta_start_index = line_index + 1
    >>> fasta_data = StringIO("\\n".join(gff_file.lines[fasta_start_index:]))
    >>> fasta_file = FastaFile()
    >>> fasta_file.read(fasta_data)
    >>> for seq_string in fasta_file.values():
    ...     print(seq_string[:60] + "...")
    TACGTAGCTAGCTGATCGATGTTGTGTGTATCGATCTAGCTAGCTAGCTGACTACACAAT...
    """
    
    def __init__(self):
        super().__init__()
        # Maps entry indices to line indices
        self._entries = None
        # Stores the directives as (directive text, line index)-tuple
        self._directives = None
        # Stores whether the file has FASTA data
        self._has_fasta = None
        self._index_entries()
        self.append_directive("gff-version", "3")
    
    def read(self, file):
        super().read(file)
        self._index_entries()
    
    def insert(self, index, seqid, source, type, start, end,
               score, strand, phase, attributes):
        if index == len(self):
            self.append(seqid, source, type, start, end,
                        score, strand, phase, attributes)
        else:
            line_index = self._entries[index]
            line = GFFFile._create_line(
                seqid, source, type, start, end,
                score, strand, phase, attributes
            )
            self.lines.insert(line_index, line)
            self._index_entries()
    
    def append(self, seqid, source, type, start, end,
               score, strand, phase, attributes):
        if self._has_fasta:
            raise NotImplementedError(
                "Cannot append feature entries, "
                "as this file contains additional FASTA data"
            )
        line = GFFFile._create_line(
            seqid, source, type, start, end, score, strand, phase, attributes
        )
        self.lines.append(line)
        # Fast update of entry index by adding last line
        self._entries.append(len(self.lines) - 1)
    
    def append_directive(self, directive, *args):
        if directive.startswith("FASTA"):
            raise NotImplementedError(
                "Adding FASTA information is not supported"
            )
        directive_line = "##" + directive + " " + " ".join(args)
        self._directives.append((directive_line[2:], len(self.lines)))
        self.lines.append(directive_line)
    
    def directives(self):
        # Sort in line order
        return sorted(self._directives, key=lambda directive: directive[1])
        
    def __setitem__(self, index, item):
        seqid, source, type, start, end, score, strand, phase, attrib = item
        line = GFFFile._create_line(
            seqid, source, type, start, end, score, strand, phase, attrib
        )
        line_index = self._entries[index]
        self.lines[line_index] = line

    
    def __getitem__(self, index):
        if (index >= 0 and  index >= len(self)) or \
           (index <  0 and -index >  len(self)):
                raise IndexError(
                    f"Index {index} is out of range for GFFFile with "
                    f"{len(self)} entries"
                )
        
        line_index = self._entries[index]
        # Columns are tab separated
        s = self.lines[line_index].split("\t")
        if len(s) != 9:
            raise InvalidFileError(f"Expected 9 columns, but got {len(s)}")
        seqid, source, type, start, end, score, strand, phase, attrib = s

        seqid = unquote(seqid)
        source = unquote(source)
        type = unquote(type)
        start = int(start)
        end = int(end)
        score = None if score == "." else float(score)
        if strand == "+":
            strand = Location.Strand.FORWARD
        elif strand == "-":
            strand = Location.Strand.REVERSE
        else:
            strand = None
        phase = None if phase == "." else int(phase)
        attrib = GFFFile._parse_attributes(attrib)

        return seqid, source, type, start, end, score, strand, phase, attrib
    
    def __delitem__(self, index):
        line_index = self._entries[index]
        del self.lines[line_index]
        self._index_entries()
    
    def __len__(self):
        return len(self._entries)
    
    def _index_entries(self):
        """
        Parse the file for comment and directive lines.
        Count these lines cumulatively, so that entry indices can be
        mapped onto line indices.
        Additionally track the line index of directive lines.
        """
        self._directives = []
        # Worst case allocation -> all lines contain actual entries
        self._entries = [None] * len(self.lines)
        self._has_fasta = False
        entry_counter = 0
        for line_i, line in enumerate(self.lines):
            if len(line) == 0 or line[0] == " ":
                # Empty line -> do nothing
                pass
            elif line.startswith("#"):
                # Comment or directive
                if line.startswith("##"):
                    # Directive
                    # Omit the leading '##'
                    self._directives.append((line[2:], line_i))
                    if line[2:] == "FASTA":
                        self._has_fasta = True
                        # This parser does not support bundled FASTA
                        # data
                        warnings.warn(
                            "Biotite does not support FASTA data mixed into "
                            "GFF files, the FASTA data will be ignored"
                        )
                        # To ignore the following FASTA data, stop
                        # parsing at this point
                        break
            else:
                # Actual entry
                self._entries[entry_counter] = line_i
                entry_counter += 1
        # Trim to correct size
        self._entries = self._entries[:entry_counter]

    @staticmethod
    def _create_line(seqid, source, type, start, end,
                     score, strand, phase, attributes):
        """
        Create a line for a newly created entry.
        """
        seqid = quote(seqid.strip(), safe=_NOT_QUOTED) \
                if seqid is not None else "."
        source = quote(source.strip(), safe=_NOT_QUOTED) \
                 if source is not None else "."
        type = type.strip()

        # Perform checks
        if len(seqid) == 0:
            raise ValueError("'seqid' must not be empty")
        if len(source) == 0:
            raise ValueError("'source' must not be empty")
        if len(type) == 0:
            raise ValueError("'type' must not be empty")
        if len(attributes) == 0:
            raise ValueError("'attributes' must not be empty")
        if seqid[0] == ">":
            raise ValueError("'seqid' must not start with '>'")
        
        score = str(score) if score is not None else "."
        if strand == Location.Strand.FORWARD:
            strand = "+"
        elif strand == Location.Strand.REVERSE:
            strand = "-"
        else:
            strand = "."
        phase = str(phase) if phase is not None else "."
        attributes = ";".join(
            [quote(key, safe=_NOT_QUOTED) + "=" + quote(val, safe=_NOT_QUOTED)
             for key, val in attributes.items()]
        )

        return "\t".join(
            [seqid, source, type, str(start), str(end),
             str(score), strand, phase, attributes]
        )
    
    @staticmethod
    def _parse_attributes(attributes):
        """
        Parse the *attributes* string into a dictionary.
        """
        attrib_dict = {}
        attrib_entries = attributes.split(";")
        for entry in attrib_entries:
            compounds = entry.split("=")
            if len(compounds) != 2:
                raise InvalidFileError(
                    f"Attribute entry '{entry}' is invalid"
                )
            key, val = compounds
            attrib_dict[unquote(key)] = unquote(val)
        return attrib_dict