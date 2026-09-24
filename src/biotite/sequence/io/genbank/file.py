from __future__ import annotations

__name__ = "biotite.sequence.io.genbank"
__author__ = "Patrick Kunzmann"
__all__ = ["GenBankFile", "GenBankRecord", "MultiFile"]

import copy
import io
from collections.abc import Iterator
from dataclasses import dataclass, field
from os import PathLike
from typing import IO, Self
from biotite.file import InvalidFileError, TextFile


@dataclass
class GenBankRecord:
    """
    A field of a GenBank file, comprising the field name, the content
    lines and the optional subfields.

    Attributes
    ----------
    name : str
        The field name, e.g. ``'DEFINITION'``.
    content : list of str
        The content lines of the field.
        While the content of metadata fields starts at the standard
        GenBank indentation of 12, the content of the *FEATURES*
        (contains the annotation) and *ORIGIN* (contains the sequence)
        fields starts without indentation.
    subfields : dict of str -> list of str, optional
        The subfields of the field.
        The dictionary maps subfield names to the content lines of
        the respective subfield.
        The *FEATURES* and *ORIGIN* fields have no subfields.
    """

    name: str
    content: list[str]
    subfields: dict[str, list[str]] = field(default_factory=dict)


class GenBankFile(TextFile):
    """
    This class represents a file in GenBank format (including GenPept).

    A GenBank file annotates a reference sequence with features such as
    positions of genes, promoters, etc.
    Additionally, it provides metadata further describing the file.

    A file is divided into separate fields, e.g. the *DEFINITION*
    field contains a description of the file.
    The field name starts at the beginning of a line,
    followed by the content.
    A field may contain subfields, whose name is indented.
    For example, the *SOURCE* field contains the *ORGANISM* subfield.
    Some fields may occur multiple times, e.g. the *REFERENCE* field.
    A sample GenBank file can be viewed at
    `<https://www.ncbi.nlm.nih.gov/Sitemap/samplerecord.html>`_.

    This class provides a low-level interface for parsing, editing and
    writing GenBank files.
    It works like a list of :class:`GenBankRecord` entries, where each
    record represents a field consisting of the field name, the field
    content and the subfields.

    Every entry can be obtained, set and deleted via the index operator.

    Notes
    -----
    This class does not support location identifiers with references
    to other Entrez database entries, e.g.
    ``join(1..100,J00194.1:100..202)``.

    Examples
    --------
    Create a GenBank file from scratch:

    >>> file = GenBankFile()
    >>> file.append(GenBankRecord(
    ...     "SOMEFIELD", ["One line", "A second line"],
    ...     subfields={"SUBFIELD1": ["Single Line"], "SUBFIELD2": ["Two", "lines"]}
    ... ))
    >>> print(file)
    SOMEFIELD   One line
                A second line
      SUBFIELD1 Single Line
      SUBFIELD2 Two
                lines
    //
    >>> record = file[0]
    >>> print(record.name)
    SOMEFIELD
    >>> print(record.content)
    ['One line', 'A second line']
    >>> print(record.subfields)
    {'SUBFIELD1': ['Single Line'], 'SUBFIELD2': ['Two', 'lines']}

    Adding an additional field:

    >>> file.insert(0, GenBankRecord("OTHERFIELD", ["Another line"]))
    >>> print(len(file))
    2
    >>> print(file)
    OTHERFIELD  Another line
    SOMEFIELD   One line
                A second line
      SUBFIELD1 Single Line
      SUBFIELD2 Two
                lines
    //

    Overwriting and deleting an existing field:

    >>> file[1] = GenBankRecord("NEWFIELD", ["Yet another line"])
    >>> print(file)
    OTHERFIELD  Another line
    NEWFIELD    Yet another line
    //
    >>> file[1] = GenBankRecord(
    ...     "NEWFIELD", ["Yet another line"], {"NEWSUB": ["Subfield line"]}
    ... )
    >>> print(file)
    OTHERFIELD  Another line
    NEWFIELD    Yet another line
      NEWSUB    Subfield line
    //
    >>> del file[1]
    >>> print(file)
    OTHERFIELD  Another line
    //

    Parsing fields from a real GenBank file:

    >>> import os.path
    >>> file = GenBankFile.read(os.path.join(path_to_sequences, "gg_avidin.gb"))
    >>> print(file)
    LOCUS       AJ311647                1224 bp    DNA     linear   VRT 14-NOV-2006
    DEFINITION  Gallus gallus AVD gene for avidin, exons 1-4.
    ACCESSION   AJ311647
    VERSION     AJ311647.1  GI:13397825
    KEYWORDS    AVD gene; avidin.
    SOURCE      Gallus gallus (chicken)
      ORGANISM  Gallus gallus
                Eukaryota; Metazoa; Chordata; Craniata; Vertebrata; Euteleostomi;
                Archelosauria; Archosauria; Dinosauria; Saurischia; Theropoda;
                Coelurosauria; Aves; Neognathae; Galloanserae; Galliformes;
                Phasianidae; Phasianinae; Gallus.
    REFERENCE   1
      AUTHORS   Wallen,M.J., Laukkanen,M.O. and Kulomaa,M.S.
      TITLE     Cloning and sequencing of the chicken egg-white avidin-encoding
                gene and its relationship with the avidin-related genes Avr1-Avr5
      JOURNAL   Gene 161 (2), 205-209 (1995)
       PUBMED   7665080
    REFERENCE   2
      AUTHORS   Ahlroth,M.K., Kola,E.H., Ewald,D., Masabanda,J., Sazanov,A.,
                Fries,R. and Kulomaa,M.S.
      TITLE     Characterization and chromosomal localization of the chicken avidin
                gene family
      JOURNAL   Anim. Genet. 31 (6), 367-375 (2000)
       PUBMED   11167523
    REFERENCE   3  (bases 1 to 1224)
      AUTHORS   Ahlroth,M.K.
      TITLE     Direct Submission
      JOURNAL   Submitted (09-MAR-2001) Ahlroth M.K., Department of Biological and
                Environmental Science, University of Jyvaskyla, PO Box 35,
                FIN-40351 Jyvaskyla, FINLAND
    FEATURES             Location/Qualifiers
         source          1..1224
                         /organism="Gallus gallus"
                         /mol_type="genomic DNA"
    ...
    >>> record = file[3]
    >>> print(record.name)
    VERSION
    >>> print(record.content)
    ['AJ311647.1  GI:13397825']
    >>> record = file[5]
    >>> print(record.name)
    SOURCE
    >>> print(record.content)
    ['Gallus gallus (chicken)']
    >>> print(record.subfields)
    {'ORGANISM': ['Gallus gallus', 'Eukaryota; Metazoa; Chordata; ...', ...]}
    """

    def __init__(self) -> None:
        super().__init__()
        # Add '//' as general terminator of a GenBank file
        self.lines = ["//"]
        # Field start and stop indices in list of lines
        # and names of categories
        self._field_pos: list[tuple[int, int, str]] = []
        self._find_field_indices()

    @classmethod
    def read(cls, file: PathLike[str] | str | IO[str]) -> Self:
        """
        Read a GenBank file.

        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively a file path can be supplied.

        Returns
        -------
        file_object : GenBankFile
            The parsed file.
        """
        gb_file = super().read(file)
        gb_file._find_field_indices()
        return gb_file

    def get_fields(self, name: str) -> list[GenBankRecord]:
        """
        Get all *GenBank* fields associated with a given field name.

        Parameters
        ----------
        name : str
            The field name.

        Returns
        -------
        fields : list of GenBankRecord
            A list containing the fields.
            For most field names, the list will only contain one
            element, but fields like *REFERENCE* are an exception.
        """
        indices = self.get_indices(name)
        return [self[i] for i in indices]

    def get_indices(self, name: str) -> list[int]:
        """
        Get the indices to all *GenBank* fields associated with a given
        field name.

        Parameters
        ----------
        name : str
            The field name.

        Returns
        -------
        fields : list of int
            A list of indices.
            For most field names, the list will only contain one
            element, but fields like *REFERENCE* are an exception.
        """
        name = name.upper()
        indices = []
        for i, (_, _, fname) in enumerate(self._field_pos):
            if fname == name:
                indices.append(i)
        return indices

    def set_field(self, record: GenBankRecord) -> None:
        """
        Set a *GenBank* field with the given content.

        If the field already exists in the file, the field is
        overwritten, otherwise a new field is created at the end of
        the file.

        Parameters
        ----------
        record : GenBankRecord
            The field to set.

        Raises
        ------
        InvalidFileError
            If the field occurs multiple times in the file.
            In this case it is ambiguous which field to overwrite.
        """
        name = record.name.upper()
        indices = self.get_indices(name)
        if len(indices) > 1:
            raise InvalidFileError(f"File contains multiple '{name}' fields")
        elif len(indices) == 1:
            # Replace existing entry
            index = indices[0]
            self[index] = record
        else:
            # Add new entry as no entry exists yet
            self.append(record)

    def __getitem__(self, index: int) -> GenBankRecord:
        index = self._translate_idx(index)
        start, stop, name = self._field_pos[index]

        if name in ["FEATURES", "ORIGIN"]:
            # For those two fields return the complete lines,
            # beginning with the line after the field name
            content = self._get_field_content(start + 1, stop, indent=0)
            subfield_dict: dict[str, list[str]] = {}

        else:
            # For all metadata fields use the
            # standard GenBank indentation (=12)
            # Find subfields
            subfield_dict = {}
            subfield_start = None
            first_subfield_start = None
            header = ""
            for i in range(start + 1, stop):
                line = self.lines[i]
                if len(line) != 0 and line[:12].strip() != "":
                    # New header -> new subfield
                    if first_subfield_start is None:
                        first_subfield_start = i
                    # Store previous subfield
                    if subfield_start is not None:
                        subfield_dict[header] = self._get_field_content(
                            subfield_start, i, indent=12
                        )
                    header = line[:12].strip()
                    subfield_start = i
            # Store last subfield
            if subfield_start is not None:
                subfield_dict[header] = self._get_field_content(
                    subfield_start, stop, indent=12
                )
            # Only include lines in field content,
            # that are not part of a subfield
            if first_subfield_start is not None:
                stop = first_subfield_start
            content = self._get_field_content(start, stop, indent=12)

        return GenBankRecord(name, content, subfield_dict)

    def __setitem__(self, index: int, record: GenBankRecord) -> None:
        index = self._translate_idx(index)
        inserted_lines = self._to_lines(record)

        # Stop of field to be replaced is start of new field
        start, old_stop, _ = self._field_pos[index]
        # If not the last element is set,
        # the following lines need to be added, too
        if old_stop is not len(self.lines):
            follow_lines = self.lines[old_stop:]
        else:
            follow_lines = []
        self.lines = self.lines[:start] + inserted_lines + follow_lines
        # Shift the start/stop indices of the following fields
        # by the amount of created fields
        shift = len(inserted_lines) - (old_stop - start)
        for i in range(index + 1, len(self._field_pos)):
            old_start, old_stop, fname = self._field_pos[i]
            self._field_pos[i] = old_start + shift, old_stop + shift, fname
        # Add new entry
        self._field_pos[index] = (
            start,
            start + len(inserted_lines),
            record.name.upper(),
        )

    def __delitem__(self, index: int) -> None:
        index = self._translate_idx(index)
        start, stop, _ = self._field_pos[index]
        # Shift the start/stop indices of the following fields
        # by the amount of deleted fields
        shift = stop - start
        for i in range(index, len(self._field_pos)):
            old_start, old_stop, name = self._field_pos[i]
            self._field_pos[i] = old_start - shift, old_stop - shift, name
        del self.lines[start:stop]
        del self._field_pos[index]

    def __len__(self) -> int:
        return len(self._field_pos)

    def insert(self, index: int, record: GenBankRecord) -> None:
        """
        Insert a *GenBank* field at the given position.

        Parameters
        ----------
        index : int
            The new field is inserted before the current field at this
            index.
            If the index is after the last field, the new field
            is appended to the end of the file.
        record : GenBankRecord
            The field to insert.
        """
        index = self._translate_idx(index, length_exclusive=False)
        inserted_lines = self._to_lines(record)

        # Stop of previous field is start of new field
        if index == 0:
            start = 0
        else:
            _, start, _ = self._field_pos[index - 1]
        # If the new lines are not inserted at the end,
        # the following lines need to be added, too
        if start is not len(self.lines):
            follow_lines = self.lines[start:]
        else:
            follow_lines = []
        self.lines = self.lines[:start] + inserted_lines + follow_lines
        # Shift the start/stop indices of the following fields
        # by the amount of created fields
        shift = len(inserted_lines)
        for i in range(index, len(self._field_pos)):
            old_start, old_stop, fname = self._field_pos[i]
            self._field_pos[i] = old_start + shift, old_stop + shift, fname
        # Add new entry
        self._field_pos.insert(
            index, (start, start + len(inserted_lines), record.name.upper())
        )

    def append(self, record: GenBankRecord) -> None:
        """
        Create a new *GenBank* field at the end of the file.

        Parameters
        ----------
        record : GenBankRecord
            The field to append.
        """
        self.insert(len(self), record)

    def _find_field_indices(self) -> None:
        """
        Identify the start and exclusive stop indices of lines
        corresponding to a field name for all fields in the file.
        """
        start = None
        name = ""
        self._field_pos = []
        for i, line in enumerate(self.lines):
            # Check if line contains a new major field
            # (Header beginning from first column)
            if len(line) != 0 and line[0] != " ":
                if line[:2] != "//":
                    stop = i
                    if start is not None:
                        # Store previous field
                        self._field_pos.append((start, stop, name))
                    start = i
                    name = line[0:12].strip()
                else:
                    # '//' means end of file
                    # -> Store last field
                    if start is not None:
                        stop = i
                        self._field_pos.append((start, stop, name))

    def _get_field_content(self, start: int, stop: int, indent: int) -> list[str]:
        if indent == 0:
            return self.lines[start:stop]
        else:
            return [line[12:] for line in self.lines[start:stop]]

    def _to_lines(self, record: GenBankRecord) -> list[str]:
        """
        Convert the field name, field content und subfield dictionary
        into text lines
        """
        if not isinstance(record, GenBankRecord):
            raise TypeError(
                f"Expected 'GenBankRecord', but got '{type(record).__name__}'"
            )
        name = record.name.strip().upper()
        content = record.content
        if len(name) == 0:
            raise ValueError("Must give a non emtpy name")
        subfields = {
            subfield_name.upper().strip(): subfield_lines
            for subfield_name, subfield_lines in record.subfields.items()
        }

        # Create lines for new field
        if name == "FEATURES":
            # Header line plus all actual feature lines
            lines = copy.copy(content)
            lines.insert(0, "FEATURES" + " " * 13 + "Location/Qualifiers")
        elif name == "ORIGIN":
            # Header line plus all actual sequence lines
            lines = copy.copy(content)
            lines.insert(0, "ORIGIN")
        else:
            name_column = []
            content_column = []
            # Create a line for the field name and empty lines
            # for each additional line required by the content
            name_column += [name] + [""] * (len(content) - 1)
            content_column += content
            for subfield_name, subfield_lines in subfields.items():
                name_column += ["  " + subfield_name] + [""] * (len(subfield_lines) - 1)
                content_column += subfield_lines
            lines = [
                f"{n_col:12}{c_col}"
                for n_col, c_col in zip(name_column, content_column)
            ]

        return lines

    def _translate_idx(self, index: int, length_exclusive: bool = True) -> int:
        """
        Check index boundaries and convert negative index to positive
        index.
        """
        if index < 0:
            new_index = len(self) + index
        else:
            new_index = index
        if length_exclusive:
            if new_index >= len(self):
                raise IndexError(f"Index {index} is out of range")
        else:
            if new_index > len(self):
                raise IndexError(f"Index {index} is out of range")
        return new_index


class MultiFile(TextFile):
    """
    This class represents a file in *GenBank* or *GenPept* format,
    that contains multiple entries, for more than one UID.

    The information for each UID are appended to each other in such a
    file.
    Objects of this class can be iterated to obtain a
    :class:`GenBankFile` for each entry in the file.

    Examples
    --------

    >>> import os.path
    >>> file_name = fetch_single_file(
    ...     ["1L2Y_A", "3O5R_A", "5UGO_A"],
    ...     os.path.join(path_to_directory, "multifile.gp"),
    ...     "protein", "gp"
    ... )
    >>> multi_file = MultiFile.read(file_name)
    >>> for gp_file in multi_file:
    ...     print(get_accession(gp_file))
    1L2Y_A
    3O5R_A
    5UGO_A
    """

    def __iter__(self) -> Iterator[GenBankFile]:
        start_i = 0
        for i in range(len(self.lines)):
            line = self.lines[i]
            if line.strip() == "//":
                # Create file with lines corresponding to that file
                file_content = "\n".join(self.lines[start_i : i + 1])
                file = GenBankFile.read(io.StringIO(file_content))
                # Reset file start index
                start_i = i
                yield file
