# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["GenBankFile", "MultiFile"]

#import textwrap
import copy
#import re
import io
from ....file import TextFile, InvalidFileError
from collections import OrderedDict
#from ...annotation import Location, Feature, Annotation, AnnotatedSequence
#from ...seqtypes import NucleotideSequence, ProteinSequence 


class GenBankFile(TextFile):
    """
    This class represents a file in GenBank format.
    
    A GenBank file provides 3 kinds of information:
    At first it contains some metadata about the file, like
    IDs, database relations and source organism.
    Secondly it contains sequence annotations, i.e. the positions of
    a reference sequence, that fulfill certain roles, like promoters or
    coding sequences.
    Finally, the file contains optionally the reference sequence.
    
    This class provides a low-level interface to the file content.
    """

    def __init__(self):
        super().__init__()
        # Add '//' as general terminator of a GenBank file
        self.lines = ["//"]
        # Field start and stop indices in list of lines
        # and names of categories
        self._field_pos = []
        self._find_field_indices()
    
    def read(self, file):
        super().read(file)
        self._find_field_indices()
    
    def get_fields(self, name):
        name = name.upper()
        fields = []
        for i, (_, _, fname) in enumerate(self._field_pos):
            if fname == name:
                _, content, subfields = self[i]
                fields.append((content, subfields))
        return fields
    
    def get_indices(self, name):
        name = name.upper()
        indices = []
        for i, (_, _, fname) in enumerate(self._field_pos):
            if fname == name:
                indices.append(i)
        return indices
    
    def set_field(self, name, content, subfield_dict=None):
        name = name.upper()
        indices = self.get_indices(name)
        if len(indices) > 1:
            raise InvalidFileError(f"File contains multiple '{name}' fields")
        elif len(indices) == 1:
            # Replace existing entry
            index = indices[0]
            self[index] = name, content, subfield_dict
        else:
            # Add new entry as no entry exists yet
            self.append(name, content, subfield_dict)

    def __getitem__(self, index):
        index = self._translate_idx(index)
        start, stop, name = self._field_pos[index]
        
        if name in ["FEATURES", "ORIGIN"]:
            # For those two fields return the complete lines,
            # beginning with the line after the field name
            content = self._get_field_content(start+1, stop, indent=0)
            subfield_dict = OrderedDict()
        
        else:
            # For all metadata fields use the
            # standard GenBank indentation (=12)
            # Find subfields
            subfield_dict = OrderedDict()
            subfield_start = None
            first_subfield_start = None
            for i in range(start+1, stop):
                line = self.lines[i]
                # Check if line contains a new subfield
                # (Header beginning from first column)
                if len(line) != 0 and line[:12].strip() != "":
                    if first_subfield_start is None:
                        first_subfield_start = i
                    # Store previous subfield
                    if subfield_start is not None:
                        subfield_dict[header] = self._get_field_content(
                            subfield_start, i, indent=12
                        )
                    header = line[:12].strip()
                    subfield_st0art = i
            # Store last subfield
            if subfield_start is not None:
                subfield_dict[header] = self._get_field_content(
                    subfield_start, stop, indent=12
                )
            # Only include lines in field content,
            # that are not part of a subfield
            content = self._get_field_content(
                start, first_subfield_start, indent=12
            )
        
        return name, content, subfield_dict
    
    def __setitem__(self, index, item):
        index = self._translate_idx(index)
        if not isinstance(item, tuple):
            raise TypeError(
                "Expected a tuple of name, content and optionally subfields"
            )
        if len(item) == 2:
            name, content = item
            subfields = None
        elif len(item) == 3:
            name, content, subfields = item
        else:
            raise TypeError(
                "Expected a tuple of name, content and optionally subfields"
            )
        inserted_lines = self._to_lines(name, content, subfields)
        
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
        for i in range(index+1, len(self._field_pos)):
            old_start, old_stop, fname = self._field_pos[i]
            self._field_pos[i] = old_start+shift, old_stop+shift, fname
        # Add new entry
        self._field_pos[index] = start, start+len(inserted_lines), name.upper()
    
    def __delitem__(self, index):
        index = self._translate_idx(index)
        start, stop, _ = self._field_pos[index]
        # Shift the start/stop indices of the following fields
        # by the amount of deleted fields
        shift = stop - start
        for i in range(index, len(self._field_pos)):
            old_start, old_stop, name = self._field_pos[i]
            self._field_pos[i] = old_start-shift, old_stop-shift, name
        del self.lines[start : stop]
        del self._field_pos[index]
    
    def __len__(self):
        return len(self._field_pos)

    def insert(self, index, name, content, subfields=None):
        index = self._translate_idx(index, length_exclusive=False)
        inserted_lines = self._to_lines(name, content, subfields)
        
        # Stop of previous field is start of new field
        if index == 0:
            start = 0
        else:
            _, start, _ = self._field_pos[index-1]
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
            self._field_pos[i] = old_start+shift, old_stop+shift, fname
        # Add new entry
        self._field_pos.insert(
            index,
            (start, start+len(inserted_lines), name.upper())
        )
    
    def append(self, name, content, subfields=None):
        self.insert(len(self), name, content, subfields)

    
    def _find_field_indices(self):
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
                    if start != -1:
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

    def _get_field_content(self, start, stop, indent):
        if indent == 0:
            return self.lines[start : stop]
        else:
            return [line[12:] for line in self.lines[start : stop]]
    
    def _to_lines(self, name, content, subfields):
        """
        Convert the field name, field content und subfield dictionary
        into text lines
        """
        if subfields is None:
            subfields = {}
        
        name = name.strip().upper()
        if len(name) == 0:
            raise ValueError(f"Must give a non emtpy name")
        subfields = OrderedDict({
            subfield_name.upper().strip() : subfield_lines
            for subfield_name, subfield_lines in subfields.items()
        })
        
        # Create lines for new field
        if name == "FEATURES":
            # Header line plus all actual feature lines
            lines = copy.copy(content)
            lines.insert(
                0, "FEATURES" + " "*13 + "Location/Qualifiers"
            )
        elif name == "ORIGIN":
            # Header line plus all actual sequence lines
            lines = copy.copy(content)
            lines.insert(0, "ORIGIN")
        else:
            name_column = []
            content_column = []
            # Create a line for the field name and empty lines
            # for each additional line required by the content 
            name_column += [name] + [""] * (len(content)-1)
            content_column += content
            for subfield_name, subfield_lines in subfields.items():
                name_column += ["  " + subfield_name] \
                               + [""] * (len(subfield_lines)-1)
                content_column += subfield_lines
            lines = [f"{n_col:12}{c_col}" for n_col, c_col
                              in zip(name_column, content_column)]
        
        return lines

    
    def _translate_idx(self, index, length_exclusive=True):
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
    This class represents a file in GenBank or GenPept format,
    that contains multiple entries, for more than one UID.
    
    The information for each UID are appended to each other in such a
    file.
    Objects of this class can be iterated to obtain a `GenBankFile`
    or `GenPeptFile` for each entry in the file.
    
    Examples
    --------
    
    >>> import os.path
    >>> file_name = fetch_single_file(
    ...     ["1L2Y_A", "3O5R_A", "5UGO_A"],
    ...     os.path.join(path_to_directory, "multifile.gp"),
    ...     "protein", "gp"
    ... )
    >>> multi_file = MultiFile(file_type="gp")
    >>> multi_file.read(file_name)
    >>> for gp_file in multi_file:
    ...     print(gp_file.get_accession())
    1L2Y_A
    3O5R_A
    5UGO_A
    """

    def __init__(self):
        super().__init__()

    def __iter__(self):
        start_i = 0
        for i in range(len(self.lines)):
            line = self.lines[i]
            if line.strip() == "//":
                # Create file with lines corresponding to that file
                file_content = "\n".join(self.lines[start_i : i+1])
                file = GenBankFile()
                file.read(io.StringIO(file_content))
                # Reset file start index
                start_i = i
                yield file