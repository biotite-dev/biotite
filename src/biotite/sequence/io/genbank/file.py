# Copyright 2017-2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from ....file import TextFile, InvalidFileError
from ...annotation import Location, Feature, Annotation
from ...seqtypes import NucleotideSequence, ProteinSequence 
import textwrap
import copy
import re

__all__ = ["GenBankFile", "GenPeptFile"]


class GenBankFile(TextFile):
    
    def __init__(self):
        super().__init__()
        # Field start and stop indices in list of lines
        # and names of categories
        self._fields = []
    
    def read(self, file_name):
        super().read(file_name)
        start = -1
        stop = -1
        name = ""
        for i, line in enumerate(self._lines):
            # Check if line contains a new major field
            # (Header beginning from first column)
            if len(line) != 0 and line[0] != " ":
                stop = i
                if start != -1:
                    # Store previous field
                    self._fields.append((start, stop, name))
                start = i
                name = line[0:12].strip()
        # Store last field
        stop = i
        self._fields.append((start, stop, name))
    
    def write(self, file_name):
        """
        Not implemented yet.
        """
        raise NotImplementedError()
    
    def get_locus(self):
        locus_dict = {}
        starts, stops = self._get_field_indices("LOCUS")
        locus_info = self._lines[starts[0]].split()
        locus_dict["name"] = locus_info[1]
        locus_dict["length"] = locus_info[2]
        locus_dict["type"] = locus_info[4]
        if locus_info[5] in ["circular", "linear"]:
            locus_dict["type"] += " " + locus_info[4]
        locus_dict["division"] = locus_info[-2]
        locus_dict["date"] = locus_info[-1]
        return locus_dict
    
    def get_definition(self):
        starts, stops = self._get_field_indices("DEFINITION")
        return self._lines[starts[0]][12:].strip()
    
    def get_accession(self):
        starts, stops = self._get_field_indices("ACCESSION")
        return self._lines[starts[0]][12:].strip()
    
    def get_version(self):
        starts, stops = self._get_field_indices("VERSION")
        return self._lines[starts[0]][12:].split()[0]
    
    def get_gi(self):
        starts, stops = self._get_field_indices("VERSION")
        version_info = self._lines[starts[0]][12:].split()
        if len(version_info) < 2 or "GI" not in version_info[1]:
            raise InvalidFileError("File does not contain GI")
        # Truncate GI
        return version_info[1][3:]
    
    def get_db_link(self):
        starts, stops = self._get_field_indices("DBLINK")
        link_dict = {}
        for i in range(starts[0], stops[0]):
            line = self._lines[i]
            content = line[12:].split(":")
            link_dict[content[0].strip()] = content[1].strip()
        return link_dict
    
    def get_source(self):
        starts, stops = self._get_field_indices("SOURCE")
        return self._lines[starts[0]][12:].strip()
    
    def get_references(self):
        references = []
        starts, stops = self._get_field_indices("REFERENCE")
        for i in range(len(starts)):
            start = starts[i]
            stop = stops[i]
            ref_dict_raw = self._get_minor_fields(start, stop)
            ref_dict = {}
            for key, val in ref_dict_raw.items():
                if key == "REFERENCE":
                    loc_info = val[val.index("(")+1 : val.index(")")].split()
                    first_base = int(loc_info[loc_info.index("bases") +1])
                    last_base = int(loc_info[loc_info.index("to") +1])
                    ref_dict["location"] = (first_base, last_base)
                elif key == "AUTHORS":
                    ref_dict["authors"] = val
                elif key == "TITLE":
                    ref_dict["title"] = val
                elif key == "JOURNAL":
                    ref_dict["journal"] = val
                elif key == "PUBMED":
                    ref_dict["pubmed"] = val
                elif key == "REMARK":
                    ref_dict["remark"] = val
            references.append(ref_dict)
        return references
    
    def get_comment(self):
        starts, stops = self._get_field_indices("COMMENT")
        return self._lines[starts[0]][12:].strip()
    
    def get_annotation(self, include_only=None):
        starts, stops = self._get_field_indices("FEATURES")
        # Remove the first line,
        # because it contains the "FEATURES" string itself.
        start = starts[0] +1
        stop = stops[0]
        
        feature_list = []
        feature_key = None
        feature_value = ""
        for i in range(start, stop):
            line = self._lines[i]
            # Check if line contains feature key
            if line[5] != " ":
                if feature_key is not None:
                    # Store old feature key and value
                    feature_list.append((feature_key, feature_value))
                # Track new key
                feature_key = line[5:20].strip()
                feature_value = ""
            feature_value += line[21:] + " "
        # Store last feature key and value (loop already exited)
        feature_list.append((feature_key, feature_value))
        
        # Process only relevant features and put them into Annotation
        annotation = Annotation()
        for key, val in feature_list:
            if include_only is None or key in include_only:
                qualifiers = val.split(" /")
                # First string in qualifier is location identifier
                # since it comes before the first qualifier
                locs = _parse_locs(qualifiers.pop(0).strip())
                qual_dict = {}
                for qual in qualifiers:
                    qual_pair = qual.replace('"','').strip().split("=")
                    if len(qual_pair) == 2:
                        qual_dict[qual_pair[0]] = qual_pair[1]
                    else:
                        # In case of e.g. '/pseudo'
                        qual_dict[qual_pair[0]] = None
                annotation.add_feature(Feature(key, locs, qual_dict))
        return annotation
    
    def get_sequence(self):
        return NucleotideSequence(self._get_seq_string())
    
    def _get_seq_string(self):
        starts, stops = self._get_field_indices("ORIGIN")
        seq_str = ""
        rx = re.compile("[0-9]| ")
        for i in range(starts[0]+1, stops[0]):
            print(self._lines[i])
            seq_str += rx.sub("", self._lines[i])
        return seq_str
            

    def _get_field_indices(self, name):
        starts = []
        stops = []
        for field in self._fields:
            if field[2] == name:
                starts.append(field[0])
                stops.append(field[1])
        if len(starts) == 0:
            raise InvalidFileError("File does not contain '{:}' category"
                                   .format(name))
        return starts, stops
    
    def _get_minor_fields(self, field_start, field_stop):
        header = ""
        content = ""
        minor_field_dict = {}
        for i in range(field_start, field_stop):
            line = self._lines[i]
            # Check if line contains a new minor field
            # (Header beginning from first column)
            if len(line) != 0 and line[:12].strip() != "":
                # Store previous minor field
                if content != "":
                    minor_field_dict[header] = content
                header = line[:12].strip()
                content = line[12:].strip()
            else:
                content += " " + line[12:].strip()
        # Store last minor field
        minor_field_dict[header] = content
        return minor_field_dict


def _parse_locs(loc_str):
    locs = []
    if loc_str.startswith(("join", "order")):
        str_list = loc_str[loc_str.index("(")+1:loc_str.rindex(")")].split(",")
        for s in str_list:
            locs.extend(_parse_locs(s))
    elif loc_str.startswith("complement"):
        compl_str = loc_str[loc_str.index("(")+1:loc_str.rindex(")")]
        compl_locs = _parse_locs(compl_str)
        for loc in compl_locs:
            loc.strand = Location.Strand.REVERSE
        locs.extend(compl_locs)
    else:
        locs = [_parse_single_loc(loc_str)]
    return locs


def _parse_single_loc(loc_str):
    if ".." in loc_str:
        split_char = ".."
        defect = Location.Defect(0)
    elif "." in loc_str:
        split_char = "."
        defect = Location.Defect.UNK_LOC
    elif "^" in loc_str:
        split_char = "^"
        loc_str_split = loc_str.split("..")
        defect = Location.Defect.BETWEEN
    else:
        first_and_last = int(loc_str)
        return Location(first_and_last, first_and_last)
    loc_str_split = loc_str.split(split_char)
    first_str = loc_str_split[0]
    last_str = loc_str_split[1]
    if first_str[0] == "<":
        first = int(first_str[1:])
        defect |= Location.Defect.BEYOND_LEFT
    else:
        first = int(first_str)
    if last_str[0] == ">":
        last = int(last_str[1:])
        defect |= Location.Defect.BEYOND_RIGHT
    else:
        last = int(last_str)
    return Location(first, last, defect=defect)



class GenPeptFile(GenBankFile):
    
    def get_locus(self):
        locus_dict = {}
        starts, stops = self._get_field_indices("LOCUS")
        locus_info = self._lines[starts[0]].split()
        locus_dict["name"] = locus_info[1]
        locus_dict["length"] = locus_info[2]
        locus_dict["division"] = locus_info[-2]
        locus_dict["date"] = locus_info[-1]
        return locus_dict
    
    def get_sequence(self):
        return ProteinSequence(self._get_seq_string())
