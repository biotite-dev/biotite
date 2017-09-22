# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from ....file import TextFile
from ...sequence import Sequence
from ...annotation import Location
from ...alphabet import AlphabetError
from ...seqtypes import NucleotideSequence, ProteinSequence
import textwrap
import copy

__all__ = ["GenBankFile"]


class GenBankFile(TextFile):
    
    def __init__(self):
        super().__init__()
        self._category_i = []
    
    def read(self, file_name):
        super().read(file_name)
        for i, line in enumerate(self._lines):
            #Check if line contains letters in its first 5 characters
            if len(line[0:5].strip()) > 0:
                category = line[0:12].strip()
                self._category_i.append((i, category))
    
    def get_feature_table(self, include_only=None):
        feature_lines = self.get_category("FEATURES", as_list=True)[0]
        # Remove the first line,
        # because it conatins the "FEATURES" string itself.
        feature_lines.pop(0)
        
        feature_list = []
        feature_key = None
        feature_value = ""
        for line in feature_lines:
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
        
        # Process only relevant features
        feature_table = []
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
                feature_table.append((key, locs, qual_dict))
        return feature_table
    
    def get_category(self, category_str, as_list=False):
        category_str = category_str.upper()
        category_list = []
        for i in range(len(self._category_i)):
            if self._category_i[i][1] == category_str:
                start = self._category_i[i][0]
                stop = self._category_i[i+1][0]
                category_list.append(self._lines[start:stop])
        if as_list:
            return category_list
        else:
            category_str_list = []
            for cat in category_list:
                cat_str = ""
                for line in cat:
                    cat_str += line + "\n"
                category_str_list.append(cat_str)
            return(category_str_list)


def _parse_locs(loc_str):
    locs = []
    if loc_str.startswith(("join", "order")):
        str_list = loc_str[loc_str.index("(")+1:loc_str.index(")")].split(",")
        for s in str_list:
            locs.extend(_parse_locs(s))
    elif loc_str.startswith("complement"):
        compl_str = loc_str[loc_str.index("(")+1:loc_str.index(")")]
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
        last = int(first_str[1:])
        defect |= Location.Defect.BEYOND_RIGHT
    else:
        last = int(first_str)
    return Location(first, last, defect=defect)
