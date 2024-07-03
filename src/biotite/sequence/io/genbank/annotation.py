# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Functions for converting an annotation from/to a GenBank file.
"""

__name__ = "biotite.sequence.io.genbank"
__author__ = "Patrick Kunzmann"
__all__ = ["get_annotation", "set_annotation"]

import re
import warnings
from biotite.file import InvalidFileError
from biotite.sequence.annotation import Annotation, Feature, Location

_KEY_START = 5
_QUAL_START = 21


def get_annotation(gb_file, include_only=None):
    """
    Get the sequence annotation from the *FEATURES* field of a
    GenBank file.

    Parameters
    ----------
    gb_file : GenBankFile
        The GenBank file to read the *FEATURES* field from.
    include_only : iterable object of str, optional
        List of names of feature keys, which should included
        in the annotation. By default all features are included.

    Returns
    -------
    annotation : Annotation
        Sequence annotation from the file.
    """
    fields = gb_file.get_fields("FEATURES")
    if len(fields) == 0:
        raise InvalidFileError("File has no 'FEATURES' field")
    if len(fields) > 1:
        raise InvalidFileError("File has multiple 'FEATURES' fields")
    lines, _ = fields[0]

    ### Parse all lines to create an index of features,
    # i.e. pairs of the feature key
    # and the text belonging to the respective feature
    feature_list = []
    feature_key = None
    feature_value = ""
    for line in lines:
        # Check if line contains feature key
        if line[_KEY_START] != " ":
            if feature_key is not None:
                # Store old feature key and value
                feature_list.append((feature_key, feature_value))
            # Track new key
            feature_key = line[_KEY_START : _QUAL_START - 1].strip()
            feature_value = ""
        feature_value += line[_QUAL_START:] + " "
    # Store last feature key and value (loop already exited)
    feature_list.append((feature_key, feature_value))

    ### Process only relevant features and put them into an Annotation
    annotation = Annotation()
    # Regex to separate qualifiers from each other
    regex = re.compile(r"""(".*?"|/.*?=)""")
    for key, val in feature_list:
        if include_only is None or key in include_only:
            qual_dict = {}

            # Split feature definition into parts
            # e.g.
            #
            # 1..12
            # /gene="abcA"
            # /product="AbcA"
            #
            # becomes
            #
            # ['1..12', '/gene=', '"abcA"', '/product=', '"AbcA"']
            qualifier_parts = [s.strip() for s in regex.split(val)]
            # Remove empty qualifier parts
            qualifier_parts = [s for s in qualifier_parts if s]
            # First part is location identifier
            loc_string = qualifier_parts.pop(0).strip()
            try:
                locs = _parse_locs(loc_string)
            except Exception:
                warnings.warn(
                    f"'{loc_string}' is an unsupported location identifier, "
                    f"skipping feature"
                )
                continue

            # The other parts are pairwise qualifier keys and values
            qual_key = None
            qual_val = None
            for part in qualifier_parts:
                if qual_key is None:
                    # This is a qualifier key
                    # When the feature contains qualifiers without
                    # value, e.g. '/pseudo'
                    # The part may contain multiple keys, e.g.
                    #
                    # '/pseudo /gene='
                    #
                    # -> split at whitespaces,
                    # as keys do not contain whitespaces
                    for subpart in part.split():
                        if "=" not in subpart:
                            # Qualifier without value, e.g. '/pseudo'
                            # -> store immediately
                            # Remove "/" -> subpart[1:]
                            qual_key = subpart[1:]
                            _set_qual(qual_dict, qual_key, None)
                            qual_key = None
                        else:
                            # Regular qualifier
                            # -> store key in variable and wait for
                            # next qualifier part to set the value
                            # Remove '/' and '=' -> subpart[1:-1]
                            qual_key = subpart[1:-1]
                else:
                    # This is a qualifier value
                    # -> remove potential quotes
                    if part[0] == '"':
                        qual_val = part[1:-1]
                    else:
                        qual_val = part
                    # Store qualifier pair
                    _set_qual(qual_dict, qual_key, qual_val)
                    qual_key = None
                    qual_val = None

            annotation.add_feature(Feature(key, locs, qual_dict))

    return annotation


def _parse_locs(loc_str):
    locs = []
    if loc_str.startswith(("join", "order")):
        str_list = loc_str[loc_str.index("(") + 1 : loc_str.rindex(")")].split(",")
        for s in str_list:
            locs.extend(_parse_locs(s.strip()))
    elif loc_str.startswith("complement"):
        compl_str = loc_str[loc_str.index("(") + 1 : loc_str.rindex(")")]
        compl_locs = [
            Location(loc.first, loc.last, Location.Strand.REVERSE, loc.defect)
            for loc in _parse_locs(compl_str)
        ]
        locs.extend(compl_locs)
    else:
        locs = [_parse_single_loc(loc_str)]
    return locs


def _parse_single_loc(loc_str):
    if ".." in loc_str:
        split_char = ".."
        defect = Location.Defect.NONE
    elif "." in loc_str:
        split_char = "."
        defect = Location.Defect.UNK_LOC
    elif "^" in loc_str:
        split_char = "^"
        loc_str_split = loc_str.split("..")
        defect = Location.Defect.BETWEEN
    else:
        # Parse single location
        defect = Location.Defect.NONE
        if loc_str[0] == "<":
            loc_str = loc_str[1:]
            defect |= Location.Defect.BEYOND_LEFT
        elif loc_str[0] == ">":
            loc_str = loc_str[1:]
            defect |= Location.Defect.BEYOND_RIGHT
        first_and_last = int(loc_str)
        return Location(first_and_last, first_and_last, defect=defect)
    # Parse location range
    loc_str_split = loc_str.split(split_char)
    first_str = loc_str_split[0]
    last_str = loc_str_split[1]
    # Parse Defects
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


def _set_qual(qual_dict, key, val):
    """
    Set a mapping key to val in the dictionary.
    If the key already exists in the dictionary, append the value (str)
    to the existing value, separated by a line break.
    """
    if key in qual_dict:
        qual_dict[key] += "\n" + val
    else:
        qual_dict[key] = val


def set_annotation(gb_file, annotation):
    """
    Set the *FEATURES* field of a GenBank file with an annotation.

    Parameters
    ----------
    gb_file : GenBankFile
        The GenBank file to be edited.
    annotation : Annotation
        The annotation that is put into the GenBank file.
    """
    lines = []
    for feature in sorted(annotation):
        line = " " * _KEY_START
        line += feature.key.ljust(_QUAL_START - _KEY_START)
        line += _convert_to_loc_string(feature.locs)
        lines.append(line)
        for key, values in feature.qual.items():
            if values is None:
                line = " " * _QUAL_START
                line += f"/{key}"
                lines.append(line)
            else:
                for val in values.split("\n"):
                    line = " " * _QUAL_START
                    line += f'/{key}="{val}"'
                    lines.append(line)
    gb_file.set_field("FEATURES", lines)


def _convert_to_loc_string(locs):
    """
    Create GenBank comptabile location strings from a list of :class:`Location`
    objects.
    """
    if len(locs) == 1:
        loc = list(locs)[0]
        loc_first_str = str(loc.first)
        loc_last_str = str(loc.last)
        if loc.defect & Location.Defect.BEYOND_LEFT:
            loc_first_str = "<" + loc_first_str
        if loc.defect & Location.Defect.BEYOND_RIGHT:
            loc_last_str = ">" + loc_last_str
        if loc.first == loc.last:
            loc_string = loc_first_str
        elif loc.defect & Location.Defect.UNK_LOC:
            loc_string = loc_first_str + "." + loc_last_str
        elif loc.defect & Location.Defect.BETWEEN:
            loc_string = loc_first_str + "^" + loc_last_str
        else:
            loc_string = loc_first_str + ".." + loc_last_str
        if loc.strand == Location.Strand.REVERSE:
            loc_string = f"complement({loc_string})"
    else:
        loc_string = ",".join([_convert_to_loc_string([loc]) for loc in locs])
        loc_string = f"join({loc_string})"
    return loc_string
