# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Functions for obtaining metadata fields of a GenBank file.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["get_locus", "get_definition", "get_accession", "get_version",
           "get_gi", "get_db_link",
           "set_locus"]

from ....file import InvalidFileError
from .file import GenBankFile


def get_locus(gb_file):
    """
    Parse the *LOCUS* field of the file.
    
    Returns
    ----------
    locus_dict : dict
        A dictionary storing the locus *name*, *length*, *type*,
        *division* and *date*.
    
    Examples
    --------
    
    >>> import os.path
    >>> file = GenBankFile()
    >>> file.read(os.path.join(path_to_sequences, "ec_bl21.gb"))
    >>> for key, val in file.get_locus().items():
    ...     print(key, ":", val)
    name : CP001509
    length : 4558953
    type : DNA circular
    division : BCT
    date : 16-FEB-2017
    """
    lines, _ = _expect_single_field(gb_file, "LOCUS")
    # 'LOCUS' field has only one line
    locus_info = lines[0]

    name = locus_info[:18].strip()
    length = int(locus_info[18 : 28])
    mol_type = locus_info[32 : 42].strip()
    is_circular = True if locus_info[43 : 51] == "circular" else False
    division = locus_info[52 : 55].strip()
    date = locus_info[56 : 67]
    return name, length, mol_type, is_circular, division, date

def get_definition(gb_file):
    """
    Parse the *DEFINITION* field of the file.
    
    Returns
    ----------
    definition : str
        Content of the *DEFINITION* field.
    """
    lines, _ = _expect_single_field(gb_file, "DEFINITION")
    return " ".join([line.strip() for line in lines])

def get_accession(gb_file):
    """
    Parse the *ACCESSION* field of the file.
    
    Returns
    ----------
    accession : str
        Content of the *ACCESSION* field.
    """
    lines, _ = _expect_single_field(gb_file, "ACCESSION")
    # 'ACCESSION' field has only one line
    return lines[0]

def get_version(gb_file):
    """
    Parse the *VERSION* field of the file.
    
    Returns
    ----------
    version : str
        Content of the *VERSION* field. Does not include GI.
    """
    lines, _ = _expect_single_field(gb_file, "VERSION")
    # 'VERSION' field has only one line
    return lines[0].split()[0]

def get_gi(gb_file):
    """
    Get the GI of the file.
    
    Returns
    ----------
    gi : str
        The GI of the file.
    """
    lines, _ = _expect_single_field(gb_file, "VERSION")
    # 'VERSION' field has only one line
    version_info = lines[0].split()
    if len(version_info) < 2 or "GI" not in version_info[1]:
        raise InvalidFileError("File does not contain GI")
    # Truncate GI
    return int(version_info[1][3:])

def get_db_link(gb_file):
    """
    Parse the *DBLINK* field of the file.
    
    Returns
    ----------
    link_dict : dict
        A dictionary storing the database links, with the database
        name as key, and the corresponding ID as value.
    
    Examples
    --------
    
    >>> import os.path
    >>> file = GenBankFile()
    >>> file.read(os.path.join(path_to_sequences, "ec_bl21.gb"))
    >>> for key, val in file.get_db_link().items():
    ...     print(key, ":", val)
    BioProject : PRJNA20713
    BioSample : SAMN02603478
    """
    lines, _ = _expect_single_field(gb_file, "DBLINK")
    link_dict = {}
    for line in lines:
        key, value = line[12:].split(":")
        link_dict[key.strip()] = value.strip()
    return link_dict


def _expect_single_field(gb_file, name):
    fields = gb_file.get_fields(name)
    if len(fields) == 0:
        raise InvalidFileError(f"File has no '{name}' field")
    if len(fields) > 1:
        raise InvalidFileError(f"File has multiple '{name}' fields")
    return fields[0]




def set_locus(gb_file, name, length, mol_type=None, is_circular=False,
              division=None, date=None):
    mol_type = "" if mol_type is None else mol_type
    restype_abbr = "aa" if mol_type in ["", "Protein"] else "bp"
    circularity = "circular" if is_circular else "linear"
    division = "" if division is None else division
    date = "" if date is None else date
    line = f"{name:18} {length:>9} {restype_abbr} {mol_type:^10} " \
           f"{circularity:8} {division:3} {date:11}"
    gb_file.set_field("LOCUS", [line])