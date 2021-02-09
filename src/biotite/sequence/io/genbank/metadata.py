# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Functions for obtaining metadata fields of a GenBank file.
"""

__name__ = "biotite.sequence.io.genbank"
__author__ = "Patrick Kunzmann"
__all__ = ["get_locus", "get_definition", "get_accession", "get_version",
           "get_gi", "get_db_link", "get_source",
           "set_locus"]

from ....file import InvalidFileError
from .file import GenBankFile


def get_locus(gb_file):
    """
    Parse the *LOCUS* field of a GenBank or GenPept file.
    
    Parameters
    ----------
    gb_file : GenBankFile
        The GenBank file to read the *LOCUS* field from.

    Returns
    -------
    name : str
        The locus name.
    length : int
        Sequence length.
    mol_type : str, optional
        The molecule type.
        Usually one of ``'DNA'``, ``'RNA'``, ``'Protein'`` or ``''``.
    is_circular : bool, optional
        True, if the sequence is circular, false otherwise.
    division : str, optional
        The GenBank division to which the file belongs.
    date : str, optional
        The date of last modification.
    
    Examples
    --------
    
    >>> import os.path
    >>> file = GenBankFile.read(os.path.join(path_to_sequences, "ec_bl21.gb"))
    >>> name, length, mol_type, is_circular, division, date = get_locus(file)
    >>> print(name)
    CP001509
    >>> print(length)
    4558953
    >>> print(mol_type)
    DNA
    >>> print(is_circular)
    True
    >>> print(division)
    BCT
    >>> print(date)
    16-FEB-2017
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
    Parse the *DEFINITION* field of a GenBank or GenPept file.
    
    Parameters
    ----------
    gb_file : GenBankFile
        The GenBank file to read the *DEFINITION* field from.
    
    Returns
    -------
    definition : str
        Content of the *DEFINITION* field.
    
    Examples
    --------
    
    >>> import os.path
    >>> file = GenBankFile.read(os.path.join(path_to_sequences, "ec_bl21.gb"))
    >>> print(get_definition(file))
    Escherichia coli BL21(DE3), complete genome.
    """
    lines, _ = _expect_single_field(gb_file, "DEFINITION")
    return " ".join([line.strip() for line in lines])

def get_accession(gb_file):
    """
    Parse the *ACCESSION* field of a GenBank or GenPept file.
    
    Parameters
    ----------
    gb_file : GenBankFile
        The GenBank file to read the *ACCESSION* field from.
    
    Returns
    -------
    accession : str
        The accession ID of the file.
    
    Examples
    --------
    
    >>> import os.path
    >>> file = GenBankFile.read(os.path.join(path_to_sequences, "ec_bl21.gb"))
    >>> print(get_accession(file))
    CP001509
    """
    lines, _ = _expect_single_field(gb_file, "ACCESSION")
    # 'ACCESSION' field has only one line
    return lines[0]

def get_version(gb_file):
    """
    Parse the version from the *VERSION* field of a GenBank or GenPept
    file.
    
    Parameters
    ----------
    gb_file : GenBankFile
        The GenBank file to read the *VERSION* field from.
    
    Returns
    -------
    version : str
        Content of the *VERSION* field. Does not include GI.
    """
    lines, _ = _expect_single_field(gb_file, "VERSION")
    # 'VERSION' field has only one line
    return lines[0].split()[0]

def get_gi(gb_file):
    """
    Parse the GI from the *VERSION* field of a GenBank or GenPept
    file.
    
    Parameters
    ----------
    gb_file : GenBankFile
        The GenBank file to read the *VERSION* field from.
    
    Returns
    -------
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
    Parse the *DBLINK* field of a GenBank or GenPept file.
    
    Parameters
    ----------
    gb_file : GenBankFile
        The GenBank file to read the *DBLINK* field from.
    
    Returns
    -------
    link_dict : dict
        A dictionary storing the database links, with the database
        name as key, and the corresponding ID as value.
    
    Examples
    --------
    
    >>> import os.path
    >>> file = GenBankFile.read(os.path.join(path_to_sequences, "ec_bl21.gb"))
    >>> for key, val in get_db_link(file).items():
    ...     print(key, ":", val)
    BioProject : PRJNA20713
    BioSample : SAMN02603478
    """
    lines, _ = _expect_single_field(gb_file, "DBLINK")
    link_dict = {}
    for line in lines:
        key, value = line.split(":")
        link_dict[key.strip()] = value.strip()
    return link_dict


def get_source(gb_file):
    """
    Parse the *SOURCE* field of a GenBank or GenPept file.
    
    Parameters
    ----------
    gb_file : GenBankFile
        The GenBank file to read the *SOURCE* field from.
    
    Returns
    -------
    accession : str
        The name of the source organism.
    """
    lines, _ = _expect_single_field(gb_file, "SOURCE")
    # 'SOURCE' field has only one line
    return lines[0]


def _expect_single_field(gb_file, name):
    fields = gb_file.get_fields(name)
    if len(fields) == 0:
        raise InvalidFileError(f"File has no '{name}' field")
    if len(fields) > 1:
        raise InvalidFileError(f"File has multiple '{name}' fields")
    return fields[0]



def set_locus(gb_file, name, length, mol_type=None, is_circular=False,
              division=None, date=None):
    """
    Set the *LOCUS* field of a GenBank file.
    
    Parameters
    ----------
    gb_file : GenBankFile
        The GenBank file to be edited.
    name : str
        The locus name.
    length : int
        Sequence length.
    mol_type : str, optional
        The molecule type.
        Usually one of ``'DNA'``, ``'RNA'``, ``'Protein'`` or ``''``.
    is_circular : bool, optional
        True, if the sequence is circular, false otherwise.
    division : str, optional
        The GenBank division to which the file belongs.
    date : str, optional
        The date of last modification.
    """
    mol_type = "" if mol_type is None else mol_type
    restype_abbr = "aa" if mol_type in ["", "Protein"] else "bp"
    circularity = "circular" if is_circular else "linear"
    division = "" if division is None else division
    date = "" if date is None else date
    line = f"{name:18} {length:>9} {restype_abbr} {mol_type:^10} " \
           f"{circularity:8} {division:3} {date:11}"
    gb_file.set_field("LOCUS", [line])