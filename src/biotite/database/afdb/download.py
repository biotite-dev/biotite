# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.afdb"
__author__ = "Patrick Kunzmann, Alex Carlin"
__all__ = ["fetch"]

import io
import re
from pathlib import Path
from xml.etree import ElementTree
import requests
from biotite.database.error import RequestError

_METADATA_URL = "https://alphafold.com/api/prediction"
_BINARY_FORMATS = ["bcif"]
# Adopted from https://www.uniprot.org/help/accession_numbers
_UNIPROT_PATTERN = (
    "[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
)


def fetch(ids, format, target_path=None, overwrite=False, verbose=False):
    """
    Download predicted protein structures from the AlphaFold DB.

    This function requires an internet connection.

    Parameters
    ----------
    ids : str or iterable object of str
        A single ID or a list of IDs of the file(s) to be downloaded.
        They can be either UniProt IDs (e.g. ``P12345``) or AlphaFold DB IDs
        (e.g. ``AF-P12345F1``).
    format : {'pdb', 'pdbx', 'cif', 'mmcif', 'bcif', 'fasta'}
        The format of the files to be downloaded.
    target_path : str, optional
        The target directory of the downloaded files.
        By default, the file content is stored in a file-like object
        (`StringIO` or `BytesIO`, respectively).
    overwrite : bool, optional
        If true, existing files will be overwritten.
        Otherwise the respective file will only be downloaded if the file does not
        exist yet in the specified target directory or if the file is empty.
    verbose : bool, optional
        If true, the function will output the download progress.

    Returns
    -------
    files : str or StringIO or BytesIO or list of (str or StringIO or BytesIO)
        The file path(s) to the downloaded files.
        If a single string (a single ID) was given in `ids`, a single string is
        returned.
        If a list (or other iterable object) was given, a list of strings is returned.
        If no `target_path` was given, the file contents are stored in either
        ``StringIO`` or ``BytesIO`` objects.

    Examples
    --------

    >>> from pathlib import Path
    >>> file = fetch("P12345", "cif", path_to_directory)
    >>> print(Path(file).name)
    P12345.cif
    >>> files = fetch(["P12345", "Q8K9I1"], "cif", path_to_directory)
    >>> print([Path(file).name for file in files])
    ['P12345.cif', 'Q8K9I1.cif']
    """
    if format not in ["pdb", "pdbx", "cif", "mmcif", "bcif", "fasta"]:
        raise ValueError(f"Format '{format}' is not supported")
    if format in ["pdbx", "mmcif"]:
        format = "cif"

    # If only a single ID is present,
    # put it into a single element list
    if isinstance(ids, str):
        ids = [ids]
        single_element = True
    else:
        single_element = False
    if target_path is not None:
        target_path = Path(target_path)
        target_path.mkdir(parents=True, exist_ok=True)

    files = []
    for i, id in enumerate(ids):
        # Verbose output
        if verbose:
            print(f"Fetching file {i + 1:d} / {len(ids):d} ({id})...", end="\r")
        # Fetch file from database
        if target_path is not None:
            file = target_path / f"{id}.{format}"
        else:
            # 'file = None' -> store content in a file-like object
            file = None
        if file is None or not file.is_file() or file.stat().st_size == 0 or overwrite:
            file_response = requests.get(_get_file_url(id, format))
            _assert_valid_file(file_response, id)
            if format in _BINARY_FORMATS:
                content = file_response.content
            else:
                content = file_response.text

            if file is None:
                if format in _BINARY_FORMATS:
                    file = io.BytesIO(content)
                else:
                    file = io.StringIO(content)
            else:
                mode = "wb+" if format in _BINARY_FORMATS else "w+"
                with open(file, mode) as f:
                    f.write(content)

        files.append(file)
    if verbose:
        print("\nDone")

    # Return paths as strings
    files = [file.as_posix() if isinstance(file, Path) else file for file in files]
    # If input was a single ID, return only a single element
    if single_element:
        return files[0]
    else:
        return files


def _get_file_url(id, format):
    """
    Get the actual file URL for the given ID from the ``prediction`` API endpoint.

    Parameters
    ----------
    id : str
        The ID of the file to be downloaded.
    format : str
        The format of the file to be downloaded.

    Returns
    -------
    file_url : str
        The URL of the file to be downloaded.
    """
    uniprot_id = _extract_id(id)
    metadata = requests.get(f"{_METADATA_URL}/{uniprot_id}").json()
    if len(metadata) == 0:
        raise RequestError(f"ID {id} is invalid")
    # A list of length 1 is always returned, if the response is valid
    return metadata[0][f"{format}Url"]


def _extract_id(id):
    """
    Extract a AFDB compatible UniProt ID from the given qualifier.
    This may comprise

    - Directly the UniProt ID (e.g. ``P12345``) (trivial case)
    - Entry ID, as also returned by the RCSB search API (e.g. ``AF-P12345-F1``)

    Parameters
    ----------
    id : str
        The qualifier to extract the UniProt ID from.

    Returns
    -------
    uniprot_id : str
        The UniProt ID.
    """
    match = re.search(_UNIPROT_PATTERN, id)
    if match is None:
        raise ValueError(f"Cannot extract AFDB identifier from '{id}'")
    return match.group()


def _assert_valid_file(response, id):
    """
    Checks whether the response is an actual structure file
    or the response a *404* error due to invalid UniProt ID.
    """
    if len(response.text) == 0:
        raise RequestError(f"Received no repsone for '{id}'")
    try:
        root = ElementTree.fromstring(response.text)
        if root.tag == "Error":
            raise RequestError(
                f"Error while fetching '{id}': {root.find('Message').text}"
            )
    except ElementTree.ParseError:
        # This is not XML -> the response is probably a valid file
        pass
