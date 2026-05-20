# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.uniprot"
__author__ = "Maximilian Greil"
__all__ = ["fetch"]

import io
import os
from collections.abc import Iterable
from os.path import getsize, isdir, isfile, join
from typing import Literal, overload
import requests
from biotite.database.uniprot.check import assert_valid_response

_fetch_url = "https://rest.uniprot.org/"

_UniprotFormat = Literal["fasta", "gff", "txt", "xml", "rdf", "tab"]


def _get_database_name(id: str) -> str:
    """
    Get the correct UniProt database from the ID of the file to be downloaded.

    Parameters
    ----------
    id: str
        ID of the file to be downloaded.

    Returns
    -------
    name : str
        E-utility UniProt database name.
    """
    if id[:3] == "UPI":
        return "uniparc"
    elif id[:6] == "UniRef":
        return "uniref"
    return "uniprotkb"


@overload
def fetch(
    ids: str,
    format: _UniprotFormat,
    target_path: str,
    overwrite: bool = False,
    verbose: bool = False,
) -> str: ...
@overload
def fetch(
    ids: str,
    format: _UniprotFormat,
    target_path: None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> io.StringIO: ...
@overload
def fetch(
    ids: Iterable[str],
    format: _UniprotFormat,
    target_path: str,
    overwrite: bool = False,
    verbose: bool = False,
) -> list[str]: ...
@overload
def fetch(
    ids: Iterable[str],
    format: _UniprotFormat,
    target_path: None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> list[io.StringIO]: ...
def fetch(
    ids: str | Iterable[str],
    format: _UniprotFormat,
    target_path: str | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> str | io.StringIO | list[str] | list[io.StringIO]:
    """
    Download files from the UniProt in various formats.

    Available databases are UniProtKB, UniRef and UniParc.
    This function requires an internet connection.

    Parameters
    ----------
    ids : str or iterable object of str
        A single ID or a list of IDs of the file(s)
        to be downloaded.
    format : {'fasta', 'gff', 'txt', 'xml', 'rdf', 'tab'}
        The format of the files to be downloaded.
    target_path : str, optional
        The target directory of the downloaded files.
        By default, the file content is stored in a `StringIO` object.
    overwrite : bool, optional
        If true, existing files will be overwritten. Otherwise the
        respective file will only be downloaded if the file does not
        exist yet in the specified target directory.
    verbose : bool, optional
        If true, the function will output the download progress.

    Returns
    -------
    files : str or StringIO or list of (str or StringIO)
        The file path(s) to the downloaded files.
        If a single string (a single ID) was given in `ids`,
        a single string is returned. If a list (or other iterable
        object) was given, a list of strings is returned.
        If no `target_path` was given, the file contents are stored in
        `StringIO` objects.

    Examples
    --------

    >>> import os.path
    >>> file = fetch("P12345", "fasta", path_to_directory)
    >>> print(os.path.basename(file))
    P12345.fasta
    >>> files = fetch(["P12345", "Q8K9I1"], "fasta", path_to_directory)
    >>> print([os.path.basename(file) for file in files])
    ['P12345.fasta', 'Q8K9I1.fasta']
    """
    # If only a single ID is present,
    # put it into a single element list
    if isinstance(ids, str):
        id_list = [ids]
        single_element = True
    else:
        id_list = list(ids)
        single_element = False
    # Create the target folder, if not existing
    if target_path is not None and not isdir(target_path):
        os.makedirs(target_path)
    files = []
    session = requests.Session()
    for i, id in enumerate(id_list):
        db_name = _get_database_name(id)
        # Verbose output
        if verbose:
            print(f"Fetching file {i + 1:d} / {len(id_list):d} ({id})...", end="\r")
        # Fetch file from database
        if target_path is not None:
            file = join(target_path, id + "." + format)
        else:
            # 'file = None' -> store content in a file-like object
            file = None
        if file is None or not isfile(file) or getsize(file) == 0 or overwrite:
            if format in ["fasta", "gff", "txt", "xml", "rdf", "tab"]:
                r = session.get(_fetch_url + db_name + "/" + id + "." + format)
                content = r.text
                assert_valid_response(r)
            else:
                raise ValueError(f"Format '{format}' is not supported")
            if file is None:
                file = io.StringIO(content)
            else:
                with open(file, "w+") as f:
                    f.write(content)
        files.append(file)
    if verbose:
        print("\nDone")
    # If input was a single ID, return only a single path
    if single_element:
        return files[0]
    else:
        return files
