# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.rcsb"
__author__ = "Patrick Kunzmann"
__all__ = ["fetch"]

import io
import os
from os.path import getsize, isfile, join
import requests
from biotite.database.error import RequestError

_standard_url = "https://files.rcsb.org/download/"
_bcif_url = "https://models.rcsb.org/"
_fasta_url = "https://www.rcsb.org/fasta/entry/"

_binary_formats = ["bcif"]
_rcsb_error_msgs = [
    "404 Not Found",
    "<title>RCSB Protein Data Bank Error Page</title>",
    "<title>PDB Archive over AWS</title>",
    "No fasta files were found.",
    "No valid PDB IDs were submitted.",
    "The requested URL was incorrect, too long or otherwise malformed.",
]


def fetch(
    pdb_ids, format, target_path=None, overwrite=False, verbose=False, gzip=False
):
    """
    Download structure files (or sequence files) from the RCSB PDB in
    various formats.

    This function requires an internet connection.

    Parameters
    ----------
    pdb_ids : str or iterable object of str
        A single PDB ID or a list of PDB IDs of the structure(s)
        to be downloaded.
    format : {'pdb', 'pdbx', 'cif', 'mmcif', 'bcif', 'fasta'}
        The format of the files to be downloaded.
        ``'pdbx'``, ``'cif'`` and ``'mmcif'`` are synonyms for
        the same format.
    target_path : str, optional
        The target directory of the downloaded files.
        By default, the file content is stored in a file-like object
        (:class:`StringIO` or :class:`BytesIO`, respectively).
    overwrite : bool, optional
        If true, existing files will be overwritten.
        Otherwise the respective file will only be downloaded, if the
        file does not exist yet in the specified target directory or if
        the file is empty.
    verbose : bool, optional
        If set to true, the function will output the download progress.
    gzip : bool, optional
        If set to true, the file will be downloaded in gzipped format.
        If `format` is not ``None``, the written files get the additional ``.gz``
        extension.
        Not supported for ``"fasta"`` format.

    Returns
    -------
    files : str or StringIO or BytesIO or list of (str or StringIO or BytesIO)
        The file path(s) to the downloaded files.
        If a single PDB ID was given in `pdb_ids`,
        a single string is returned. If a list (or other iterable
        object) was given, a list of strings is returned.
        If no `target_path` was given, the file contents are stored in
        either :class:`StringIO` or :class:`BytesIO` objects.

    Warnings
    --------
    Even if you give valid input to this function, in rare cases the
    database might return no or malformed data to you.
    In these cases the request should be retried.
    When the issue occurs repeatedly, the error is probably in your
    input.

    Examples
    --------

    >>> import os.path
    >>> file = fetch("1l2y", "cif", path_to_directory)
    >>> print(os.path.basename(file))
    1l2y.cif
    >>> files = fetch(["1l2y", "3o5r"], "cif", path_to_directory)
    >>> print([os.path.basename(file) for file in files])
    ['1l2y.cif', '3o5r.cif']
    """
    # If only a single PDB ID is present,
    # put it into a single element list
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]
        single_element = True
    else:
        single_element = False
    # Create the target folder, if not existing
    if target_path is not None and not os.path.isdir(target_path):
        os.makedirs(target_path)

    if gzip:
        gz_suffix = ".gz"
        if format == "fasta":
            raise ValueError("Gzip is not supported for 'fasta' format")
    else:
        gz_suffix = ""

    files = []
    session = requests.Session()
    for i, id in enumerate(pdb_ids):
        # Verbose output
        if verbose:
            print(f"Fetching file {i + 1:d} / {len(pdb_ids):d} ({id})...", end="\r")

        # Fetch file from database
        if target_path is not None:
            file = join(target_path, id + "." + format + gz_suffix)
        else:
            # 'file = None' -> store content in a file-like object
            file = None

        if file is None or not isfile(file) or getsize(file) == 0 or overwrite:
            if format == "pdb":
                r = session.get(_standard_url + id + ".pdb" + gz_suffix)
                _assert_valid_file(r, id)
                if gzip:
                    content = r.content
                else:
                    content = r.text
            elif format in ["cif", "mmcif", "pdbx"]:
                r = session.get(_standard_url + id + ".cif" + gz_suffix)
                _assert_valid_file(r, id)
                if gzip:
                    content = r.content
                else:
                    content = r.text
            elif format in ["bcif"]:
                r = session.get(_bcif_url + id + ".bcif" + gz_suffix)
                _assert_valid_file(r, id)
                content = r.content
            elif format == "fasta":
                r = session.get(_fasta_url + id)
                _assert_valid_file(r, id)
                content = r.text
            else:
                raise ValueError(f"Format '{format}' is not supported")

            if file is None:
                if format in _binary_formats or gzip:
                    file = io.BytesIO(content)
                else:
                    file = io.StringIO(content)
            else:
                mode = "wb+" if format in _binary_formats or gzip else "w+"
                with open(file, mode) as f:
                    f.write(content)

        files.append(file)
    if verbose:
        print("\nDone")
    # If input was a single ID, return only a single path
    if single_element:
        return files[0]
    else:
        return files


def _assert_valid_file(response, pdb_id):
    """
    Checks whether the response is an actual structure file
    or the response a *404* error due to invalid PDB ID.
    """
    if response.status_code == 404:
        raise RequestError(f"PDB ID {pdb_id} is invalid")
    # Fallback for other errors
    try:
        response.raise_for_status()
    except requests.HTTPError:
        raise RequestError(f"PDB ID {pdb_id} is invalid")

    content_type = response.headers.get("Content-Type", "")
    # Structure file and FASTA file retrieval
    # have different error messages
    if "text" in content_type or "html" in content_type:
        text = response.text
        if len(text) == 0 or any(err_msg in text for err_msg in _rcsb_error_msgs):
            raise RequestError(f"PDB ID {pdb_id} is invalid")
