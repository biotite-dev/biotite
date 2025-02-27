# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.entrez"
__author__ = "Patrick Kunzmann"
__all__ = ["fetch", "fetch_single_file"]

import io
import os
from os.path import getsize, isdir, isfile, join
import requests
from biotite.database.entrez.check import check_for_errors
from biotite.database.entrez.dbnames import sanitize_database_name
from biotite.database.entrez.key import get_api_key
from biotite.database.error import RequestError

_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch(
    uids,
    target_path,
    suffix,
    db_name,
    ret_type,
    ret_mode="text",
    overwrite=False,
    verbose=False,
):
    """
    Download files from the NCBI Entrez database in various formats.

    The data for each UID will be fetched into a separate file.

    A list of valid database, retrieval type and mode combinations can
    be found under
    `<https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly>`_

    This function requires an internet connection.

    Parameters
    ----------
    uids : str or iterable object of str
        A single *unique identifier* (UID) or a list of UIDs of the
        file(s) to be downloaded.
    target_path : str or None
        The target directory of the downloaded files.
        If ``None``, the file content is stored in a file-like object
        (`StringIO` or `BytesIO`, respectively).
    suffix : str
        The file suffix of the downloaded files. This value is
        independent of the retrieval type.
    db_name : str:
        E-utility or common database name.
    ret_type : str
        Retrieval type.
    ret_mode : str, optional
        Retrieval mode.
    overwrite : bool, optional
        If true, existing files will be overwritten. Otherwise the
        respective file will only be downloaded if the file does not
        exist yet in the specified target directory or if the file is
        empty.
    verbose : bool, optional
        If true, the function will output the download progress.

    Returns
    -------
    files : str or StringIO or BytesIO or list of (str or StringIO or BytesIO)
        The file path(s) to the downloaded files.
        If a single string (a single UID) was given in `uids`,
        a single string is returned. If a list (or other iterable
        object) was given, a list of strings is returned.
        If `target_path` is ``None``, the file contents are stored in
        either `StringIO` or `BytesIO` objects.

    Warnings
    --------
    Even if you give valid input to this function, in rare cases the
    database might return no or malformed data to you.
    In these cases the request should be retried.
    When the issue occurs repeatedly, the error is probably in your
    input.

    See Also
    --------
    fetch_single_file : Fetch multiple entries as a single file.

    Examples
    --------

    >>> import os.path
    >>> files = fetch(["1L2Y_A","3O5R_A"], path_to_directory, suffix="fa",
    ...               db_name="protein", ret_type="fasta")
    >>> print([os.path.basename(file) for file in files])
    ['1L2Y_A.fa', '3O5R_A.fa']
    """
    # If only a single UID is present,
    # put it into a single element list
    if isinstance(uids, str):
        uids = [uids]
        single_element = True
    else:
        single_element = False
    # Create the target folder, if not existing
    if target_path is not None and not isdir(target_path):
        os.makedirs(target_path)
    files = []
    for i, id in enumerate(uids):
        # Verbose output
        if verbose:
            print(f"Fetching file {i + 1:d} / {len(uids):d} ({id})...", end="\r")
        # Fetch file from database
        if target_path is not None:
            file = join(target_path, id + "." + suffix)
        else:
            file = None
        if file is None or not isfile(file) or getsize(file) == 0 or overwrite:
            param_dict = {
                "db": sanitize_database_name(db_name),
                "id": id,
                "rettype": ret_type,
                "retmode": ret_mode,
                "tool": "Biotite",
                "mail": "padix.key@gmail.com",
            }
            api_key = get_api_key()
            if api_key is not None:
                param_dict["api_key"] = api_key
            r = requests.get(_fetch_url, params=param_dict)
            content = r.text
            check_for_errors(content)
            if content.startswith(" Error"):
                raise RequestError(content[8:])
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


def fetch_single_file(
    uids, file_name, db_name, ret_type, ret_mode="text", overwrite=False
):
    """
    Almost the same as :func:`fetch()`, but the data for the given UIDs
    will be stored in a single file.

    Parameters
    ----------
    uids : iterable object of str
        A list of UIDs of the
        file(s) to be downloaded.
    file_name : str or None
        The file path, including file name, to the target file.
    db_name : str:
        E-utility or common database name.
    ret_type : str
        Retrieval type.
    ret_mode : str, optional
        Retrieval mode.
    overwrite : bool, optional
        If false, the file is only downloaded, if no file with the same
        name already exists.

    Returns
    -------
    file : str or StringIO or BytesIO
        The file name of the downloaded file.
        If `file_name` is ``None``, the file content is stored in
        either a `StringIO` or a `BytesIO` object.

    Warnings
    --------
    Even if you give valid input to this function, in rare cases the
    database might return no or malformed data to you.
    In these cases the request should be retried.
    When the issue occurs repeatedly, the error is probably in your
    input.

    See Also
    --------
    fetch : Fetch one or multiple entries as separate files.
    """
    if (
        file_name is not None
        and os.path.isfile(file_name)
        and getsize(file_name) > 0
        and not overwrite
    ):
        # Do no redownload the already existing file
        return file_name
    uid_list_str = ""
    for id in uids:
        uid_list_str += id + ","
    # Remove terminal comma
    uid_list_str = uid_list_str[:-1]
    param_dict = {
        "db": sanitize_database_name(db_name),
        "id": uid_list_str,
        "rettype": ret_type,
        "retmode": ret_mode,
        "tool": "Biotite",
        "mail": "padix.key@gmail.com",
    }
    api_key = get_api_key()
    if api_key is not None:
        param_dict["api_key"] = api_key
    r = requests.get(_fetch_url, params=param_dict)
    content = r.text
    check_for_errors(content)
    if content.startswith(" Error"):
        raise RequestError(content[8:])
    if file_name is None:
        return io.StringIO(content)
    else:
        with open(file_name, "w+") as f:
            f.write(content)
        return file_name
