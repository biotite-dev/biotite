# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.entrez"
__author__ = "Patrick Kunzmann"
__all__ = ["set_api_key", "get_api_key"]


_API_KEY = None


def get_api_key():
    """
    Get the
    `NCBI API key <https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/>`_.

    Returns
    -------
    api_key : str or None
        The API key, if it was already set before, ``None`` otherwise.
    """
    global _API_KEY
    return _API_KEY


def set_api_key(key):
    """
    Set the
    `NCBI API key <https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/>`_.

    Using an API key increases the request limit on the NCBI servers
    and is automatically used by functions in
    :mod:`biotite.database.entrez`.
    This key is kept only in memory and hence removed in the end of the
    Python session.

    Parameters
    ----------
    key : str
        The API key.
    """
    global _API_KEY
    _API_KEY = key
