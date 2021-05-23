# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.uniprot"
__author__ = "Maximilian Greil"
__all__ = ["assert_valid_response"]

from ..error import RequestError


# Taken from https://www.uniprot.org/help/api_retrieve_entries
def assert_valid_response(response_status_code):
    """
    Checks whether the response is valid.

    Parameters
    ----------
    response_status_code: int
        Status code of request.get.
    """
    if response_status_code == 400:
        raise RequestError("Bad request. There is a problem with your input.")
    elif response_status_code == 404:
        raise RequestError("Not found. The resource you requested doesn't exist.")
    elif response_status_code == 410:
        raise RequestError("Gone. The resource you requested was removed.")
    elif response_status_code == 500:
        raise RequestError(
            "Internal server error. Most likely a temporary problem, but if the problem persists please contact UniProt team.")
    elif response_status_code == 503:
        raise RequestError("Service not available. The server is being updated, try again later.")
