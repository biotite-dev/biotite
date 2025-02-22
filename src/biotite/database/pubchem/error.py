# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.pubchem"
__author__ = "Patrick Kunzmann"
__all__ = ["parse_error_details"]


def parse_error_details(response_text):
    """
    Parse the ``Detail: ...`` or alternatively ``Message: ...`` part of
    an error response.

    Parameters
    ----------
    response_text : str
        The text of the response.

    Returns
    -------
    error_details : str
        The error details.
    """
    for message_line_indicator in ["Detail: ", "Message: "]:
        for line in response_text.splitlines():
            if line.startswith(message_line_indicator):
                return line[len(message_line_indicator) :]
    # No 'Detail: ...' or 'Message: ' line found
    return "Unknown error"
