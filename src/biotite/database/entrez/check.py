# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.entrez"
__author__ = "Patrick Kunzmann, Maximilian Dombrowsky"
__all__ = ["check_for_errors"]

import json
from biotite.database.error import RequestError

# Taken from https://github.com/kblin/ncbi-entrez-error-messages
_error_messages = [
    "Error reading from remote server",
    "Bad gateway",
    "Bad Gateway",
    "Cannot process ID list",
    "server is temporarily unable to service your request",
    "Service unavailable",
    "Server Error",
    "ID list is empty",
    "Supplied id parameter is empty",
    "Resource temporarily unavailable",
    "Failed to retrieve sequence",
    "Failed to understand id",
]


def check_for_errors(message):
    """
    Check for common error messages in NCBI Entrez database responses.

    Parameters
    ----------
    message : str
        The message received from NCBI Entrez.

    Raises
    ------
    RequestError
        If the message contains an error message.
    """
    # Server can respond short JSON error messages
    if len(message) < 500:
        try:
            message_json = json.loads(message)
            if "error" in message_json:
                raise RequestError(message_json["error"])
        except json.decoder.JSONDecodeError:
            # It is not a JSON message
            pass

    # Error always appear at the end of message
    message_end = message[-200:]
    # Seemingly arbitrary '+' characters are in NCBI error messages
    message_end = message_end.replace("+", "")
    for error_msg in _error_messages:
        # Often whitespace is also replaced by '+' in error message
        if error_msg.replace(" ", "") in message_end:
            raise RequestError(error_msg)
