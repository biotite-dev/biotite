# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database"
__author__ = "Patrick Kunzmann"
__all__ = ["RequestError"]


class RequestError(Exception):
    """
    Indicates that the database returned a response with an error
    message or other malformed content.
    """
    pass