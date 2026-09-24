__name__ = "biotite.database"
__author__ = "Patrick Kunzmann"
__all__ = ["RequestError"]


class RequestError(Exception):
    """
    Indicates that the database returned a response with an error
    message or other malformed content.
    """

    pass
