__name__ = "biotite.interface"
__author__ = "Patrick Kunzmann"
__all__ = ["LossyConversionWarning"]


class LossyConversionWarning(UserWarning):
    """
    Warning raised, when some information is lost during conversion.

    Note that most conversion functions will be inherently lossy to some extent.
    This warning is only raised, when the loss of information happens only for
    some edge case.
    """

    pass
