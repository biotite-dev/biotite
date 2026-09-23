"""
This module contains all possible errors of the `structure` subpackage.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "BadStructureError",
    "InconsistentBondTypeWarning",
    "IncompleteStructureWarning",
    "UnexpectedStructureWarning",
]


class BadStructureError(Exception):
    """
    Indicates that a structure is not suitable for a certain operation.
    """

    pass


class InconsistentBondTypeWarning(Warning):
    """
    Indicates that the bond types are inconsistent with the valences and formal charges
    of the atoms.
    """

    pass


class IncompleteStructureWarning(Warning):
    """
    Indicates that a structure is not complete.
    """

    pass


class UnexpectedStructureWarning(Warning):
    """
    Indicates that a structure was not expected.
    """

    pass
