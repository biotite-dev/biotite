# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "BondList",
    "BondType",
]

from enum import IntEnum
from biotite.rust.structure import BondList, bond_type_members


def _without_aromaticity(self):
    """
    Get the non-aromatic counterpart of this bond type.

    If this bond type is already non-aromatic, it is returned unchanged.

    Returns
    -------
    BondType
        The non-aromatic counterpart of this bond type.

    Examples
    --------
    >>> BondType.AROMATIC_DOUBLE.without_aromaticity()
    <BondType.DOUBLE: 2>
    >>> BondType.SINGLE.without_aromaticity()
    <BondType.SINGLE: 1>
    """
    match self:
        case BondType.AROMATIC_SINGLE:
            return BondType.SINGLE
        case BondType.AROMATIC_DOUBLE:
            return BondType.DOUBLE
        case BondType.AROMATIC_TRIPLE:
            return BondType.TRIPLE
        case BondType.AROMATIC:
            return BondType.ANY
        case _:
            return self


# Create BondType IntEnum dynamically from Rust enum members
BondType = IntEnum(
    "BondType",
    {name: value for name, value in bond_type_members().items()},
    module=__name__,
)
BondType.__doc__ = """
This enum type represents the type of a chemical bond.

- ``ANY`` - Used if the actual type is unknown
- ``SINGLE`` - Single bond
- ``DOUBLE`` - Double bond
- ``TRIPLE`` - Triple bond
- ``QUADRUPLE`` - A quadruple bond
- ``AROMATIC_SINGLE`` - Aromatic bond with a single formal bond
- ``AROMATIC_DOUBLE`` - Aromatic bond with a double formal bond
- ``AROMATIC_TRIPLE`` - Aromatic bond with a triple formal bond
- ``COORDINATION`` - Coordination complex involving a metal atom
- ``AROMATIC`` - Aromatic bond without specification of the formal bond
"""
BondType.without_aromaticity = _without_aromaticity
