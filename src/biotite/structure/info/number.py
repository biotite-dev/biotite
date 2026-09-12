# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["atomic_number"]

import numpy as np
from biotite.structure.atoms import AtomArray
from biotite.typing import N, NDArray1

# fmt: off
_ATOMIC_NUMBERS = {
    "H":  1,   "HE": 2,   "LI": 3,   "BE": 4,   "B":  5,   "C":  6,
    "N":  7,   "O":  8,   "F":  9,   "NE": 10,  "NA": 11,  "MG": 12,
    "AL": 13,  "SI": 14,  "P":  15,  "S":  16,  "CL": 17,  "AR": 18,
    "K":  19,  "CA": 20,  "SC": 21,  "TI": 22,  "V":  23,  "CR": 24,
    "MN": 25,  "FE": 26,  "CO": 27,  "NI": 28,  "CU": 29,  "ZN": 30,
    "GA": 31,  "GE": 32,  "AS": 33,  "SE": 34,  "BR": 35,  "KR": 36,
    "RB": 37,  "SR": 38,  "Y":  39,  "ZR": 40,  "NB": 41,  "MO": 42,
    "TC": 43,  "RU": 44,  "RH": 45,  "PD": 46,  "AG": 47,  "CD": 48,
    "IN": 49,  "SN": 50,  "SB": 51,  "TE": 52,  "I":  53,  "XE": 54,
    "CS": 55,  "BA": 56,  "LA": 57,  "CE": 58,  "PR": 59,  "ND": 60,
    "PM": 61,  "SM": 62,  "EU": 63,  "GD": 64,  "TB": 65,  "DY": 66,
    "HO": 67,  "ER": 68,  "TM": 69,  "YB": 70,  "LU": 71,  "HF": 72,
    "TA": 73,  "W":  74,  "RE": 75,  "OS": 76,  "IR": 77,  "PT": 78,
    "AU": 79,  "HG": 80,  "TL": 81,  "PB": 82,  "BI": 83,  "PO": 84,
    "AT": 85,  "RN": 86,  "FR": 87,  "RA": 88,  "AC": 89,  "TH": 90,
    "PA": 91,  "U":  92,  "NP": 93,  "PU": 94,  "AM": 95,  "CM": 96,
    "BK": 97,  "CF": 98,  "ES": 99,  "FM": 100, "MD": 101, "NO": 102,
    "LR": 103, "RF": 104, "DB": 105, "SG": 106, "BH": 107, "HS": 108,
    "MT": 109, "DS": 110, "RG": 111, "CN": 112, "NH": 113, "FL": 114,
    "MC": 115, "LV": 116, "TS": 117, "OG": 118,
}
# fmt: on


def atomic_number(
    atoms: AtomArray[N] | NDArray1[N, np.str_],
) -> NDArray1[N, np.integer]:
    """
    Get the atomic number for each atom in the given structure.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or ndarray, shape=(n,), dtype=str
        The atoms to get the atomic numbers for.
        Alternatively, the elements can be given directly.

    Returns
    -------
    numbers : ndarray, shape=(n,), dtype=uint8
        The atomic number for each atom.
        Unknown elements are indicated by ``0``.

    Examples
    --------

    >>> molecule = residue("URA")
    >>> print(molecule.element)
    ['N' 'C' 'O' 'N' 'C' 'O' 'C' 'C' 'H' 'H' 'H' 'H']
    >>> print(atomic_number(molecule))
    [7 6 8 7 6 8 6 6 1 1 1 1]
    """
    if isinstance(atoms, AtomArray):
        elements = atoms.element
    else:
        elements = atoms
    # Look up each element only once instead of each atom
    unique_elements, inverse = np.unique(elements, return_inverse=True)
    unique_numbers = np.array(
        [_ATOMIC_NUMBERS.get(element.upper(), 0) for element in unique_elements],
        dtype=np.uint8,
    )
    return unique_numbers[inverse]
