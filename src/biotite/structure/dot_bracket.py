# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module handles conversion of RNA structures to
dot-bracket-notation.
"""

__name__ = "biotite.structure"
__author__ = "Tom David Müller"
__all__ = ["dot_bracket"]

import numpy as np
from .basepairs import base_pairs
from .pseudoknots import pseudoknots
from .residues import get_residue_starts

_OPENING_BRACKETS = "([<ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_CLOSING_BRACKETS = ")]>abcdefghijklmnopqrstuvwxyz"


def dot_bracket(nucleic_acid_strand):
    """
    Convert structural information

    Args:
        nucleic_acid_strand ([type]): [description]

    Returns:
        [type]: [description]
    """
    basepairs = base_pairs(nucleic_acid_strand)
    pseudoknot_order = pseudoknots(basepairs)
    residue_starts = get_residue_starts(nucleic_acid_strand)

    notation = [""]*len(pseudoknot_order)
    for s, solution in enumerate(pseudoknot_order):
        opened_brackets = set()
        for residue_start in residue_starts:
            if residue_start not in basepairs:
                notation[s] += "."
            else:
                # Get position in ``basepairs`` and ``pseudoknot_order``
                pos = np.where(basepairs == residue_start)[0][0]
                if residue_start in opened_brackets:
                    notation[s] += _CLOSING_BRACKETS[solution[pos]]
                else:
                    for base in basepairs[pos]:
                        opened_brackets.add(base)
                    notation[s] += _OPENING_BRACKETS[solution[pos]]

    return notation
