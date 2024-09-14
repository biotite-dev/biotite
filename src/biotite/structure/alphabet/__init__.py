# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
NumPy port of the ``foldseek`` code for encoding structures to 3di.
"""

__name__ = "biotite.structure.alphabet"
__author__ = "Martin Larralde"
__all__ = ["Encoder", "StructureSequence", "to_3di"]

from .encoder import Encoder
from .sequence import StructureSequence

def to_3di(array):
    r"""
    Encode the atoms to the 3di structure alphabet.

    Parameters
    ----------
    atoms : AtomArray
        The atom array to encode to 3di. All atoms must be part of
        a single chain.

    Returns
    -------
    sequence : StructureSequence
        The encoded structure sequence.

    Note
    ----
    To encode atoms in different chains, use :func:`apply_chain_wise` to
    return a list with one sequence per chain.
    """
    encoder = Encoder()
    sequence = StructureSequence()
    sequence.code = encoder.encode_atoms(array).filled()
    return sequence
