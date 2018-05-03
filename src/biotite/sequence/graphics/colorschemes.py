# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["color_schemes"]

import numpy as np
from ..seqtypes import NucleotideSequence, ProteinSequence

_rainbow_protein_colors = np.array([
    (203, 245, 55 ),
    (245, 55,  165),
    (55,  122, 245),
    (55,  57,  245),
    (55,  245, 122),
    (234, 245, 55 ),
    (245, 55,  55 ),
    (140, 245, 55 ),
    (245, 55,  85 ),
    (109, 245, 55 ),
    (77,  245, 55 ),
    (189, 55,  245),
    (245, 191, 55 ),
    (144, 55,  245),
    (245, 100, 55 ),
    (245, 55,  211),
    (234, 55,  245),
    (171, 245, 55 ),
    (55,  245, 185),
    (55,  245, 255),
    (55,  122, 245),
    (55,  57,  245),
    (255, 255, 255),
    (255, 255, 255)
]) / 255


_rainbow_dna_colors = np.array([
    (55,  55,  245),
    (55,  245, 55 ),
    (245, 245, 55 ),
    (245, 55,  55 )
]) / 255



color_schemes = {
    NucleotideSequence.alphabet : {
        "rainbow" : _rainbow_dna_colors
    },
    ProteinSequence.alphabet : {
        "rainbow" : _rainbow_protein_colors
    }
}