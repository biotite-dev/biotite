# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Tom David Müller"
__all__ = ["is_nucleotide"]

import json
from os.path import join, dirname, realpath


_info_dir = dirname(realpath(__file__))
# TODO: components.cif version
with open(join(_info_dir, "nucleotides.json"), "r") as file:
    _nucleotides = json.load(file)

def is_nucleotide(three_letter_code):
    """TODO: Docstring
    """
    print(three_letter_code)
    if three_letter_code in _nucleotides:
        return True
    return False