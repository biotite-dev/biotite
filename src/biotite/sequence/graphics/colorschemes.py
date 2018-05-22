# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["get_color_scheme"]

import numpy as np
import json
from os.path import join, dirname, realpath
import glob
import os
from ..alphabet import Alphabet


_scheme_dir = join(dirname(realpath(__file__)), "color_schemes")

_color_schemes = []

for file_name in glob.glob(_scheme_dir + os.sep + "*.json"):
    with open(file_name, "r") as file:
        scheme = json.load(file)
        alphabet = Alphabet(scheme["alphabet"])
        # Store alphabet as 'Alphabet' object
        scheme["alphabet"] = alphabet
        colors = [None] * len(scheme["colors"])
        for key, value in scheme["colors"].items():
            index = alphabet.encode(key)
            colors[index] = value
        _color_schemes.append(scheme)
        # Store colors as symbol code ordered list of colors,
        # rather than dictionary
        scheme["colors"] = colors
        _color_schemes.append(scheme)

def get_color_scheme(name, alphabet, default="#FFFFFF"):
    for scheme in _color_schemes:
        if scheme["name"] == name and scheme["alphabet"].extends(alphabet):
            colors = scheme["colors"]
            # Replace None values with default color
            colors = [color if color is not None else default
                      for color in colors]
            return colors
    raise ValueError("Unkown scheme '{:}' for given alphabet".format(name))