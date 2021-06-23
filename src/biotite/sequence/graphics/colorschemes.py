# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.graphics"
__author__ = "Patrick Kunzmann"
__all__ = ["get_color_scheme", "list_color_scheme_names", "load_color_scheme"]

import numpy as np
import json
from os.path import join, dirname, realpath
import glob
import os
from ..alphabet import Alphabet


def load_color_scheme(file_name):
    """
    Load a color scheme from a JSON file.

    A color scheme is a list of colors that correspond to symbols of an
    alphabet. The color for a symbol is list of colors indexed by the
    corresponding symbol code.

    Parameters
    ----------
    file_name : str
        The file name of the JSON file containing the scheme.
    
    Returns
    -------
    scheme : dict
        A dictionary representing the color scheme, It contains the
        following keys, if the input file is proper:
        
           - **name** - Name of the scheme.
           - **alphabet** - :class:`Alphabet` instance describing the
             type of sequence the scheme can be used for.
           - **colors** - List of *Matplotlib* compatible colors
    """
    with open(file_name, "r") as file:
        scheme = json.load(file)
        alphabet = Alphabet(scheme["alphabet"])
        # Store alphabet as 'Alphabet' object
        scheme["alphabet"] = alphabet
        colors = [None] * len(alphabet)
        for key, value in scheme["colors"].items():
            index = alphabet.encode(key)
            colors[index] = value
        # Store colors as symbol code ordered list of colors,
        # rather than dictionary
        scheme["colors"] = colors
        return scheme


def get_color_scheme(name, alphabet, default="#FFFFFF"):
    """
    Get a color scheme by name and alphabet.

    A color scheme is a list of colors that correspond to symbols of an
    alphabet. The color for a symbol is list of colors indexed by the
    corresponding symbol code.

    Parameters
    ----------
    name : str
        The name of the color scheme.
    alphabet : Alphabet
        The alphabet to obtain the scheme for. The alphabet of the
        scheme must equal or extend this parameter.
    default : str or tuple, optional
        A *Matplotlib* compatible color that is used for symbols that
        have no defined color in the scheme.
    
    Returns
    -------
    colors : list
        A list of *Matplotlib* compatible colors. The colors in the list
        have the same order as the symbols in the given `alphabet`.
        Since the alphabet of the color scheme may extend the given
        `alphabet`, the list of colors can be longer than the
        `alphabet`.

    Notes
    -----
    There can be multiple color schemes with the same name but for
    different alphabets (e.g. one for dna and one for protein
    sequences).

    Examples
    --------

    >>> alphabet = NucleotideSequence.alphabet_unamb
    >>> color_scheme = get_color_scheme("rainbow", alphabet)
    >>> print(color_scheme)
    ['#3737f5', '#37f537', '#f5f537', '#f53737']
    """
    for scheme in _color_schemes:
        if scheme["name"] == name and scheme["alphabet"].extends(alphabet):
            colors = scheme["colors"]
            # Replace None values with default color
            colors = [color if color is not None else default
                      for color in colors]
            # Only return colors that are in scope of this alphabet
            # and not the extended alphabet
            return colors[:len(alphabet)]
    raise ValueError(f"Unkown scheme '{name}' for given alphabet")


def list_color_scheme_names(alphabet):
    """
    Get a list of available color scheme names for a given alphabet.

    Parameters
    ----------
    alphabet : Alphabet
        The alphbet to get the color scheme names for.
        The alphabet of the scheme must equal or extend this parameter,
        to be included in the list.
    
    Returns
    -------
    schemes : list of str
        A list of available color schemes.
    """
    scheme_list = []
    for scheme in _color_schemes:
        if scheme["alphabet"].extends(alphabet):
            scheme_list.append(scheme["name"])
    return scheme_list


_scheme_dir = join(dirname(realpath(__file__)), "color_schemes")

_color_schemes = []

for file_name in glob.glob(_scheme_dir + os.sep + "*.json"):
    scheme = load_color_scheme(file_name)
    _color_schemes.append(scheme)