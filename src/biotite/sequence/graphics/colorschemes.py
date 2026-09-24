from __future__ import annotations

__name__ = "biotite.sequence.graphics"
__author__ = "Patrick Kunzmann"
__all__ = [
    "ColorScheme",
    "get_color_scheme",
    "list_color_scheme_names",
    "load_color_scheme",
]

import glob
import json
import os
from dataclasses import dataclass
from os import PathLike
from os.path import dirname, join, realpath
from biotite.sequence.alphabet import Alphabet
from biotite.typing import MplColor


@dataclass(frozen=True)
class ColorScheme:
    """
    A color scheme assigning a color to each symbol of an alphabet.

    Attributes
    ----------
    name : str
        The name of the scheme.
    alphabet : Alphabet
        The alphabet the scheme is defined for.
    colors : list of (str or tuple or None), length=n
        A *Matplotlib* compatible color for each symbol in the
        `alphabet`, indexed by the symbol code.
        ``None`` indicates that no color is defined for the respective
        symbol.

    Examples
    --------

    >>> alphabet = NucleotideSequence.alphabet_unamb
    >>> scheme = ColorScheme("example", alphabet, ["red", "green", None, "blue"])
    >>> print(scheme.fit(alphabet, default="black"))
    ['red', 'green', 'black', 'blue']
    """

    name: str
    alphabet: Alphabet
    colors: list[MplColor | None]

    def __post_init__(self) -> None:
        if len(self.colors) != len(self.alphabet):
            raise ValueError(
                f"The scheme has {len(self.colors)} colors, "
                f"but the alphabet has {len(self.alphabet)} symbols"
            )

    def fit(self, alphabet: Alphabet, default: MplColor = "#FFFFFF") -> list[MplColor]:
        """
        Get the colors of this scheme for the symbols of the given
        alphabet.

        Parameters
        ----------
        alphabet : Alphabet
            The alphabet to obtain the colors for.
            The alphabet of this scheme must equal or extend this
            alphabet.
        default : str or tuple, optional
            A *Matplotlib* compatible color that is used for symbols that
            have no defined color in the scheme.

        Returns
        -------
        colors : list of (str or tuple), length=n
            A list of *Matplotlib* compatible colors.
            The colors in the list have the same order as the symbols in
            the given `alphabet`.
        """
        if not self.alphabet.extends(alphabet):
            raise ValueError(
                f"The scheme '{self.name}' does not cover the given alphabet"
            )
        # Only return colors that are in scope of this alphabet
        # and replace undefined colors with the default color
        return [
            color if color is not None else default
            for color in self.colors[: len(alphabet)]
        ]


def load_color_scheme(file_name: PathLike[str] | str) -> ColorScheme:
    """
    Load a color scheme from a JSON file.

    Parameters
    ----------
    file_name : str or PathLike
        The file name of the JSON file containing the scheme.

    Returns
    -------
    scheme : ColorScheme
        The loaded color scheme.
        Symbols of the alphabet, that are not defined in the file, get
        ``None`` as color.
    """
    with open(file_name, "r") as file:
        scheme = json.load(file)
    alphabet = Alphabet(scheme["alphabet"])
    # Store colors as symbol code ordered list of colors,
    # rather than dictionary
    colors: list[MplColor | None] = [None] * len(alphabet)
    for key, value in scheme["colors"].items():
        index = alphabet.encode(key)
        colors[index] = value
    return ColorScheme(scheme["name"], alphabet, colors)


def get_color_scheme(
    name: str,
    alphabet: Alphabet,
    default: MplColor = "#FFFFFF",
) -> list[MplColor]:
    """
    Get the colors of a built-in color scheme by name and alphabet.

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
    colors : list of (str or tuple)
        A list of *Matplotlib* compatible colors. The colors in the list
        have the same order as the symbols in the given `alphabet`.

    See Also
    --------
    ColorScheme.fit : The method used to fit the scheme to the alphabet.

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
    # Try exact alphabet match first
    for scheme in _color_schemes:
        if scheme.name == name and scheme.alphabet == alphabet:
            return scheme.fit(alphabet, default)
    # If no exact match was found, try to find a scheme for an alphabet
    # that extends the given alphabet
    for scheme in _color_schemes:
        if scheme.name == name and scheme.alphabet.extends(alphabet):
            return scheme.fit(alphabet, default)

    raise ValueError(f"Unknown scheme '{name}' for given alphabet")


def list_color_scheme_names(alphabet: Alphabet, strict: bool = False) -> list[str]:
    """
    Get a list of available built-in color scheme names for a given
    alphabet.

    Parameters
    ----------
    alphabet : Alphabet
        The alphabet to get the color scheme names for.
    strict : bool, optional
        If set to true, only schemes with an exact match to the given
        alphabet are included in the list.
        If set to false, schemes with an alphabet that extends the given
        alphabet are also included.

    Returns
    -------
    schemes : list of str
        A list of available color schemes.
    """
    scheme_list = []
    for scheme in _color_schemes:
        if strict and scheme.alphabet == alphabet:
            scheme_list.append(scheme.name)
        if not strict and scheme.alphabet.extends(alphabet):
            scheme_list.append(scheme.name)
    return scheme_list


_scheme_dir = join(dirname(realpath(__file__)), "color_schemes")

_color_schemes = [
    load_color_scheme(file_name)
    for file_name in glob.glob(_scheme_dir + os.sep + "*.json")
]
