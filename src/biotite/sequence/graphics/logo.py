# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.graphics"
__author__ = "Patrick Kunzmann"
__all__ = ["plot_sequence_logo"]

import numpy as np
from ...visualize import set_font_size_in_coord
from ..alphabet import LetterAlphabet
from .colorschemes import get_color_scheme
import warnings
from ..align import Alignment
from .. import SequenceProfile


def plot_sequence_logo(axes, profile, scheme=None, **kwargs):
    """
    Create a sequence logo. :footcite:`Schneider1990`

    A sequence logo is visualizes the positional composition and
    conservation of a profile encoded in the size of the letters.
    Each position displays all symbols that are occurring at this
    position stacked on each other, with their relative heights depicting
    their relative frequency.
    The height of such a stack depicts its conservation.
    It is the maximum possible Shannon entropy of the alphabet
    subtracted by the positional entropy.

    Parameters
    ----------
    axes : Axes
        The axes to draw the logo one.
    profile: SequenceProfile
        The logo is created based on this profile.
    scheme : str or list of (tuple or str)
        Either a valid color scheme name
        (e.g. ``"rainbow"``, ``"clustalx"``, ``blossom``, etc.)
        or a list of *Matplotlib* compatible colors.
        The list length must be at least as long as the
        length of the alphabet used by the `profile`.
    **kwargs
        Additional `text parameters <https://matplotlib.org/api/text_api.html#matplotlib.text.Text>`_.
    
    References
    ----------
    
    .. footbibliography::
    """
    from matplotlib.text import Text

    if isinstance(profile, Alignment):
        warnings.warn("Using an alignment for this method is deprecated; use a profile instead", DeprecationWarning)
        profile = SequenceProfile.from_alignment(profile)

    alphabet = profile.alphabet
    if not isinstance(alphabet, LetterAlphabet):
        raise TypeError("The sequences' alphabet must be a letter alphabet")

    if scheme is None:
        colors = get_color_scheme("rainbow", alphabet)
    elif isinstance(scheme, str):
        colors = get_color_scheme(scheme, alphabet)
    else:
        colors = scheme
    
    # 'color' and 'size' property is not passed on to text
    kwargs.pop("color", None)
    kwargs.pop("size",  None)
    
    frequencies, entropies, max_entropy = _get_entropy(profile)
    stack_heights = (max_entropy - entropies)
    symbols_heights = stack_heights[:, np.newaxis] * frequencies
    index_order = np.argsort(symbols_heights, axis=1)
    for i in range(symbols_heights.shape[0]):
        # Iterate over the alignment columns
        index_order = np.argsort(symbols_heights)
        start_height = 0
        for j in index_order[i]:
            # Stack the symbols at position on top of the preceeding one
            height = symbols_heights[i,j]
            if height > 0:
                symbol = alphabet.decode(j)
                text = axes.text(
                    i+0.5, start_height, symbol,
                    ha="left", va="bottom", color=colors[j],
                    # Best results are obtained with this font size
                    size=1,
                    **kwargs
                )
                text.set_clip_on(True)
                set_font_size_in_coord(text, width=1, height=height)
                start_height += height

    axes.set_xlim(0.5, len(profile.symbols)+0.5)
    axes.set_ylim(0, max_entropy)


def _get_entropy(profile):
    freq = profile.symbols
    freq = freq / np.sum(freq, axis=1)[:, np.newaxis]
    # 0 * log2(0) = 0 -> Convert NaN to 0
    no_zeros = freq != 0
    pre_entropies = np.zeros(freq.shape)
    pre_entropies[no_zeros] \
        = freq[no_zeros] * np.log2(freq[no_zeros])
    entropies = -np.sum(pre_entropies, axis=1)
    max_entropy = np.log2(len(profile.alphabet))
    return freq, entropies, max_entropy