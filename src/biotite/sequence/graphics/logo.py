# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["SequenceLogo"]

import numpy as np
from ...visualize import Visualizer
from .colorschemes import get_color_scheme

class SequenceLogo(Visualizer):
    """
    This class is used for creation of sequence logos. [1]_

    EXPERIMENTAL: Future API changes are probable.

    Parameters
    ----------
    alignment : Alignment
        The logo is created based on this alignment.
    width : float
        The width of the figure (pixels).
    height : float
        The height of the figure (pixels).
    
    References
    ----------
    
    .. [1] TD Schneider and RM Stephens,
       "Sequence logos: a new way to display consensus sequences"
       Nucleic Acids Res, 18, 6097-6100 (1990).
    """
    
    def __init__(self, alignment, width, height):
        super().__init__()
        # Check if all sequences share the same alphabet
        sequences = alignment.sequences
        self._alphabet = sequences[0].get_alphabet()
        for seq in sequences:
            if seq.get_alphabet() != self._alphabet:
                raise ValueError("Alphabets of the sequences in the alignment "
                                 "are not equal")
        
        trace = alignment.trace
        self._freq = np.zeros((len(trace), len(self._alphabet)))
        for i in range(trace.shape[0]):
            for j in range(trace.shape[1]):
                index = trace[i,j]
                if index != -1:
                    code = sequences[j].code[index]
                    self._freq[i, code] += 1
        self._freq = self._freq / np.sum(self._freq, axis=1)[:, np.newaxis]
        # 0 * log2(0) = 0 -> Convert NaN to 0
        no_zeros = self._freq != 0
        pre_entropies = np.zeros(self._freq.shape)
        pre_entropies[no_zeros] \
            = self._freq[no_zeros] * np.log2(self._freq[no_zeros])
        ## 0 * log2(0) = 0 -> Convert NaN to 0
        #pre_entropies[np.isnan(pre_entropies)] = 0
        self._entropies = -np.sum(pre_entropies, axis=1)
        self._max_entropy = np.log2(len(self._alphabet))

        self._width = width
        self._height = height
        self._font = None
        self._colors = get_color_scheme("rainbow", self._alphabet)
    
    def set_font(self, font):
        """
        Set the font used for the symbols in the logo.

        Parameters
        ----------
        font : FontProperties
            The font that should be used for symbol rendering.
        """
        self._font = font
    
    def set_color_scheme(self, scheme):
        """
        Set the color scheme used for the logo.

        Parameters
        ----------
        scheme : str or list of (tuple or str)
            Either a valid color scheme name
            (e.g. ``"rainbow"``, ``"clustalx"``, etc.)
            or a list of `matplotlib` compatible colors.
            The list length must be at least as long as the
            length of the alphabet used by the sequences.
        """
        if isinstance(scheme, str):
            self._colors = get_color_scheme(scheme, self._alphabet)
        else:
            self._colors = scheme

    def generate(self):
        from matplotlib.patches import Rectangle
        from matplotlib.text import Text
        from matplotlib.patheffects import AbstractPathEffect

        class ScaleEffect(AbstractPathEffect):
            def __init__(self, scale_x, scale_y):
                self._scale_x = scale_x
                self._scale_y = scale_y

            def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
                affine = affine \
                        .identity() \
                        .scale(self._scale_x, self._scale_y) \
                        + affine
                renderer.draw_path(gc, tpath, affine, rgbFace)

        fig = self.create_figure(size=(self._width, self._height))
        renderer = fig.canvas.get_renderer()
        
        symbol_width = self._width / len(self._entropies)
        pos_heights = (1 - self._entropies/self._max_entropy) * self._height
        symbols_heights = pos_heights[:, np.newaxis] * self._freq
        index_order = np.argsort(symbols_heights, axis=1)
        for i in range(symbols_heights.shape[0]):
            index_order = np.argsort(symbols_heights)
            start_height = 0
            for j in index_order[i]:
                height = symbols_heights[i,j]
                if height > 0:
                    symbol = self._alphabet.decode(j)
                    text = Text(i*symbol_width, start_height, symbol,
                                ha="left", va="bottom", color=self._colors[j],
                                fontproperties=self._font, figure=fig)
                    fig.texts.append(text)
                    # Rescale symbols,
                    # so that they fit the given width and height
                    # Scale factor is desired size
                    # divided by current size
                    bounds = text.get_window_extent(renderer=renderer).bounds
                    text.set_path_effects([
                        ScaleEffect(symbol_width / bounds[2],
                                    height / bounds[3])
                    ])
                    start_height += height
        return fig