# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["SymbolPlotter", "LetterPlotter", "LetterSimilarityPlotter",
           "LetterTypePlotter", "plot_alignment"]

import abc
import numpy as np
from ...visualize import set_font_size_in_coord, colors
from .colorschemes import get_color_scheme


class SymbolPlotter(metaclass=abc.ABCMeta):

    def __init__(self, axes):
        self._axes = axes
    
    @property
    def axes(self):
        return self._axes

    @abc.abstractmethod
    def plot_symbol(self, bbox, alignment, column_i, seq_i):
        """
        Get the color of a symbol at a specified position in the
        alignment.

        The symbol is specified as position in the alignment's trace
        (``trace[pos_i, seq_i]``).

        PROTECTED: Override when inheriting. 

        Parameters
        ----------
        bbox : Bbox
            The axes area to plot the symbol in
        alignment : Alignment
            The respective alignment.
        column_i : int
            The position index in the trace.
        seq_i : int
            The sequence index in the trace.
        """
        pass


class LetterPlotter(SymbolPlotter, metaclass=abc.ABCMeta):

    def __init__(self, axes, color_symbols=False, font_param=None):
        super().__init__(axes)
        self._color_symbols = color_symbols
        self._font_param = font_param if font_param is not None else {}

    def plot_symbol(self, bbox, alignment, column_i, seq_i):
        from matplotlib.patches import Rectangle
        
        trace = alignment.trace
        if trace[column_i,seq_i] != -1:
            symbol = alignment.sequences[seq_i][trace[column_i,seq_i]]
        else:
            symbol = "-"
        color = self.get_color(alignment, column_i, seq_i)
        
        box = Rectangle(bbox.p0, bbox.width, bbox.height)
        self.axes.add_patch(box)
        text = self.axes.text(
            bbox.x0 + bbox.width/2, bbox.y0 + bbox.height/2,
            symbol, color="black", ha="center", va="center",
            size=10, **self._font_param)
        text.set_clip_on(True)
        #set_font_size_in_coord(text, bbox.width, bbox.height, "maximum")
        
        if self._color_symbols:
            box.set_color("None")
            text.set_color(color)
        else:
            box.set_color(color)
    
    @abc.abstractmethod
    def get_color(self, alignment, column_i, seq_i):
        """
        Get the color of a symbol at a specified position in the
        alignment.

        The symbol is specified as position in the alignment's trace
        (``trace[pos_i, seq_i]``).

        PROTECTED: Override when inheriting. 

        Parameters
        ----------
        alignment : Alignment
            The respective alignment.
        column_i : int
            The position index in the trace.
        seq_i : int
            The sequence index in the trace.
        """
        pass


class LetterSimilarityPlotter(LetterPlotter):
    r"""
    This `AlignmentVisualizer` colors the symbols based on the
    similarity with the other symbols in the same column.

    The color intensity (or colormap value, repsectively) of a symbols
    scales with similarity of the respective symbol to the other symbols
    in the same alignment column.

    EXPERIMENTAL: Future API changes are probable.

    Parameters
    ----------
    alignment : Alignment
        The alignment to be visualized.
        All sequences in the alignment must have an equal alphabet.
    matrix : SubstitutionMatrix, optional
        The substitution matrix to use the similarity scores from.
        By default the normalized similarity is 1 for identity and 0
        for non-identity.
    
    Notes
    -----
    For determination of the color, this `AlignmentVisualizer` uses a
    measure called *average normalized similarity*.

    The *normalized similarity* of one symbol *a* to another symbol *b*
    (both in aphabet *X*) is defined as

    .. math:: S_{norm}(a,b) = \frac{S(a,b) - \min\limits_x(S(a,x))} {\max\limits_x(S(a,x)) - \min\limits_x(S(a,x))}

    .. math:: a,b,x \in X

    where *S(x,y)* is the similarity score of the two symbols
    *x* and *y* described in the substitution matrix.
    The similarity *S(x,-)* is always 0.
    As the normalization is conducted only with respect to *a*,
    the *normalized similarity* is not commutative.

    The *average normalized similarity* of a symbol *a* is
    determined by averaging the normalized similarity over each
    symbol *b*\ :sub:`i` in the same alignment column.

    .. math:: S_{norm,av}(a) = \frac{1}{n-1} \left[\left(\sum\limits_{i=1}^n S_{norm}(a,b_i)\right) - S_{norm}(a,a)\right]

    The normalized similarity of *a* to itself is subtracted,
    because *a* does also occur in *b*\ :sub:`i`.
    """

    def __init__(self, axes, matrix=None,
                 color_symbols=False, font_param=None):
        from matplotlib import cm

        super().__init__(axes, color_symbols, font_param)
        if matrix is not None:
            self._matrix = matrix.score_matrix()
        else:
            self._matrix = None
        # Default colormap
        self._cmap = self._generate_colormap(colors["dimgreen"],
                                             self._color_symbols)
    
    def set_color(self, color=None, cmap=None):
        """
        Set the alignemnt colors used by the `AlignmentVisualizer`.

        This function takes either a color or a colormap.

        Parameters
        ----------
        color : tuple or str, optional
            A `matplotlib` compatible color.
            If this parameter is given, the box color in an interpolated
            value between white and the given color,
            or, if `color_symbols` is set, between the given color and
            black.
            The interpolation percentage is given by the normalized
            similarity.
        cmap : Colormap, optional
            The boxes (or symbols, if `color_symbols` is set) are
            colored based on the normalized similarity value on the
            given Colormap.
        """
        from matplotlib import cm

        if color is None and cmap is None:
            raise ValueError("Either color or colormap must be set")
        elif color is not None:
            self._cmap = self._generate_colormap(color, self._color_symbols)
        else:
            # cmap is not None
            if isinstance(cmap, str):
                self._cmap = cm.get_cmap(cmap)
            else:
                # cmap is a colormap
                self._cmap = cmap

    def get_color(self, alignment, column_i, seq_i):
        # Calculate average normalize similarity 
        index1 = alignment.trace[column_i, seq_i]
        if index1 == -1:
            similarity = 0
        else:
            code1 = alignment.sequences[seq_i].code[index1]
            similarities = np.zeros(alignment.trace.shape[1])
            for i in range(alignment.trace.shape[1]):
                index2 = alignment.trace[column_i, i]
                if index2 == -1:
                    similarities[i] = 0
                else:
                    code2 = alignment.sequences[i].code[index2]
                    similarities[i] = self._get_similarity(self._matrix,
                                                           code1, code2)
            # Delete self-similarity
            similarities = np.delete(similarities, seq_i)
            similarity = np.average(similarities)
        return self._cmap(similarity)
    
    def _get_similarity(self, matrix, code1, code2):
        if matrix is None:
            return 1 if code1 == code2 else 0
        else:
            sim = matrix[code1, code2]
            # Normalize (range 0.0 - 1.0)
            min_sim = np.min(matrix[code1])
            max_sim = np.max(matrix[code1])
            sim = (sim - min_sim) / (max_sim - min_sim)
            return sim
    
    @staticmethod
    def _generate_colormap(color, to_black):
        from matplotlib.colors import ListedColormap, to_rgb

        color = to_rgb(color)
        if to_black:
            # From color to black
            cmap_val = np.stack(
                [np.interp(np.linspace(0, 1, 100), [0, 1], [color[i], 0])
                 for i in range(len(color))]
            ).transpose()
        else:
            # From white to color
            cmap_val = np.stack(
                [np.interp(np.linspace(0, 1, 100), [0, 1], [1, color[i]])
                 for i in range(len(color))]
            ).transpose()
        return ListedColormap(cmap_val)


class LetterTypePlotter(LetterPlotter):
    """
    This `AlignmentVisualizer` colors each symbol based on the general
    color of that symbol defined by a color scheme.

    EXPERIMENTAL: Future API changes are probable.

    Parameters
    ----------
    alignment : Alignment
        The alignment to be visualized.
        All sequences in the alignment must have an equal alphabet.
    """

    def __init__(self, axes, color_scheme=None,
                 color_symbols=False, font_param=None):
        super().__init__(axes, color_symbols, font_param)
        alphabet = alignment.sequences[0].get_alphabet()
        
        if color_scheme is None:
            self._colors = get_color_scheme("rainbow", alphabet)
        if isinstance(color_scheme, str):
            alphabet = self._alignment.sequences[0].get_alphabet()
            self._colors = get_color_scheme(color_scheme, alphabet)
        else:
            self._colors = color_scheme
    
    def get_color(self, alignment, column_i, seq_i):
        index = alignment.trace[column_i, seq_i]
        if index == -1:
            # Gaps are white
            return (1, 1, 1)
        code = alignment.sequences[seq_i].code[index]
        return self._colors[code]


def plot_alignment(axes, alignment, symbol_plotter, symbols_per_line=50,
                   show_numbers=False, number_font_size=None,
                   number_font_param=None, number_func=None,
                   labels=None, label_font_size=None, label_font_param=None,
                   spacing=1):
        from matplotlib.transforms import Bbox

        if number_func is None:
            number_func = [lambda x: x + 1] * len(alignment.sequences)

        seq_num = alignment.trace.shape[1]
        seq_len = alignment.trace.shape[0]
        line_count = seq_len // symbols_per_line
        # Only extend line count by 1 if there is a remainder
        # (remaining symbols)
        if seq_len % symbols_per_line != 0:
            line_count += 1

        """
        ### Draw labels ###
        if self._labels is not None:
            y = fig_size_y - self._margin
            y -= self._box_size[1] / 2
            for i in range(line_count):
                for j in range(seq_num):
                    label = self._labels[j]
                    text = Text(self._margin, y, label,
                                color="black", ha="left", va="center",
                                size=self._label_font_size, figure=fig,
                                fontproperties=self._label_font)
                    fig.texts.append(text)
                    y -= self._box_size[1]
                y -= self._spacing
        """
        
        """
        ### Draw numbers  ###
        if self._show_numbers:
            y = fig_size_y - self._margin
            y -= self._box_size[1] / 2
            for i in range(line_count):
                for j in range(seq_num):
                    if i == line_count-1:
                        # Last line -> get number of last column in trace
                        trace_pos = len(self._alignment.trace) -1
                    else:
                        trace_pos = (i+1) * self._symbols_per_line -1
                    seq_index = self._get_last_real_index(self._alignment,
                                                          trace_pos, j)
                    # if -1 -> terminal gap
                    # -> skip number for this sequence in this line
                    if seq_index != -1:
                        number = self._number_func[j](seq_index)
                        text = Text(fig_size_x - self._margin, y, str(number),
                                    color="black", ha="right", va="center",
                                    size=self._number_font_size, figure=fig,
                                    fontproperties=self._number_font)
                        fig.texts.append(text)
                    y -= self._box_size[1]
                y -= self._spacing
        """

        ### Draw symbols in boxes ###
        x = 0
        y = 0
        y_start = 0
        line_pos = 0
        for i in range(seq_len):
            y = y_start
            for j in range(seq_num):
                bbox = Bbox([[x,y],[x+1,y+1]])
                symbol_plotter.plot_symbol(bbox, alignment, i, j)
                y += 1
            line_pos += 1
            if line_pos >= symbols_per_line:
                line_pos = 0
                x = 0
                y_start += seq_num + spacing
            else:
                x += 1
        
        axes.set_xlim(0, symbols_per_line)
        # y-axis starts from top
        axes.set_ylim(seq_num*line_count + spacing*(line_count-1), 0)