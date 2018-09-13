# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["AlignmentVisualizer", "AlignmentSimilarityVisualizer",
           "AlignmentSymbolVisualizer"]

import abc
import numpy as np
from ...visualize import Visualizer, colors
from .colorschemes import get_color_scheme

class AlignmentVisualizer(Visualizer, metaclass=abc.ABCMeta):
    """
    An `AlignmentVisualizer` displays an `Alignment` as figure.
    This is similar to the string representation of an `Alignment`,
    but with enhanced styling, symbol coloring and optional sequence
    labels and sequence position numbering.
    
    This is an abstract base class, since it does not define the
    coloring of the symbols or its backgrounds. Subclasses define these
    by overriding the `get_color()` method.

    EXPERIMENTAL: Future API changes are probable.

    Parameters
    ----------
    alignment : Alignment
        The alignment to be visualized.
        All sequences in the alignment must have an equal alphabet.
    """

    def __init__(self, alignment):
        super().__init__()
        self._alignment        = alignment

        self._show_numbers     = False
        self._number_size      = 50
        self._number_font      = None
        self._number_font_size = 16
        self._number_func      = [lambda x: x + 1] * len(alignment.sequences)

        self._show_labels      = False
        self._labels           = None
        self._label_size       = 150
        self._label_font       = None
        self._label_font_size  = 16

        self._box_size         = (20,30)
        self._symbols_per_line = 50
        self._symbol_font      = None
        self._symbol_font_size = 16
        self._color_symbols    = False

        self._spacing          = 30
        self._margin           = 10
        

        # Check if all sequences share the same alphabet
        alphabet = alignment.sequences[0].get_alphabet()
        for seq in alignment.sequences:
            if seq.get_alphabet() != alphabet:
                raise ValueError("Alphabets of the sequences in the alignment "
                                 "are not equal")
    
    def add_location_numbers(self, size=50, font_size=16, font=None,
                             number_functions=None):
        """
        Add numbers to the right side of the figure, that display the
        respective sequence location of the last sequence symbol in the
        line.

        Parameters
        ----------
        size : float, optional
            The size of the number column in x-direction (pixels).
            This value is a determining factor for the width
            of the figure.
            (Default: 50)
        font_size : float, optional
            Font size of the numbers.
            (Default: 16)
        font : FontProperties, optional
            `matplotlib` `FontProperties` for customization of the
            font used by the numbers.
        number_functions : iterable object of function, optional
            A list of functions, where each function alters the location
            number for each sequence.
            Must have the same length as the number of sequences in the
            alignment.
            Each function converts a sequence index to a location
            number.
            When the list element is `None`, the default location
            numbers are used for the respective sequence.
            By default the location number is the sequence index + 1
            (Sequence index starts at 0, location starts at 1).
        """
        self._show_numbers     = True
        self._number_size      = size
        self._number_font      = font
        self._number_font_size = font_size
        if number_functions is not None:
            for i, func in enumerate(number_functions):
                if func is not None:
                    self._number_func[i] = func
    
    def add_labels(self, labels, size=150, font_size=16, font=None):
        """
        Add sequence labels to the left side of the figure.

        Parameters
        ----------
        labels : iterable object of str
            The sequence labels.
            Must be the same size and order as the `sequences`
            attribute in the `Alignment`.
        size : float, optional
            The size of the label column in x-direction (pixels).
            This value is a determining factor for the width
            of the figure.
            (Default: 150)
        font_size : float, optional
            Font size of the labels.
            (Default: 16)
        font : FontProperties, optional
            `matplotlib` `FontProperties` for customization of the
            font used by the labels.
        """
        self._show_labels      = True
        self._labels           = labels
        self._label_size       = size
        self._label_font       = font
        self._label_font_size  = font_size
    
    def set_alignment_properties(self, box_size=(20,30), symbols_per_line=50,
                                 font_size=16, font=None, color_symbols=False):
        """
        Set visual properties of the alignment to be visualized.

        Parameters
        ----------
        box_size : tuple of (float), length=2
            An (x,y) tuple defining the size of the boxes behind the
            symbols. This value is a determining factor for the size
            of the figure.
            (Default: (20,30))
        symbols_per_line : int, optional
            The amount of sequence symbols displayed per line.
            This value is a determining factor for the width
            of the figure.
            (Default: 50)
        font_size : float, optional
            Font size of the sequence symbols.
            (Default: 16)
        font : FontProperties, optional
            `matplotlib` `FontProperties` for customization of the
            font used by the sequence symbols.
        color_symbols : bool, optional
            If true, the symbols themselves are colored.
            If false, the symbols are black, and the boxes behind the
            symbols are colored.
            (Default: False)
        """
        self._box_size         = box_size
        self._symbols_per_line = symbols_per_line
        self._symbol_font      = font
        self._symbol_font_size = font_size
        self._color_symbols    = color_symbols
    
    def set_spacing(self, spacing):
        """
        Set the spacing between the line batches.

        Parameters
        ----------
        spacing : float
            The spacing between the line batches.
            This value is a determining factor for the height of the
            figure.
        """
        self._spacing = spacing
    
    def set_margin(self, margin):
        """
        Set the margin of the figure.

        Parameters
        ----------
        margin : float
            The margin of the figure.
            This value is a determining factor for the size of the
            figure.
        """
        self._margin = margin

    @abc.abstractmethod
    def get_color(self, alignment, pos_i, seq_i):
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
        pos_i : int
            The position index in the trace.
        seq_i : int
            The sequence index in the trace.
        """
        pass

    def generate(self):
        from matplotlib.patches import Rectangle
        from matplotlib.text import Text

        fig_size_x = self._box_size[0] * self._symbols_per_line
        if self._show_labels:
            fig_size_x += self._label_size
        if self._show_numbers:
            fig_size_x += self._number_size
        fig_size_x += 2 * self._margin
        
        seq_num = self._alignment.trace.shape[1]
        seq_len = self._alignment.trace.shape[0]
        line_count = seq_len // self._symbols_per_line
        # Only extend line count by 1 if there is a remainder
        # (remaining symbols)
        if seq_len % self._symbols_per_line != 0:
            line_count += 1
        fig_size_y = line_count * self._box_size[1] * seq_num
        fig_size_y += (line_count-1) * self._spacing
        fig_size_y += 2 * self._margin

        fig = self.create_figure(size=(fig_size_x, fig_size_y))

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
                    number = self._number_func[j](seq_index)
                    text = Text(fig_size_x - self._margin, y, str(number),
                                color="black", ha="right", va="center",
                                size=self._number_font_size, figure=fig,
                                fontproperties=self._number_font)
                    fig.texts.append(text)
                    y -= self._box_size[1]
                y -= self._spacing

        ### Draw symbols in boxes ###
        x_start = self._label_size if self._labels is not None else 0
        x_start += self._margin
        y_start = fig_size_y - self._box_size[1]
        y_start -= self._margin
        x = x_start
        line_pos = 0
        for i in range(seq_len):
            y = y_start
            for j in range(seq_num):
                if self._alignment.trace[i,j] != -1:
                    symbol = self._alignment.sequences[j] \
                            [self._alignment.trace[i,j]]
                else:
                    symbol = "-"
                color = self.get_color(self._alignment, i, j)
                box = Rectangle((x,y), self._box_size[0]-1,self._box_size[1]-1)
                text = Text(x + self._box_size[0]/2, y + self._box_size[1]/2,
                            symbol, color="black", ha="center", va="center",
                            size=self._symbol_font_size, figure=fig,
                            fontproperties=self._symbol_font)
                if self._color_symbols:
                    box.set_color("None")
                    text.set_color(color)
                else:
                    box.set_color(color)
                fig.patches.append(box)
                fig.texts.append(text)
                y -= self._box_size[1]
            line_pos += 1
            if line_pos >= self._symbols_per_line:
                line_pos = 0
                x = x_start
                y_start -= seq_num * self._box_size[1] + self._spacing
            else:
                x += self._box_size[0]

        return fig
    
    def _get_last_real_index(self, alignment, i, j):
        index_found = False
        while not index_found:
            if i == 0:
                index = 0
                index_found = True
            else:
                index = alignment.trace[i,j]
                if index != -1:
                    index_found = True
            i -= 1
        return index


class AlignmentSimilarityVisualizer(AlignmentVisualizer):
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
    As the normalization is conducted only with respect to *a*,
    the *normalized similarity* is not commutative.

    The *average normalized similarity* of a symbol *a* is
    determined by averaging the normalized similarity over each
    symbol *b*\ :sub:`i` in the same alignment column.

    .. math:: S_{norm,av}(a) = \frac{1}{n-1} \left[\left(\sum\limits_{i=1}^n S_{norm}(a,b_i)\right) - S_{norm}(a,a)\right]

    The normalized similarity of *a* to itself is subtracted,
    because *a* does also occur in *b*\ :sub:`i`.
    """

    def __init__(self, alignment, matrix=None):
        from matplotlib import cm
        super().__init__(alignment)
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

    def get_color(self, alignment, pos_i, seq_i):
        # Calculate average normalize similarity 
        index1 = alignment.trace[pos_i, seq_i]
        if index1 == -1:
            similarity = 0
        else:
            code1 = alignment.sequences[seq_i].code[index1]
            similarities = np.zeros(alignment.trace.shape[1])
            for i in range(alignment.trace.shape[1]):
                index2 = alignment.trace[pos_i, i]
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


class AlignmentSymbolVisualizer(AlignmentVisualizer):
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

    def __init__(self, alignment):
        super().__init__(alignment)
        alphabet = alignment.sequences[0].get_alphabet()
        self._colors = get_color_scheme("rainbow", alphabet)
    
    def set_color_scheme(self, scheme):
        """
        Set the color scheme used for the alignemnt.

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
            alphabet = self._alignment.sequences[0].get_alphabet()
            self._colors = get_color_scheme(scheme, alphabet)
        else:
            self._colors = scheme
    
    def get_color(self, alignment, pos_i, seq_i):
        index = alignment.trace[pos_i, seq_i]
        if index == -1:
            # Gaps are white
            return (1, 1, 1)
        code = alignment.sequences[seq_i].code[index]
        return self._colors[code]