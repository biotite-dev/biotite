# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.graphics"
__author__ = "Patrick Kunzmann"
__all__ = ["SymbolPlotter", "LetterPlotter", "LetterSimilarityPlotter",
           "LetterTypePlotter",
           "plot_alignment", "plot_alignment_similarity_based",
           "plot_alignment_type_based"]

import abc
import numpy as np
from ...visualize import colors
from .colorschemes import get_color_scheme


class SymbolPlotter(metaclass=abc.ABCMeta):
    """
    Subclasses of this abstract base class define how symbols in
    an alignment are drawn onto an :class:`Axes` object.

    Subclasses must override the :func:`plot_symbol()` method.

    Parameters
    ----------
    axes : Axes
        A *Matplotlib* axes, that is used as plotting area.
    """

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
    """
    This abstract :class:`SymbolPlotter` is the most widely used one.
    Symbols are visualized as character on a colored background box or
    as colored character, if `color_symbols` is set to true.

    Subclasses must override the :class:`get_color()` method.

    Parameters
    ----------
    axes : Axes
        A *Matplotlib* axes, that is used as plotting area.
    color_symbols : bool, optional
        If true, the symbols themselves are colored.
        If false, the symbols are black, and the boxes behind the
        symbols are colored.
    font_size : float, optional
        Font size of the sequence symbols.
    font_param : dict, optional
        Additional parameters that is given to the
        :class:`matplotlib.Text` instance of each symbol.
    """

    def __init__(self, axes, color_symbols=False,
                 font_size=None, font_param=None):
        super().__init__(axes)
        self._color_symbols = color_symbols
        self._font_size = font_size
        self._font_param = font_param if font_param is not None else {}

    def plot_symbol(self, bbox, alignment, column_i, seq_i):
        from matplotlib.patches import Rectangle

        trace = alignment.trace
        if trace[column_i, seq_i] != -1:
            symbol = alignment.sequences[seq_i][trace[column_i, seq_i]]
        else:
            symbol = "-"
        color = self.get_color(alignment, column_i, seq_i)

        box = Rectangle(bbox.p0, bbox.width, bbox.height)
        self.axes.add_patch(box)
        text = self.axes.text(
            bbox.x0 + bbox.width/2, bbox.y0 + bbox.height/2,
            symbol, color="black", ha="center", va="center",
            size=self._font_size, **self._font_param)
        text.set_clip_on(True)

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

        Returns
        -------
        color : object
            A *Matplotlib* compatible color used for the background
            or the symbol itself at the specifed position
        """
        pass


class LetterSimilarityPlotter(LetterPlotter):
    r"""
    This :class:`SymbolPlotter` colors the symbols based on the
    similarity with the other symbols in the same column.

    The color intensity (or colormap value, respectively) of a symbol
    scales with similarity of the respective symbol to the other symbols
    in the same alignment column.

    Parameters
    ----------
    axes : Axes
        A *Matplotlib* axes, that is used as plotting area.
    matrix : SubstitutionMatrix, optional
        The substitution matrix to use the similarity scores from.
        By default the normalized similarity is 1 for identity and 0
        for non-identity.
    color_symbols : bool, optional
        If true, the symbols themselves are colored.
        If false, the symbols are black, and the boxes behind the
        symbols are colored.
    font_size : float, optional
        Font size of the sequence symbols.
    font_param : dict, optional
        Additional parameters that is given to the
        :class:`matplotlib.Text` instance of each symbol.

    Notes
    -----
    For determination of the color, this a measure called
    *average normalized similarity* is used.

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

    def __init__(self, axes, matrix=None, color_symbols=False,
                 font_size=None, font_param=None):

        super().__init__(axes, color_symbols, font_size, font_param)
        if matrix is not None:
            self._matrix = matrix.score_matrix()
        else:
            self._matrix = None
        # Default colormap
        self._cmap = self._generate_colormap(colors["dimgreen"],
                                             self._color_symbols)

    def set_color(self, color=None, cmap=None):
        """
        Set the alignemnt colors used for plotting.

        This function takes either a color or a colormap.

        Parameters
        ----------
        color : tuple or str, optional
            A *Matplotlib* compatible color.
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
    This `SymbolPloter` colors each symbol based on the general
    color of that symbol defined by a color scheme.

    EXPERIMENTAL: Future API changes are probable.

    Parameters
    ----------
    axes : Axes
        A *Matplotlib* axes, that is used as plotting area.
    alphabet : Alphabet
        The alphabet of the alignment(s) to be plotted
    color_scheme : str or list of (tuple or str), optional
        Either a valid color scheme name
        (e.g. ``"rainbow"``, ``"clustalx"``, ``blossom``, etc.)
        or a list of *Matplotlib* compatible colors.
        The list length must be at least as long as the
        length of the alphabet used by the sequences.
    color_symbols : bool, optional
        If true, the symbols themselves are colored.
        If false, the symbols are black, and the boxes behind the
        symbols are colored.
    font_size : float, optional
        Font size of the sequence symbols.
    font_param : dict, optional
        Additional parameters that is given to the
        :class:`matplotlib.Text` instance of each symbol.
    """

    def __init__(self, axes, alphabet, color_scheme=None, color_symbols=False,
                 font_size=None, font_param=None):
        super().__init__(axes, color_symbols, font_size, font_param)

        if color_scheme is None:
            self._colors = get_color_scheme("rainbow", alphabet)
        elif isinstance(color_scheme, str):
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
                   show_numbers=False, number_size=None, number_functions=None,
                   labels=None, label_size=None,
                   show_line_position=False,
                   spacing=1, symbol_spacing=None):
    """
    Plot a pairwise or multiple sequence alignment.

    The output is similar to a string representation of an
    :class:`Alignment`, but with enhanced styling, symbol coloring and
    optional sequence labels and sequence position numbering.
    How each symbol of the alignment is drawn is determined by the
    given :class:`SymbolPlotter` object.

    Parameters
    ----------
    axes : Axes
        A *Matplotlib* axes, that is used as plotting area.
    alignment : Alignment
        The pairwise or multiple sequence alignment to be plotted.
    symbol_plotter : SymbolPlotter
        Defines how the symbols in the alignment are drawn.
    symbols_per_line : int, optional
        The amount of alignment columns that are diplayed per line.
    show_numbers : bool, optional
        If true, the sequence position of the symbols in the last
        alignment column of a line is shown on the right side of the
        plot.
        If the last symbol is a gap, the position of the last actual
        symbol before this gap is taken.
        If the first symbol did not occur up to this point,
        no number is shown for this line.
        By default the first symbol of a sequence has the position 1,
        but this behavior can be changed using the `number_functions`
        parameter.
    number_size : float, optional
        The font size of the position numbers
    number_functions : list of [(None or Callable(int -> int)], optional
        By default the position of the first symbol in a sequence is 1,
        i.e. the sequence position is the sequence index incremented by
        1.
        The behavior can be changed with this parameter:
        If supplied, the length of the list must match the number of
        sequences in the alignment.
        Every entry is a function that maps a sequence index (*int*) to
        a sequence position (*int*) for the respective sequence.
        A ``None`` entry means, that the default numbering is applied
        for the sequence.
    labels : list of str, optional
        The sequence labels.
        Must be the same size and order as the sequences in the
        alignment.
    label_size : float, optional
        Font size of the labels
    show_line_position : bool, optional
        If true the position within a line is plotted below the
        alignment.
    spacing : float, optional
        The spacing between the alignment lines. 1.0 means that the size
        is equal to the size of a symbol box.
    symbol_spacing : int, optional
        А space is placed between each number of elements desired
        by variable.

    See also
    --------
    plot_alignment_similarity_based
    plot_alignment_type_based

    Notes
    -----
    The labels are placed on the Y-axis of the `axes` and the numbers
    are placed on the Y-axis of the `axes` twin.
    The position within a line is placed on the X-axis of the `axes`.
    Further modification of these can be performed using the usual
    *Matplotlib* API.
    """
    from matplotlib.transforms import Bbox

    if number_functions is None:
        number_functions = [lambda x: x + 1] * len(alignment.sequences)
    else:
        if len(number_functions) != len(alignment.sequences):
            raise ValueError(
                f"The amount of renumbering functions is "
                f"{len(number_functions)} but the amount if sequences in the "
                f"alignment is {len(alignment.sequences)}"
            )
        for i, func in enumerate(number_functions):
            if func is None:
                number_functions[i] = (lambda x: x + 1)

    seq_num = alignment.trace.shape[1]
    seq_len = alignment.trace.shape[0]
    line_count = seq_len // symbols_per_line
    # Only extend line count by 1 if there is a remainder
    # (remaining symbols)
    if seq_len % symbols_per_line != 0:
        line_count += 1

    if symbol_spacing:
        spacing_ratio = symbols_per_line / symbol_spacing
        if spacing_ratio % 1 != 0:
            raise ValueError("symbols_per_line not multiple of symbol_spacing")
        # Initializing symbols_to_print to print symbols_per_line
        # symbols on one line + spacing between symbols
        symbols_to_print = int(spacing_ratio) + symbols_per_line - 1
    else:
        symbols_to_print = symbols_per_line

    ### Draw symbols ###
    x = 0
    y = 0
    y_start = 0
    line_pos = 0
    for i in range(seq_len):
        y = y_start
        for j in range(seq_num):
            bbox = Bbox([[x, y], [x+1, y+1]])
            symbol_plotter.plot_symbol(bbox, alignment, i, j)
            y += 1
        line_pos += 1
        if line_pos >= symbols_to_print:
            line_pos = 0
            x = 0
            y_start += seq_num + spacing
        else:
            x += 1
            if (symbol_spacing
               and (i + 1) % symbol_spacing == 0):
                line_pos += 1
                x += 1

    ### Draw labels ###
    ticks = []
    tick_labels = []
    if labels is not None:
        # Labels at center height of each line of symbols -> 0.5
        y = 0.5
        for i in range(line_count):
            for j in range(seq_num):
                ticks.append(y)
                tick_labels.append(labels[j])
                y += 1
            y += spacing
    axes.set_yticks(ticks)
    axes.set_yticklabels(tick_labels)

    ### Draw numbers  ###
    # Create twin to allow different tick labels on right side
    number_axes = axes.twinx()
    ticks = []
    tick_labels = []
    if show_numbers:
        # Numbers at center height of each line of symbols -> 0.5
        y = 0.5
        for i in range(line_count):
            for j in range(seq_num):
                if i == line_count-1:
                    # Last line -> get number of last column in trace
                    trace_pos = len(alignment.trace) - 1
                else:
                    trace_pos = (i+1) * symbols_per_line - 1
                seq_index = _get_last_valid_index(
                    alignment, trace_pos, j
                )
                # if -1 -> terminal gap
                # -> skip number for this sequence in this line
                if seq_index != -1:
                    # Convert sequence index to position
                    # (default index + 1)
                    number = number_functions[j](seq_index)
                    ticks.append(y)
                    tick_labels.append(str(number))
                y += 1
            y += spacing
    number_axes.set_yticks(ticks)
    number_axes.set_yticklabels(tick_labels)

    axes.set_xlim(0, symbols_to_print)
    # Y-axis starts from top
    lim = seq_num*line_count + spacing*(line_count-1)
    axes.set_ylim(lim, 0)
    number_axes.set_ylim(lim, 0)
    axes.set_frame_on(False)
    number_axes.set_frame_on(False)
    # Remove ticks and set label and number size
    axes.yaxis.set_tick_params(
        left=False, right=False, labelsize=label_size
    )
    number_axes.yaxis.set_tick_params(
        left=False, right=False, labelsize=number_size
    )

    if show_line_position:
        axes.xaxis.set_tick_params(
            top=False, bottom=True, labeltop=False, labelbottom=True
        )
    else:
        axes.xaxis.set_tick_params(
            top=False, bottom=False, labeltop=False, labelbottom=False
        )


def plot_alignment_similarity_based(axes, alignment, symbols_per_line=50,
                                    show_numbers=False, number_size=None,
                                    number_functions=None,
                                    labels=None, label_size=None,
                                    show_line_position=False,
                                    spacing=1,
                                    color=None, cmap=None, matrix=None,
                                    color_symbols=False, symbol_spacing=None,
                                    symbol_size=None, symbol_param=None):
    r"""
    Plot a pairwise or multiple sequence alignment highlighting
    the similarity per alignment column.

    This function works like :func:`plot_alignment()` with a
    :class:`SymbolPlotter`, that colors the symbols based on the
    similarity with the other symbols in the same column.
    The color intensity (or colormap value, respectively) of a symbol
    scales with similarity of the respective symbol to the other symbols
    in the same alignment column.

    Parameters
    ----------
    axes : Axes
        A *Matplotlib* axes, that is used as plotting area.
    alignment : Alignment
        The pairwise or multiple sequence alignment to be plotted.
        The alphabet of each sequence in the alignment must be the same.
    symbol_plotter : SymbolPlotter
        Defines how the symbols in the alignment are drawn.
    symbols_per_line : int, optional
        The amount of alignment columns that are diplayed per line.
    show_numbers : bool, optional
        If true, the sequence position of the symbols in the last
        alignment column of a line is shown on the right side of the
        plot.
        If the last symbol is a gap, the position of the last actual
        symbol before this gap is taken.
        If the first symbol did not occur up to this point,
        no number is shown for this line.
        By default the first symbol of a sequence has the position 1,
        but this behavior can be changed using the `number_functions`
        parameter.
    number_size : float, optional
        The font size of the position numbers
    number_functions : list of [(None or Callable(int -> int)], optional
        By default the position of the first symbol in a sequence is 1,
        i.e. the sequence position is the sequence index incremented by
        1.
        The behavior can be changed with this parameter:
        If supplied, the length of the list must match the number of
        sequences in the alignment.
        Every entry is a function that maps a sequence index (*int*) to
        a sequence position (*int*) for the respective sequence.
        A `None` entry means, that the default numbering is applied
        for the sequence.
    labels : list of str, optional
        The sequence labels.
        Must be the same size and order as the sequences in the
        alignment.
    label_size : float, optional
        Font size of the labels
    show_line_position : bool, optional
        If true the position within a line is plotted below the
        alignment.
    spacing : float, optional
        The spacing between the alignment lines. 1.0 means that the size
        is equal to the size of a symbol box.
    color : tuple or str, optional
        A *Matplotlib* compatible color.
        If this parameter is given, the box color in an interpolated
        value between white and the given color,
        or, if `color_symbols` is set to true, between the given color
        and black.
        The interpolation percentage is given by the average normalized
        similarity.
    cmap : Colormap or str, optional
        The boxes (or symbols, if `color_symbols` is set) are
        colored based on the normalized similarity value on the
        given *Matplotlib* Colormap.
    matrix : SubstitutionMatrix
        The substitution matrix used to determine the similarity
        of two symbols. By default an identity matrix is used, i.e.
        only match and mismatch is distinguished.
    color_symbols : bool, optional
        If true, the symbols themselves are colored.
        If false, the symbols are black, and the boxes behind the
        symbols are colored.
    symbol_size : float, optional
        Font size of the sequence symbols.
    symbol_param : dict
        Additional parameters that is given to the
        :class:`matplotlib.Text` instance of each symbol.
    symbol_spacing : int, optional
        А space is placed between each number of elements desired
        by variable.

    See also
    --------
    plot_alignment
    LetterSimilarityPlotter

    Notes
    -----
    For determination of the color, a measure called
    *average normalized similarity* is used.

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
    symbol_plotter = LetterSimilarityPlotter(
        axes, matrix=matrix, font_size=symbol_size, font_param=symbol_param,
        color_symbols=color_symbols
    )
    if color is not None or cmap is not None:
        symbol_plotter.set_color(color=color, cmap=cmap)
    plot_alignment(
        axes=axes, alignment=alignment, symbol_plotter=symbol_plotter,
        symbols_per_line=symbols_per_line,
        show_numbers=show_numbers, number_size=number_size,
        number_functions=number_functions,
        labels=labels, label_size=label_size,
        show_line_position=show_line_position,
        spacing=spacing, symbol_spacing=symbol_spacing
    )


def plot_alignment_type_based(axes, alignment, symbols_per_line=50,
                              show_numbers=False, number_size=None,
                              number_functions=None,
                              labels=None, label_size=None,
                              show_line_position=False,
                              spacing=1,
                              color_scheme=None, color_symbols=False,
                              symbol_size=None, symbol_param=None,
                              symbol_spacing=None):
    """
    Plot a pairwise or multiple sequence alignment coloring each symbol
    based on the symbol type.

    This function works like :func:`plot_alignment()` with a
    :class:`SymbolPlotter`, that colors the symbols based on a color
    scheme.
    The color intensity (or colormap value, respectively) of a symbol
    scales with similarity of the respective symbol to the other symbols
    in the same alignment column.

    Parameters
    ----------
    axes : Axes
        A *Matplotlib* axes, that is used as plotting area.
    alignment : Alignment
        The pairwise or multiple sequence alignment to be plotted.
        The alphabet of each sequence in the alignment must be the same.
    symbol_plotter : SymbolPlotter
        Defines how the symbols in the alignment are drawn.
    symbols_per_line : int, optional
        The amount of alignment columns that are diplayed per line.
    show_numbers : bool, optional
        If true, the sequence position of the symbols in the last
        alignment column of a line is shown on the right side of the
        plot.
        If the last symbol is a gap, the position of the last actual
        symbol before this gap is taken.
        If the first symbol did not occur up to this point,
        no number is shown for this line.
        By default the first symbol of a sequence has the position 1,
        but this behavior can be changed using the `number_functions`
        parameter.
    number_size : float, optional
        The font size of the position numbers
    number_functions : list of [(None or Callable(int -> int)], optional
        By default the position of the first symbol in a sequence is 1,
        i.e. the sequence position is the sequence index incremented by
        1.
        The behavior can be changed with this parameter:
        If supplied, the length of the list must match the number of
        sequences in the alignment.
        Every entry is a function that maps a sequence index (*int*) to
        a sequence position (*int*) for the respective sequence.
        A `None` entry means, that the default numbering is applied
        for the sequence.
    labels : list of str, optional
        The sequence labels.
        Must be the same size and order as the sequences in the
        alignment.
    label_size : float, optional
        Font size of the labels
    show_line_position : bool, optional
        If true the position within a line is plotted below the
        alignment.
    spacing : float, optional
        The spacing between the alignment lines. 1.0 means that the size
        is equal to the size of a symbol box.
    color_scheme : str or list of (tuple or str), optional
        Either a valid color scheme name
        (e.g. ``"rainbow"``, ``"clustalx"``, ``blossom``, etc.)
        or a list of *Matplotlib* compatible colors.
        The list length must be at least as long as the
        length of the alphabet used by the sequences.
    color_symbols : bool, optional
        If true, the symbols themselves are colored.
        If false, the symbols are black, and the boxes behind the
        symbols are colored.
    symbol_size : float, optional
        Font size of the sequence symbols.
    symbol_param : dict
        Additional parameters that is given to the
        :class:`matplotlib.Text` instance of each symbol.
    symbol_spacing : int, optional
        А space is placed between each number of elements desired
        by variable.

    See also
    --------
    plot_alignment
    LetterTypePlotter
    """
    alphabet = alignment.sequences[0].get_alphabet()
    symbol_plotter = LetterTypePlotter(
        axes, alphabet, font_size=symbol_size, font_param=symbol_param,
        color_symbols=color_symbols, color_scheme=color_scheme
    )
    plot_alignment(
        axes=axes, alignment=alignment, symbol_plotter=symbol_plotter,
        symbols_per_line=symbols_per_line,
        show_numbers=show_numbers, number_size=number_size,
        number_functions=number_functions,
        labels=labels, label_size=label_size,
        show_line_position=show_line_position,
        spacing=spacing, symbol_spacing=symbol_spacing
    )


def _get_last_valid_index(alignment, column_i, seq_i):
    """
    Find the last trace value that belongs to a valid sequence index
    (no gap -> no -1) up to the specified column.
    """
    index_found = False
    while not index_found:
        if column_i == -1:
            # Iterated from column_i back to beyond the beginning
            # and no index has been found
            # -> Terminal gap
            # -> First symbol of sequence has not occured yet
            # -> return -1
            index = -1
            index_found = True
        else:
            index = alignment.trace[column_i, seq_i]
            if index != -1:
                index_found = True
        column_i -= 1
    return index
