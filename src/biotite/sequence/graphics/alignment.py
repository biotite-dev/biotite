# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["AlignmentVisualizer", "AlignmentSimilarityVisualizer",
           "AlignmentSymbolVisualizer"]

import abc
import numpy as np
from ...visualize import Visualizer
from .colorschemes import color_schemes

class AlignmentVisualizer(Visualizer, metaclass=abc.ABCMeta):
    
    def __init__(self, alignment):
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
        self._border_size      = 10
        

        # Check if all sequences share the same alphabet
        alphabet = alignment.sequences[0].get_alphabet()
        for seq in alignment.sequences:
            if seq.get_alphabet() != alphabet:
                raise ValueError("Alphabets of the sequences in the alignment "
                                 "are not equal")
    
    def add_location_numbers(self, size=50, font_size=16, font=None,
                             number_functions=None):
        self._show_numbers     = True
        self._number_size      = size
        self._number_font      = font
        self._number_font_size = font_size
        if number_functions is not None:
            for i, func in enumerate(number_functions):
                if func is not None:
                    self._number_func[i] = func
    
    def add_labels(self, labels, size=150, font_size=16, font=None):
        self._show_labels      = True
        self._labels           = labels
        self._label_size       = size
        self._label_font       = font
        self._label_font_size  = font_size
    
    def set_alignment_properties(self, box_size=(20,30), symbols_per_line=50,
                                 font_size=16, font=None, color_symbols=False):
        self._box_size         = box_size
        self._symbols_per_line = symbols_per_line
        self._symbol_font      = font
        self._symbol_font_size = font_size
        self._color_symbols    = color_symbols
    
    def set_spacing(self, spacing):
        self._spacing = spacing
    
    def set_border_size(self, border_size):
        self._border_size = border_size

    @abc.abstractmethod
    def get_color(self, alignment, pos_i, seq_i):
        pass

    def generate(self):
        from matplotlib.patches import Rectangle
        from matplotlib.text import Text

        fig_size_x = self._box_size[0] * self._symbols_per_line
        if self._labels is not None:
            fig_size_x += self._label_size
        if self._show_numbers is not None:
            fig_size_x += self._number_size
        fig_size_x += 2 * self._border_size
        
        seq_num = self._alignment.trace.shape[1]
        seq_len = self._alignment.trace.shape[0]
        line_count = seq_len // self._symbols_per_line
        # Only extend line count by 1 if there is a remainder
        # (remaining symbols)
        if seq_len % self._symbols_per_line != 0:
            line_count += 1
        fig_size_y = line_count * self._box_size[1] * seq_num
        fig_size_y += (line_count-1) * self._spacing
        fig_size_y += 2 * self._border_size

        fig = self.create_figure(size=(fig_size_x, fig_size_y))

        ### Draw labels ###
        if self._labels is not None:
            y = fig_size_y - self._border_size
            y -= self._box_size[1] / 2
            for i in range(line_count):
                for j in range(seq_num):
                    label = self._labels[j]
                    text = Text(self._border_size, y, label,
                                color="black", ha="left", va="center",
                                size=self._label_font_size, figure=fig,
                                fontproperties=self._label_font)
                    fig.texts.append(text)
                    y -= self._box_size[1]
                y -= self._spacing
        
        ### Draw numbers  ###
        if self._show_numbers:
            y = fig_size_y - self._border_size
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
                    text = Text(fig_size_x - self._border_size, y, str(number),
                                color="black", ha="right", va="center",
                                size=self._number_font_size, figure=fig,
                                fontproperties=self._number_font)
                    fig.texts.append(text)
                    y -= self._box_size[1]
                y -= self._spacing

        ### Draw symbols in boxes ###
        x_start = self._label_size if self._labels is not None else 0
        x_start += self._border_size
        y_start = fig_size_y - self._box_size[1]
        y_start -= self._border_size
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

    def __init__(self, alignment, matrix=None):
        from matplotlib import cm
        super().__init__(alignment)
        if matrix is not None:
            self._matrix = matrix.score_matrix()
        else:
            self._matrix = None 
        # Default colormap
        green = [89/255, 184/255, 76/255, 1]
        self._cmap = self._generate_colormap(green, self._color_symbols)
    
    def set_color(self, color=None, cmap=None):
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

    def __init__(self, alignment):
        super().__init__(alignment)
        alphabet = alignment.sequences[0].get_alphabet()
        self._colors = color_schemes[alphabet]["rainbow"]
    
    def set_color_scheme(self, colors):
        if isinstance(colors, str):
            alphabet = alignment.sequences[0].get_alphabet()
            self._colors = color_schemes[alphabet][colors]
        else:
            self._colors = colors
    
    def get_color(self, alignment, pos_i, seq_i):
        index = alignment.trace[pos_i, seq_i]
        if index == -1:
            # Gaps are white
            return (1, 1, 1)
        code = alignment.sequences[seq_i].code[index]
        return self._colors[code]