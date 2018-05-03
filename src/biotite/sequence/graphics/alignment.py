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
    
    def __init__(self, alignment,
                 symbols_per_line=50, padding=30, border_size=10,
                 box_size=(20,30),
                 labels=None, label_size=150,
                 show_numbers=True, number_size=50,
                 label_font=None, label_font_size=16,
                 symbol_font=None, symbol_font_size=16, color_symbols=False):
        self._alignment        = alignment
        self._symbols_per_line = symbols_per_line
        self._padding          = padding
        self._border_size      = border_size
        self._box_size         = box_size
        self._labels           = labels
        self._label_size       = label_size
        self._show_numbers     = show_numbers
        self._number_size      = number_size
        self._label_font       = label_font
        self._label_font_size  = label_font_size
        self._symbol_font      = symbol_font
        self._symbol_font_size = symbol_font_size
        self._color_symbols    = color_symbols

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
        line_count = (seq_len // self._symbols_per_line)
        # Only extend line count by 1 if there is a remainder
        # (remaining symbols)
        if seq_len % self._symbols_per_line != 0:
            line_count += 1
        fig_size_y = line_count * self._box_size[1] * seq_num
        fig_size_y += (line_count-1) * self._padding
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
                                size=self._label_font_size, figure=fig)
                    fig.texts.append(text)
                    y -= self._box_size[1]
                y -= self._padding
        
        ### Draw numbers  ###
        if self._show_numbers:
            y = fig_size_y - self._border_size
            y -= self._box_size[1] / 2
            for i in range(line_count-1):
                for j in range(seq_num):
                    number = self._get_seq_pos(
                        self._alignment, (i+1) * self._symbols_per_line -1, j
                    )
                    text = Text(fig_size_x - self._border_size, y, str(number),
                                color="black", ha="right", va="center",
                                size=self._label_font_size, figure=fig)
                    fig.texts.append(text)
                    y -= self._box_size[1]
                y -= self._padding

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
                            size=self._symbol_font_size, figure=fig)
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
                y_start -= seq_num * self._box_size[1] + self._padding
            else:
                x += self._box_size[0]

        #fig.patches.append(Rectangle((10, 20), 30, 40))
        return fig
    
    def _get_seq_pos(self, alignment, i, j):
        pos_found = False
        while not pos_found:
            if i == 0:
                pos = 0
                pos_found = True
            else:
                pos = alignment.trace[i,j]
                if pos != -1:
                    pos_found = True
            i -= 1
        return pos + 1


class AlignmentSimilarityVisualizer(AlignmentVisualizer):

    def __init__(self, alignment,
                 symbols_per_line=50, padding=30, border_size=10,
                 box_size=(20,30),
                 labels=None, label_size=150,
                 show_numbers=True, number_size=50,
                 label_font=None, label_font_size=16,
                 symbol_font=None, symbol_font_size=16, color_symbols=False,
                 color=None, cmap=None, matrix=None):
        from matplotlib import cm
        super().__init__(alignment, symbols_per_line, padding, border_size,
                        box_size, labels, label_size, show_numbers,
                        number_size, label_font, label_font_size, symbol_font,
                        symbol_font_size, color_symbols)
        if color is not None:
            self._cmap = self._generate_colormap(color, color_symbols)
        elif cmap is not None:
            if isinstance(cmap, str):
                self._cmap = cm.get_cmap(cmap)
            else:
                # cmap is a colormap
                self._cmap = cmap
        else:
            # Default colormap
            green = [89/255, 184/255, 76/255, 1]
            self._cmap = self._generate_colormap(green, color_symbols)
        if matrix is not None:
            self._matrix = matrix.score_matrix()
        else:
            self._matrix = None 
    
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

    def __init__(self, alignment,
                 symbols_per_line=50, padding=30, border_size=10,
                 box_size=(20,30),
                 labels=None, label_size=150,
                 show_numbers=True, number_size=50,
                 label_font=None, label_font_size=16,
                 symbol_font=None, symbol_font_size=16, color_symbols=False,
                 colors=None):
        super().__init__(alignment, symbols_per_line, padding, border_size,
                        box_size, labels, label_size, show_numbers,
                        number_size, label_font, label_font_size, symbol_font,
                        symbol_font_size, color_symbols)
        if colors is None:
            alphabet = alignment.sequences[0].get_alphabet()
            self._colors = color_schemes[alphabet]["rainbow"]
        else:
            self._colors = colors
    
    def get_color(self, alignment, pos_i, seq_i):
        index = alignment.trace[pos_i, seq_i]
        if index == -1:
            return (1, 1, 1)
        code = alignment.sequences[seq_i].code[index]
        return self._colors[code]