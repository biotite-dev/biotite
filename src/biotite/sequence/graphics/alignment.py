# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["AlignmentVisualizer"]

import abc
from ...visualize import Visualizer

class AlignmentVisualizer(Visualizer, metaclass=abc.ABCMeta):
    
    def __init__(self, alignment,
                 chars_per_line=50, padding=20, border_size=10,
                 box_size=(20,30),
                 labels=None, label_size=100,
                 show_numbers=True, number_size=30,
                 label_font=None, label_font_size=16,
                 symbol_font=None, symbol_font_size=16, color_symbols=False):
        self._alignment        = alignment
        self._chars_per_line  = chars_per_line
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

    #@abc.abstractmethod
    def get_color(self, alignment, pos):
        i = pos[0]
        j = pos[1]
        if alignment.trace[i,j] != -1:
            symbol = alignment.sequences[j][alignment.trace[i,j]]
            if   symbol == "A":
                return "green"
            elif symbol == "T":
                return "red"
            elif symbol == "G":
                return "yellow"
            elif symbol == "C":
                return "blue"
            else:
                return "white"
        else:
            return "white"

    def generate(self):
        from matplotlib.patches import Rectangle
        from matplotlib.text import Text

        fig_size_x = self._box_size[0] * self._chars_per_line
        if self._labels is not None:
            fig_size_x += self._label_size
        if self._show_numbers is not None:
            fig_size_x += self._number_size
        fig_size_x += 2 * self._border_size
        
        seq_num = self._alignment.trace.shape[1]
        seq_len = self._alignment.trace.shape[0]
        line_count = (seq_len // self._chars_per_line) + 1
        fig_size_y = line_count * self._box_size[1] * seq_num
        fig_size_y += (line_count-1) * self._padding
        fig_size_y += 2 * self._border_size

        fig = self.create_figure(size=(fig_size_x, fig_size_y))

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
                color = self.get_color(self._alignment, (i, j))
                box = Rectangle((x,y), self._box_size[0]-1, self._box_size[1]-1)
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
            if line_pos >= self._chars_per_line:
                line_pos = 0
                x = x_start
                y_start -= seq_num * self._box_size[1] + self._padding
            else:
                x += self._box_size[0]

        #fig.patches.append(Rectangle((10, 20), 30, 40))
        return fig