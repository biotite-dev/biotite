# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["Visualizer", "colors"]

import abc
from collections import OrderedDict

class Visualizer(metaclass=abc.ABCMeta):
    
    def __init__(self):
        pass
    
    def create_figure(self, size, dpi=100):
        import matplotlib.pyplot as plt
        return plt.figure(figsize=(size[0]/dpi, size[1]/dpi))

    @abc.abstractmethod
    def generate(self):
        pass


colors = OrderedDict([
    ("brightorange" , "#ffb569ff"),
    ("lightorange"  , "#ff982dff"),
    ("orange"       , "#ff8405ff"),
    ("dimorange"    , "#dc7000ff"),
    ("darkorange"   , "#b45c00ff"),
    ("brightgreen"  , "#98e97fff"),
    ("lightgreen"   , "#6fe04cff"),
    ("green"        , "#52da2aff"),
    ("dimgreen"     , "#45bc20ff"),
    ("darkgreen"    , "#389a1aff"),
])