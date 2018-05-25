# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["Visualizer", "colors"]

import abc
from collections import OrderedDict

class Visualizer(metaclass=abc.ABCMeta):
    """
    Base class for all classes that are used for `matplotlib` based
    visualization of bioinformatics objects
    (sequences, alignments, etc.).
    
    This class merely provides the `create_figure()` method, which is
    used by child classes to obtain an empty `matplotlib` `Figure` with
    defined size (specified in pixels).

    Child classes must override the `generate()` method.

    EXPERIMENTAL: Future API changes are probable.
    """

    def __init__(self):
        pass
    
    def create_figure(self, size, dpi=100):
        """
        Obtain an empty `matplotlib` `Figure` with defined size.

        PROTECTED: Do not call from outside.
        
        Parameters
        ----------
        size : tuple, length=2
            The size of the figure (x,y) in pixels (rather than inch).
        dpi : int, optional
            A custom DPI of the figure. Usually this does not have any
            effect on the look of the resulting figure.
        
        Returns
        -------
        figure : Figure
            An empty figure.
        """
        import matplotlib.pyplot as plt
        return plt.figure(figsize=(size[0]/dpi, size[1]/dpi))

    @abc.abstractmethod
    def generate(self):
        """
        Generate the visualization.
        
        Returns
        -------
        figure : Figure
            The generated visualization.
            When saving the figure as vector graphics using `savefig()`,
            use ``bbox_inches="tight"``, since otherwise the figure
            might be truncated. 
        """
        pass


# Biotite themed colors
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