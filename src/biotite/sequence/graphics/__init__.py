# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for visualization of sequence related objects.

The visualizations make use of *Matplotlib* for plotting.
Therefore, each plotting function requires an :class:`Axes` object,
where the visualization should be performed.

The resulting plots are customizable:
Labels, location numbers, etc. are usually placed into the axes tick
labels, making tham accessible for modification via the usual
*Matplotlib* API.
Some aspects of the plotting itself are also customizable: For example
the :func:`plot_alignment()` function uses a interchangable
:class:`SymbolPlotter`, that knows how to draw the symbols of an
alignment.
Similarily, the appearance of sequence features in the function
:func:`plot_feature_map()` is customized via :class:`FeaturePlotter`
objects.
"""

__name__ = "biotite.sequence.graphics"
__author__ = "Patrick Kunzmann"

from .alignment import *
from .colorschemes import *
from .dendrogram import *
from .features import *
from .logo import *
from .plasmid import *
