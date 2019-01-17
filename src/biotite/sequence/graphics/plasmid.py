# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["plot_plasmid_map"]

import copy
import abc
import numpy as np
from ...visualize import colors, AdaptiveFancyArrow
from ..annotation import Annotation, Feature, Location


def plot_plasmid_map(axes, annotation, loc_range=None):
    axes.set_xlim(0, 2*np.pi)
    axes.set_ylim(0, 1)
    axes.yaxis.set_tick_params(
        left=False, right=False, labelsize=number_size
    )