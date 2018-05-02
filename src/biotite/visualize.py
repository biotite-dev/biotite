# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["Visualizer"]

import abc

class Visualizer(metaclass=abc.ABCMeta):
    
    def __init__(self):
        pass
    
    def create_figure(self, size):
        return plt.figure(figsize=size)

    @abc.abstractmethod
    def generate(self):
        pass