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
        import matplotlib.pyplot as plt
        dpi = 100
        return plt.figure(figsize=(size[0]/dpi, size[1]/dpi))

    @abc.abstractmethod
    def generate(self):
        pass