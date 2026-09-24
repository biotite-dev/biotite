"""
A subpackage that provides interfaces to the *ViennaRNA* software
package.

Secondary structures can be predicted using *RNAfold* and plotted using
*RNAplot*.
"""

__name__ = "biotite.application.viennarna"
__author__ = "Tom David Müller"

from .rnaalifold import *
from .rnafold import *
from .rnaplot import *
