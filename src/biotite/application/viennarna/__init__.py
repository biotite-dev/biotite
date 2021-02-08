# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage that provides interfaces to the *ViennaRNA* software
package.

Secondary structures can be predicted using *RNAfold* and plotted using
*RNAplot*.
"""

__name__ = "biotite.application.viennarna"
__author__ = "Tom David Müller"

from .rnafold import *
from .rnaplot import *
