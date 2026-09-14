# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage that provides interfaces to the *ViennaRNA* software
package.

Secondary structures can be predicted using a variety of folding
programs (:mod:`fold`) and plotted using ``RNAplot``.
"""

__name__ = "biotite.application_v2.viennarna"
__author__ = "Tom David Müller"

from .fold import *
from .plot import *
from .result import *
