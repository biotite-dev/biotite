# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
NumPy port of the ``foldseek`` code for encoding structures to 3di.
"""

__all__ = ["Encoder", "FeatureEncoder", "PartnerIndexEncoder", "VirtualCenterEncoder"]
__author__ = "Martin Larralde <martin.larralde@embl.de>"
__name__ = "biotite.structure.alphabet"

from .encoder import Encoder, FeatureEncoder, PartnerIndexEncoder, VirtualCenterEncoder
