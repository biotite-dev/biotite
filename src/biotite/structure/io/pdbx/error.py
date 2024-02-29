# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdbx"
__author__ = "Patrick Kunzmann"
__all__ = ["SerializationError", "DeserializationError"]


class SerializationError(Exception):
    pass

class DeserializationError(Exception):
    pass