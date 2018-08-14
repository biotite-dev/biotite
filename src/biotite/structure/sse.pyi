# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from biotite.structure.atoms import AtomArray
from numpy import ndarray


def annotate_sse(atom_array: AtomArray, chain_id: str) -> ndarray: ...
