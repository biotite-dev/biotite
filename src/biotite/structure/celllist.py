# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["CellList"]

from biotite.rust.structure import CellList, CellListResult

# Expose the `CellListResult` enum as more ergonomic `CellList.Result` to the user
CellListResult.__name__ = "Result"
CellListResult.__qualname__ = "CellList.Result"
CellList.Result = CellListResult
