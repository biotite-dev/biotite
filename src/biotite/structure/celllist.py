__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["CellList"]

from biotite.rust.structure import CellList, CellListResult

# Expose the `CellListResult` enum as more ergonomic `CellList.Result` to the user
CellListResult.__name__ = "Result"
CellListResult.__qualname__ = "CellList.Result"
CellList.Result = CellListResult
