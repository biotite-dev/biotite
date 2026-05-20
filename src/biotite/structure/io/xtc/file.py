# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.xtc"
__author__ = "Patrick Kunzmann"
__all__ = ["XTCFile"]

from typing import Any
import biotraj
import numpy as np
from biotite.structure.io.trajfile import TrajectoryFile
from biotite.typing import XYZ, NDArray1, NDArray3


class XTCFile(TrajectoryFile):
    """
    This file class represents a XTC trajectory file.
    """

    @classmethod
    def traj_type(cls) -> type[biotraj.XTCTrajectoryFile]:
        return biotraj.XTCTrajectoryFile

    @classmethod
    def process_read_values(
        cls, read_values: Any
    ) -> tuple[
        NDArray3[Any, Any, XYZ, np.floating],
        NDArray3[Any, XYZ, XYZ, np.floating] | None,
        NDArray1[Any, np.floating] | None,
    ]:
        # nm to Angstrom
        coord = read_values[0] * 10
        box = read_values[3]
        if box is not None:
            box *= 10
        time = read_values[1]
        return coord, box, time

    @classmethod
    def prepare_write_values(
        cls,
        coord: NDArray3[Any, Any, XYZ, np.floating] | None,
        box: NDArray3[Any, XYZ, XYZ, np.floating] | None,
        time: NDArray1[Any, np.floating] | None,
    ) -> dict[str, Any]:
        # Angstrom to nm
        xyz = np.divide(coord, 10, dtype=np.float32) if coord is not None else None
        time = time.astype(np.float32, copy=False) if time is not None else None
        box = np.divide(box, 10, dtype=np.float32) if box is not None else None
        return {
            "xyz": xyz,
            "box": box,
            "time": time,
        }
