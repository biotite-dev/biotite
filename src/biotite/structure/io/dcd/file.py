# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.dcd"
__author__ = "Patrick Kunzmann"
__all__ = ["DCDFile"]

import numpy as np
from ..trajfile import TrajectoryFile
from ...box import vectors_from_unitcell, unitcell_from_vectors


class DCDFile(TrajectoryFile):
    """
    This file class represents a DCD trajectory file.
    """
    
    @classmethod
    def traj_type(cls):
        import mdtraj.formats as traj
        return traj.DCDTrajectoryFile
    
    @classmethod
    def process_read_values(cls, read_values):
        # .netcdf files use Angstrom
        coord = read_values[0]
        cell_lengths = read_values[1]
        cell_angles = read_values[2]
        if cell_lengths is None or cell_angles is None:
             box = None
        else:
            box = np.stack(
                [vectors_from_unitcell(a, b, c, alpha, beta, gamma)
                for (a, b, c), (alpha, beta, gamma)
                in zip(cell_lengths, np.deg2rad(cell_angles))],
                axis=0
            )
        return coord, box, None
    
    @classmethod
    def prepare_write_values(cls, coord, box, time):
        xyz = coord.astype(np.float32, copy=False) \
              if coord is not None else None
        if box is None:
            cell_lengths = None
            cell_angles  = None
        else:
            cell_lengths = np.zeros((len(box), 3), dtype=np.float32)
            cell_angles  = np.zeros((len(box), 3), dtype=np.float32)
            for i, model_box in enumerate(box):
                a, b, c, alpha, beta, gamma = unitcell_from_vectors(model_box)
                cell_lengths[i] = np.array((a, b, c))
                cell_angles[i] = np.rad2deg((alpha, beta, gamma))
        return {
            "xyz" : xyz,
            "cell_lengths" : cell_lengths,
            "cell_angles" : cell_angles,
        }

    def set_time(self, time):
        if time is not None:
            raise NotImplementedError(
                "This trajectory file does not support writing simulation time"
            )