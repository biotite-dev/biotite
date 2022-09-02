# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.netcdf"
__author__ = "Patrick Kunzmann"
__all__ = ["NetCDFFile"]

import numpy as np
from ..trajfile import TrajectoryFile
from ...box import vectors_from_unitcell, unitcell_from_vectors


class NetCDFFile(TrajectoryFile):
    """
    This file class represents a NetCDF trajectory file.
    """
    
    @classmethod
    def traj_type(cls):
        import mdtraj.formats as traj
        return traj.NetCDFTrajectoryFile
    
    @classmethod
    def process_read_values(cls, read_values):
        # .dcd files use Angstrom
        coord = read_values[0]
        time = read_values[1]
        cell_lengths = read_values[2]
        cell_angles = read_values[3]
        if cell_lengths is None or cell_angles is None:
             box = None
        else:
            box = np.stack(
                [vectors_from_unitcell(a, b, c, alpha, beta, gamma)
                for (a, b, c), (alpha, beta, gamma)
                in zip(cell_lengths, np.deg2rad(cell_angles))],
                axis=0
            )
        return coord, box, time
    
    @classmethod
    def prepare_write_values(cls, coord, box, time):
        coord = coord.astype(np.float32, copy=False) \
              if coord is not None else None
        time = time.astype(np.float32, copy=False) \
               if time is not None else None
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
            "coordinates" : coord,
            "time" : time,
            "cell_lengths" : cell_lengths,
            "cell_angles" : cell_angles,
        }