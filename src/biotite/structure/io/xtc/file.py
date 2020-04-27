# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.xtc"
__author__ = "Patrick Kunzmann"
__all__ = ["XTCFile"]

import numpy as np
from ..trajfile import TrajectoryFile


class XTCFile(TrajectoryFile):
    """
    This file class represents a XTC trajectory file.
    """
    
    @classmethod
    def traj_type(cls):
        import mdtraj.formats as traj
        return traj.XTCTrajectoryFile

    @classmethod
    def process_read_values(cls, read_values):
        # nm to Angstrom
        coord = read_values[0] * 10
        box = read_values[3]
        if box is not None:
            box *= 10
        time = read_values[1]
        return coord, box, time
    
    @classmethod
    def prepare_write_values(cls, coord, box, time):
        # Angstrom to nm
        xyz = np.divide(coord, 10, dtype=np.float32) \
              if coord is not None else None
        time = time.astype(np.float32, copy=False) \
               if time is not None else None
        box = np.divide(box, 10, dtype=np.float32) \
              if box is not None else None
        return {
            "xyz"  : xyz,
            "box"  : box,
            "time" : time,
        }
