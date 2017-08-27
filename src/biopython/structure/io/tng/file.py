# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import mdtraj.formats as traj
from ..trajfile import TrajectoryFile

__all__ = ["TNGFile"]


class TNGFile(TrajectoryFile):
    
    def _traj_type(self):
        return traj.TNGTrajectoryFile
    
    def _output_value_index(self, value):
        if value == "coord":
            return 0
        if value == "time":
            return 1
        if value == "box":
            return 3
