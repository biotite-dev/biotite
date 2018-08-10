from mdtraj.formats.xtc import XTCTrajectoryFile
from typing import Type


class XTCFile:
    def traj_type(self) -> Type[XTCTrajectoryFile]: ...
