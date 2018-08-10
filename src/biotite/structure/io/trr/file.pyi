from mdtraj.formats.trr import TRRTrajectoryFile
from typing import Type


class TRRFile:
    def output_value_index(self, value: str) -> int: ...
    def traj_type(self) -> Type[TRRTrajectoryFile]: ...
