# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import abc
import numpy as np
from ..atoms import AtomArray, AtomArrayStack, stack
from ...file import File

__all__ = ["TrajectoryFile"]


class TrajectoryFile(File, metaclass=abc.ABCMeta):
    
    def __init__(self):
        super().__init__()
        self._coord = None
        self._time = None
        self._box = None
    
    def read(self, file_name, start=None, stop=None, step=None, atom_i=None):
        super().read(file_name)
        traj_type = self._traj_type()
        with traj_type(file_name, 'r') as f:
            if start is not None and start != 0:
                # Discard atoms before start
                f.read(n_frames=start, stride=None, atom_indices=atom_i)
            # The next interval is saved
            if start is None or stop is None:
                result = f.read(stride=step, atom_indices=atom_i)
            else:
                result = f.read(stop-start, step, atom_i)
            self._coord = result[self._output_value_index("coord")]
            self._time  = result[self._output_value_index("time")]
            self._box   = result[self._output_value_index("box")]
    
    def get_coord(self):
        return self._coord
    
    def get_structure(self, template):
        if template.array_length() != self._coord.shape[-2]:
            raise ValueError("Template")
        if isinstance(template, AtomArray):
            array_stack = stack([template])
        else:
            array_stack = template.copy()
        array_stack.coord = np.copy(self._coord)
        return array_stack
    
    def get_time(self):
        return self._time
    
    def get_box(self):
        return self._box
    
    def write(self, file_name):
        super().write(file_name)
        traj_type = self._traj_type()
        with traj_type(file_name, 'w') as f:
            f.write(xyz=self._coord, time=self._time, box=self._box)
    
    def copy():
        raise NotImplementedError("Copying is not implemented "
                                  "for trajectory files")
    
    @abc.abstractmethod
    def _traj_type(self):
        pass
    
    @abc.abstractmethod
    def _output_value_index(self, value):
        pass
    