# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["TrajectoryFile"]

import abc
import numpy as np
from ..atoms import AtomArray, AtomArrayStack, stack
from ...file import File


class TrajectoryFile(File, metaclass=abc.ABCMeta):
    """
    This file class represents a trajectory file interfacing a
    trajectory file class from `MDtraj`.
    
    A trajectory file stores atom coordinates over multiple (time)
    frames. The file formats are usually binary and involve sometimes
    heavy compression, so that a large number of frames can be stored
    in relatively small space.
    Since all `TrajectoryFile` subclasses interface `MDtraj` trajectory
    file classes, `MDtraj` must be installed to use any of them.
    """
    
    def __init__(self):
        super().__init__()
        self._coord = None
        self._time = None
        self._box = None
    
    def read(self, file_name, start=None, stop=None, step=None, atom_i=None):
        """
        Read a trajectory file.
        
        A trajectory file can be seen as a file representation of an
        `AtomArrayStack`. `start`, `stop` and `step` represent therefore
        slice parameters of the index of the first dimension and
        `atom_i` represents an index array for the second dimension.
        
        Parameters
        ----------
        file_name : str
            The path of the file to be read.
            A file-like-object cannot be used.
        start : int, optional
            The frame index, where file parsing is started. If no value
            is given, parsing starts at the first frame. The index
            starts at 0.
        stop : int, optional
            The exclusive frame index, where file parsing ends.
            If no value is given, parsing stops after the last frame.
            The index starts at 0.
        step : int, optional
            If this value is set, the method reads only every n-th frame
            from the file.
        atom_i : ndarray, dtype=int
            The atom indices to be read from the file.
        """
        traj_type = self.traj_type()
        with traj_type(file_name, 'r') as f:
            if start is not None and start != 0:
                # Discard atoms before start
                f.read(n_frames=start, stride=None, atom_indices=atom_i)
            # The next interval is saved
            if start is None or stop is None:
                result = f.read(stride=step, atom_indices=atom_i)
            else:
                result = f.read(stop-start, step, atom_i)
            # nm to Angstrom
            self._coord = result[self.output_value_index("coord")] * 10
            self._time  = result[self.output_value_index("time")]
            self._box   = result[self.output_value_index("box")]
    
    def get_coord(self):
        """
        Extract only the coordinates from the trajectory file.
        
        Returns
        -------
        indices : ndarray, dtype=float
            The coordinates stored in the trajectory file.
        """
        return self._coord
    
    def get_structure(self, template):
        """
        Convert the trajectory file content into an `AtomArrayStack`.
        
        Since trajectory files usually only contain atom coordinate
        information and no topology information, this method requires
        a template atom array or stack. This template can be acquired
        for example from a PDB file, which is associated with the
        trajectory file. 
        
        Parameters
        ----------
        template : AtomArray or AtomArrayStack
            The template array or stack, where the atom annotation data
            is taken from.
        
        Returns
        -------
        array_stack : AtomArrayStack
            A stack containing the annontation arrays from `template`
            but the coordinates from the trajectory file.
        """
        if template.array_length() != self._coord.shape[-2]:
            raise ValueError("Template and trajectory have "
                             "unequal amount of atoms")
        if isinstance(template, AtomArray):
            array_stack = stack([template])
        else:
            array_stack = template.copy()
        array_stack.coord = np.copy(self._coord)
        return array_stack
    
    def get_time(self):
        """
        Get the time values for each frame.
        
        Returns
        -------
        time : ndarray, dtype=float
            A one dimensional array containing the time values for the
            frames, that were read fro the file.
        """
        return self._time
    
    def get_box(self):
        """
        Get the box dimensions for each frame (nm).
        
        Returns
        -------
        time : ndarray, dtype=float, shape=(n,3)
            An array containing the box dimensions for the
            frames, that were read from the file.
        """
        return self._box
    
    def write(self, file_name):
        """
        Write the content into a trajectory file.

        Parameters
        ----------
        file_name : str
            The path of the file to be written to.
            A file-like-object cannot be used.
        """
        traj_type = self.traj_type()
        with traj_type(file_name, 'w') as f:
            f.write(xyz=self._coord, time=self._time, box=self._box)
    
    def copy(self):
        """
        This operation is not implemented for trajectory files.
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Copying is not implemented "
                                  "for trajectory files")
    
    @abc.abstractmethod
    def traj_type(self):
        """
        The `MDtraj` files class to be used.
        
        PROTECTED: Override when inheriting.
        
        Returns
        -------
        class
            An `MDtraj` subclass of `TrajectoryFile`.
        """
        pass
    
    @abc.abstractmethod
    def output_value_index(self, value):
        """
        Map the values "coord", "time" and "box" to indices in the
        tuple returned from `MDtraj` `TrajectoryFile.read()` method.
        
        PROTECTED: Override when inheriting.
        
        Parameters
        ----------
        value : str
            "coord", "time" or "box".
        
        Returns
        -------
        int
            The index where `value` is in the returned tuple.
        """
        pass
    