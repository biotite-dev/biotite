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

    When extracting data from or setting data in the file, only a
    shallow copy as created.
    """
    
    def __init__(self):
        super().__init__()
        self._coord = None
        self._time = None
        self._box = None
        self._model_count = None
    
    def read(self, file_name, start=None, stop=None, step=None, atom_i=None):
        """
        Read a trajectory file.
        
        A trajectory file can be seen as a file representation of an
        `AtomArrayStack`.
        Therefore, `start`, `stop` and `step` represent slice parameters
        of the index of the first dimension and
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
        coord, box, time = self.process_read_values(result)
        self.set_coord(coord)
        self.set_box(box)
        self.set_time(time)
    
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
        param = self.prepare_write_values(self._coord, self._box, self._time)
        with traj_type(file_name, 'w') as f:
            f.write(**param)
    
    def get_coord(self):
        """
        Extract only the atom coordinates from the trajectory file.
        
        Returns
        -------
        coord : ndarray, dtype=float, shape=(m,n,3)
            The coordinates stored in the trajectory file.
        """
        return self._coord
    
    def get_time(self):
        """
        Get the simlation time in ps values for each frame.
        
        Returns
        -------
        time : ndarray, dtype=float, shape=(m,)
            A one dimensional array containing the time values for the
            frames, that were read fro the file.
        """
        return self._time
    
    def get_box(self):
        """
        Get the box vectors for each frame.
        
        Returns
        -------
        box : ndarray, dtype=float, shape=(m,3,3)
            An array containing the box dimensions for the
            frames, that were read from the file.
        """
        return self._box
    
    def set_coord(self, coord):
        """
        Set the atom coordinates in the trajectory file.
        
        Parameters
        ----------
        coord : ndarray, dtype=float, shape=(m,n,3)
            The coordinates to be set.
        """
        self._check_model_count(coord)
        self._coord = coord
    
    def set_time(self, time):
        """
        Set the simulation time of each frame in the trajectory file.
        
        Parameters
        ----------
        time : ndarray, dtype=float, shape=(m,)
            The simulation time to be set.
        """
        self._check_model_count(time)
        self._time = time
    
    def set_box(self, box):
        """
        Set the periodic box vectors of each frame in the trajectory
        file.
        
        Parameters
        ----------
        time : ndarray, dtype=float, shape=(m,3,3)
            The box vectors to be set.
        """
        self._check_model_count(box)
        self._box = box
    
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
            but the coordinates and the simulation boxes from the
            trajectory file.
        """
        if template.array_length() != self.get_coord().shape[-2]:
            raise ValueError(
                f"Template has {template.array_length()} atoms and trajectory "
                f"has {self.get_coord().shape[-2]} atoms, must be equal"
            )
        if isinstance(template, AtomArray):
            array_stack = stack([template])
        else:
            array_stack = template.copy()
        array_stack.coord = self.get_coord()
        array_stack.box = self.get_box()
        return array_stack
    
    def set_structure(self, structure, time=None):
        """
        Write an atom array (stack) into the trajectory file object.
        
        The topology information (chain, residue, etc.) is not saved in
        the file.
        
        Parameters
        ----------
        structure : AtomArray or AtomArrayStack
            The structure to be put into the trajectory file.
        time : ndarray, dtype=float, shape=(n,), optional
            The simulation time for each frame in `structure`.
        """
        coord = structure.coord
        box = structure.box
        if coord.ndim == 2:
            coord = coord[np.newaxis, :, :]
        if box is not None and box.ndim == 2:
            box = box[np.newaxis, :, :]
        self.set_coord(coord)
        if box is not None:
            self.set_box(box)
        if time is not None:
            self.set_time(time)

    
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
    def process_read_values(self, read_values):
        """
        Convert the return value of the `read()` method of the
        respective *MDTraj* `TrajectoryFile` into coordinates,
        simulation box and simulation time.
        
        PROTECTED: Override when inheriting.
        
        Parameters
        ----------
        read_values : tuple
            The return value of the respective *MDTraj* `TrajectoryFile`
            `read()` method.
        
        Returns
        -------
        coord : ndarray, dtype=float, shape=(m,n,3)
            The atom coordinates in Å for each frame.
        box : ndarray, dtype=float, shape=(m,3,3) or None
            The box vectors in Å for each frame.
        time : ndarray, dtype=float, shape=(m,) or None
            The simulation time in ps for each frame.
        """
        pass
    
    @abc.abstractmethod
    def prepare_write_values(self, coord, box, time):
        """
        Convert the `coord`, `box` and `time` attribute into a
        dictionary that is given as *kwargs* to the respective
        *MDTraj* `TrajectoryFile` `write()` method.

        PROTECTED: Override when inheriting.

        Parameters
        ----------
        coord : ndarray, dtype=float, shape=(m,n,3)
            The atom coordinates in Å for each frame.
        box : ndarray, dtype=float, shape=(m,3,3)
            The box vectors in Å for each frame.
        time : ndarray, dtype=float, shape=(m,)
            The simulation time in ps for each frame.
        
        Returns
        -------
        parameters : dict
            This dictionary is given as *kwargs* parameter to the
            respective *MDTraj* `TrajectoryFile` `write()` method.
        """
        pass

    def _check_model_count(self, array):
        """
        Check if the amount of models in the given array is equal to
        the amount of models in the file.
        If not, raise an exception.
        If the amount of models in the file is not set yet, set it with
        the amount of models in the array.
        """
        if array is None:
            return
        if self._model_count is None:
            self._model_count = len(array)
        else:
            if self._model_count != len(array):
                raise ValueError(
                    f"{len(array)} models were given, "
                    f"but the file contains {self._model_count} models"
                )