# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io"
__author__ = "Patrick Kunzmann"
__all__ = ["TrajectoryFile"]

import abc
import numpy as np
from ..atoms import AtomArray, AtomArrayStack, stack, from_template
from ...file import File


class TrajectoryFile(File, metaclass=abc.ABCMeta):
    """
    This file class represents a trajectory file interfacing a
    trajectory file class from `MDtraj`.
    
    A trajectory file stores atom coordinates over multiple (time)
    frames. The file formats are usually binary and involve sometimes
    heavy compression, so that a large number of frames can be stored
    in relatively small space.
    Since all :class:`TrajectoryFile` subclasses interface *MDtraj*
    trajectory file classes, `MDtraj` must be installed to use any of
    them.

    Notes
    -----
    When extracting data from the file, only a reference to the
    data arrays stored in this file are created.
    The same is true, when setting data in the file.
    Therefore, it is strongly recommended to make a copy of the
    respective array, if the array is modified.
    """
    
    def __init__(self):
        super().__init__()
        self._coord = None
        self._time = None
        self._box = None
        self._model_count = None
    

    @classmethod
    def read(cls, file_name, start=None, stop=None, step=None,
             atom_i=None, chunk_size=None):
        """
        Read a trajectory file.
        
        A trajectory file can be seen as a file representation of an
        :class:`AtomArrayStack`.
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
            is given, parsing starts at the first frame.
            The index starts at 0.
        stop : int, optional
            The exclusive frame index, where file parsing ends.
            If no value is given, parsing stops after the last frame.
            The index starts at 0.
        step : int, optional
            If this value is set, the function reads only every n-th
            frame from the file.
        atom_i : ndarray, dtype=int, optional
            If this parameter is set, only the atoms at the given
            indices are read from each frame.
        chunk_size : int, optional
            If this parameter is set, the trajectory is read in chunks:
            Only the number of frames specified by this parameter are
            read at once.
            The resulting chunks of frames are automatically
            concatenated, after all chunks are collected.
            Use this parameter, if a :class:`MemoryError` is raised
            when a trajectory file is read.
            Although lower values can decrease the memory consumption of
            reading trajectories, they also increase the computation
            time.
        
        Returns
        -------
        file_object : TrajectoryFile
            The parsed trajectory file.
        """
        file = cls()

        if chunk_size is not None:
            if chunk_size < 1:
                raise ValueError("Chunk size must be greater than 0")
            # Chunk size must be a multiple of step size to ensure that
            # the step distance at the chunk border is the same as
            # within a chunk
            # -> round chunk size up to a multiple of step size
            if step is not None and chunk_size % step != 0:
                chunk_size = ((chunk_size // step) + 1) * step

        traj_type = cls.traj_type()
        with traj_type(file_name, "r") as f:
            
            if start is None:
                start = 0
            # Discard atoms before start
            if start != 0:
                if chunk_size is None or chunk_size > start:
                    f.read(n_frames=start, stride=None, atom_indices=atom_i)
                else:
                    TrajectoryFile._read_chunk_wise(
                        f, start, None, atom_i, chunk_size, discard=True
                    )
            
            # The upcoming frames are saved
            # Calculate the amount of frames to be read
            if stop is None:
                n_frames = None
            else:
                n_frames = stop-start
            if step is not None and n_frames is not None:
                # Divide number of frames by 'step' in order to convert
                # 'step' into 'stride'
                # Since the 0th frame is always included,
                # the number of frames is decremented before division
                # and incremented afterwards again
                n_frames = ((n_frames - 1) // step) + 1
            
            # Read frames
            if chunk_size is None:
                result = f.read(n_frames, stride=step, atom_indices=atom_i)
            else:
                result = TrajectoryFile._read_chunk_wise(
                    f, n_frames, step, atom_i, chunk_size, discard=False
                )
        
        # nm to Angstrom
        coord, box, time = cls.process_read_values(result)
        file.set_coord(coord)
        file.set_box(box)
        file.set_time(time)

        return file
    

    @classmethod
    def read_iter(cls, file_name, start=None, stop=None, step=None,
                  atom_i=None):
        """
        Create an iterator over each frame of the given trajectory file
        in the selected range.
        
        Parameters
        ----------
        file_name : str
            The path of the file to be read.
            A file-like-object cannot be used.
        start : int, optional
            The frame index, where file parsing is started. If no value
            is given, parsing starts at the first frame.
            The index starts at 0.
        stop : int, optional
            The exclusive frame index, where file parsing ends.
            If no value is given, parsing stops at the end of file.
            The index starts at 0.
        step : int, optional
            If this value is set, the function reads only every n-th
            frame from the file.
        atom_i : ndarray, dtype=int, optional
            If this parameter is set, only the atoms at the given
            indices are read from each frame.
        
        Yields
        ------
        coord : ndarray, dtype=float32, shape=(n,3)
            The atom coordinates in the current frame.
        box : ndarray, dtype=float32, shape=(3,3)
            The box vectors of the current frame.
        time : float
            the simlation time of the current frame in *ps*.
        
        See also
        --------
        read_iter_structure
        
        Notes
        -----
        The `step` parameter does currently not work for *DCD* files.
        """
        traj_type = cls.traj_type()
        with traj_type(file_name, "r") as f:
            
            if start is None:
                start = 0
            # Discard atoms before start
            if start != 0:
                f.read(n_frames=start, stride=None, atom_indices=atom_i)
            
            # The upcoming frames are read
            # Calculate the amount of frames to be read
            if stop is None:
                n_frames = None
            else:
                n_frames = stop-start
            if step is not None and n_frames is not None:
                # Divide number of frames by 'step' in order to convert
                # 'step' into 'stride'
                # Since the 0th frame is always included,
                # the number of frames is decremented before division
                # and incremented afterwards again
                n_frames = ((n_frames - 1) // step) + 1
            
            # Read frames
            frame_i = 0
            while True:
                if n_frames is not None and frame_i >= n_frames:
                    # Stop frame reached -> stop interation
                    break
                # Read one frame per 'yield'
                result = f.read(1, stride=step, atom_indices=atom_i)
                if len(result[0]) == 0:
                    # Empty array was read
                    # -> no frames left -> stop interation
                    break
                coord, box, time = cls.process_read_values(result)
                # Only one frame
                # -> only one element in first dimension
                # -> remove first dimension
                coord = coord[0]
                box = box[0] if box is not None else None
                time = float(time[0]) if time is not None else None
                yield coord, box, time
                frame_i += 1
    

    @classmethod
    def read_iter_structure(cls, file_name, template, start=None, stop=None,
                            step=None, atom_i=None):
        """
        Create an iterator over each frame of the given trajectory file
        in the selected range.

        In contrast to :func:`read_iter()`, this function creates an
        iterator over the structure as :class:`AtomArray`.
        Since trajectory files usually only contain atom coordinate
        information and no topology information, this method requires
        a template atom array or stack. This template can be acquired
        for example from a PDB file, which is associated with the
        trajectory file. 
        
        Parameters
        ----------
        file_name : str
            The path of the file to be read.
            A file-like-object cannot be used.
        template : AtomArray or AtomArrayStack
            The template array or stack, where the atom annotation data
            is taken from.
        start : int, optional
            The frame index, where file parsing is started. If no value
            is given, parsing starts at the first frame.
            The index starts at 0.
        stop : int, optional
            The exclusive frame index, where file parsing ends.
            If no value is given, parsing stops at the end of file.
            The index starts at 0.
        step : int, optional
            If this value is set, the function reads only every n-th
            frame from the file.
        atom_i : ndarray, dtype=int, optional
            If this parameter is set, only the atoms at the given
            indices are read from each frame.
        
        Yields
        ------
        structure : AtomArray
            The structure of the current frame.
        
        See also
        --------
        read_iter
        
        Notes
        -----
        This iterator creates a new copy of the given template for every
        frame.
        If a higher efficiency is required, please use the
        :func:`read_iter()` function.

        The `step` parameter does currently not work for *DCD* files.
        """
        if isinstance(template, AtomArrayStack):
            template = template[0]
        elif not isinstance(template, AtomArray):
            raise TypeError(
                f"An 'AtomArray' or 'AtomArrayStack' is expected as template, "
                f"not '{type(template).__name__}'"
            )
        for coord, box, _ in cls.read_iter(
            file_name, start, stop, step, atom_i
        ):
            frame = template.copy()
            frame.coord = coord
            frame.box = box
            yield frame

    
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
        Get the simlation time in *ps* values for each frame.
        
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
        Convert the trajectory file content into an
        :class:`AtomArrayStack`.
        
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
        return from_template(template, self.get_coord(), self.get_box())
    

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
    

    @classmethod
    @abc.abstractmethod
    def traj_type(cls):
        """
        The `MDtraj` files class to be used.
        
        PROTECTED: Override when inheriting.
        
        Returns
        -------
        class
            An `MDtraj` subclass of :class:`TrajectoryFile`.
        """
        pass
    

    @classmethod
    @abc.abstractmethod
    def process_read_values(cls, read_values):
        """
        Convert the return value of the `read()` method of the
        respective :class:`mdtraj.TrajectoryFile` into coordinates,
        simulation box and simulation time.
        
        PROTECTED: Override when inheriting.
        
        Parameters
        ----------
        read_values : tuple
            The return value of the respective
            :func:`mdtraj.TrajectoryFile.read()` method.
        
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
    

    @classmethod
    @abc.abstractmethod
    def prepare_write_values(cls, coord, box, time):
        """
        Convert the `coord`, `box` and `time` attribute into a
        dictionary that is given as *kwargs* to the respective
        :func:`mdtraj.TrajectoryFile.write()` method.

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
            respective :func:`mdtraj.TrajectoryFile.write()` method.
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
    

    @staticmethod
    def _read_chunk_wise(file, n_frames, step, atom_i, chunk_size,
                         discard=False):
        """
        Similar to :func:`read()`, just for chunk-wise reading of the
        trajectory.

        `n_frames` is already the actual number of frames in the outout
        arrays, i.e. the original number was divided by `step`.
        """
        chunks = []
        remaining_frames = n_frames
        # If n_frames is None, this is condition is never False
        # -> break out of loop when read chunk is empty (see below)
        while remaining_frames != 0:
            if remaining_frames is not None:
                n = min(remaining_frames, chunk_size)
            else:
                n = chunk_size
            try:
                chunk = file.read(n_frames=n, stride=step, atom_indices=atom_i)
            except ValueError as e:
                # MDTraj raises exception because no coordinates can be
                # concatenated
                # -> all frames have been read
                # -> stop reading chunks
                if str(e) != "need at least one array to concatenate":
                    raise
                else:
                    break
            if len(chunk[0]) == 0:
                # Coordinates have a length of 0
                # -> all frames have been read
                # -> stop reading chunks
                break
            if not discard:
                chunks.append(chunk)
            if remaining_frames is not None:
                remaining_frames -= n
        
        if not discard:
            # Assemble the chunks into contiguous arrays
            # for each value (coord, box, time)
            result = [None] * len(chunks[0])
            # Iterate over all valuesin the result tuple
            # and concatenate the corresponding value from each chunk,
            # if the value is not None
            # The amount of values is determined from the first chunk
            for i in range(len(chunks[0])):
                if chunks[0][i] is not None:
                    result[i] = np.concatenate([chunk[i] for chunk in chunks])
                else:
                    result[i] = None
            return tuple(result)
        else:
            return None