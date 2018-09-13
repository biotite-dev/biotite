# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["DsspApp"]

from ..localapp import LocalApp
from ..application import AppState, requires_state
from ...temp import temp_file
from ...structure.io.pdb import PDBFile
import numpy as np


class DsspApp(LocalApp):
    r"""
    Annotate the secondary structure of a protein structure using the
    DSSP software.
    
    Internally this creates a `Popen` instance, which handles
    the execution.
    
    DSSP differentiates between 8 different types of secondary
    structure elements:
    
       - C: loop, coil or irregular
       - H: :math:`{\alpha}`-helix
       - B: :math:`{\beta}`-bridge
       - E: extended strand, participation in :math:`{\beta}`-ladder
       - G: 3 :sub:`10`-helix
       - I: :math:`{\pi}`-helix
       - T: hydrogen bonded turn
       - S: bend 
    
    Parameters
    ----------
    atom_array : AtomArray
        The atom array to be annotated.
    bin_path : str, optional
        Path of the DDSP binary.
    """
    
    def __init__(self, atom_array, bin_path="mkdssp"):
        super().__init__(bin_path)
        self._array = atom_array
        self._in_file_name  = temp_file("pdb")
        self._out_file_name = temp_file("pdb")

    def run(self):
        in_file = PDBFile()
        in_file.set_structure(self._array)
        in_file.write(self._in_file_name)
        self.set_options(["-i", self._in_file_name, "-o", self._out_file_name])
        super().run()
    
    def evaluate(self):
        super().evaluate()
        with open(self._out_file_name, "r") as f:
            lines = f.read().split("\n")
        # Index where SSE records start
        sse_start = None
        for i, line in enumerate(lines):
            if line.startswith("  #  RESIDUE AA STRUCTURE"):
                sse_start = i+1
        if sse_start is None:
            raise ValueError("DSSP file does not contain SSE records")
        lines = [line for line in lines[sse_start:] if len(line) != 0]
        self._sse = np.zeros(len(lines), dtype="U1")
        # Parse file for SSE letters
        for i, line in enumerate(lines):
            self._sse[i] = line[16]
        # Remove "!" for missing residues
        self._sse = self._sse[self._sse != "!"]
        self._sse[self._sse == " "] = "C"
    
    @requires_state(AppState.JOINED)
    def get_sse(self):
        """
        Get the resulting secondary structure assignment.
        
        Returns
        -------
        sse : ndarray, dtype="U1"
            An array containing DSSP secondary structure symbols
            corresponding to the residues in the input atom array.
        """
        return self._sse
    
    @staticmethod
    def annotate_sse(atom_array, bin_path="mkdssp"):
        """
        Perform a secondary structure assignment to an atom array.
        
        This is a convenience function, that wraps the `DsspApp`
        execution.
        
        Parameters
        ----------
        atom_array : AtomArray
            The atom array to be annotated.
        bin_path : str, optional
            Path of the DDSP binary.
        
        Returns
        -------
        sse : ndarray, dtype="U1"
            An array containing DSSP secondary structure symbols
            corresponding to the residues in the input atom array.
        """
        app = DsspApp(atom_array, bin_path)
        app.start()
        app.join()
        return app.get_sse()
