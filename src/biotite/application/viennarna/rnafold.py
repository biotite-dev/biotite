# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.viennarna"
__author__ = "Tom David Müller"
__all__ = ["RNAfoldApp"]

from tempfile import NamedTemporaryFile
from ..localapp import LocalApp, cleanup_tempfile
from ..application import AppState, requires_state
from ...sequence.io.fasta import FastaFile, set_sequence
from ...sequence import NucleotideSequence
from ...structure.dotbracket import base_pairs_from_dot_bracket

class RNAfoldApp(LocalApp):
    """
    Compute the minimum free energy secondary structure of a ribonucleic
    acid sequence using *ViennaRNA's* *RNAfold* software.

    Internally this creates a :class:`Popen` instance, which handles
    the execution.

    Parameters
    ----------
    sequence : NucleotideSequence
        The RNA sequence.
    temperature : int, optional
        The temperature (°C) to be assumed for the energy parameters.
    bin_path : str, optional
        Path of the *RNAfold* binary.

    Examples
    --------

    >>> sequence = NucleotideSequence("CGACGTAGATGCTAGCTGACTCGATGC")
    >>> app = RNAfoldApp(sequence)
    >>> app.start()
    >>> app.join()
    >>> print(app.get_mfe())
    -1.3
    >>> print(app.get_dot_bracket())
    (((.((((.......)).)))))....
    """

    def __init__(self, sequence, temperature=37, bin_path="RNAfold"):
        super().__init__(bin_path)
        self._sequence = sequence
        self._in_file  = NamedTemporaryFile("w", suffix=".fa",  delete=False)
        self._temperature = str(temperature)

    def run(self):
        in_file = FastaFile()
        set_sequence(in_file, self._sequence)
        in_file.write(self._in_file)
        self._in_file.flush()
        self.set_arguments(
            [self._in_file.name, "--noPS", "-T", self._temperature]
        )
        super().run()

    def evaluate(self):
        super().evaluate()
        lines = self.get_stdout().split("\n")
        content = lines[2]
        dotbracket, mfe = content.split(" ", maxsplit=1)
        mfe = float(mfe[1:-1])

        self._mfe = mfe
        self._dotbracket = dotbracket

    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._in_file)

    @requires_state(AppState.CREATED)
    def set_temperature(self, temperature):
        """
        Adjust the energy parameters according to a temperature in
        degrees Celsius.

        Parameters
        ----------
        temperature : int
            The temperature.
        """
        self._temperature = str(temperature)

    @requires_state(AppState.JOINED)
    def get_mfe(self):
        """
        Get the minimum free energy of the input sequence.

        Returns
        -------
        mfe : float
            The minimum free energy.

        Examples
        --------

        >>> sequence = NucleotideSequence("CGACGTAGATGCTAGCTGACTCGATGC")
        >>> app = RNAfoldApp(sequence)
        >>> app.start()
        >>> app.join()
        >>> print(app.get_mfe())
        -1.3
        """
        return self._mfe

    @requires_state(AppState.JOINED)
    def get_dot_bracket(self):
        """
        Get the minimum free energy secondary structure of the input
        sequence in dot bracket notation.

        Returns
        -------
        dotbracket : str
            The secondary structure in dot bracket notation.

        Examples
        --------

        >>> sequence = NucleotideSequence("CGACGTAGATGCTAGCTGACTCGATGC")
        >>> app = RNAfoldApp(sequence)
        >>> app.start()
        >>> app.join()
        >>> print(app.get_dot_bracket())
        (((.((((.......)).)))))....
        """
        return self._dotbracket

    @requires_state(AppState.JOINED)
    def get_base_pairs(self):
        """
        Get the base pairs from the minimum free energy secondary
        structure of the input sequence.

        Returns
        -------
        base_pairs : ndarray, shape=(n,2)
            Each row corresponds to the positions of the bases in the
            sequence.

        Examples
        --------

        >>> sequence = NucleotideSequence("CGACGTAGATGCTAGCTGACTCGATGC")
        >>> app = RNAfoldApp(sequence)
        >>> app.start()
        >>> app.join()
        >>> print(app.get_base_pairs())
            [[ 0 22]
             [ 1 21]
             [ 2 20]
             [ 4 19]
             [ 5 18]
             [ 6 16]
             [ 7 15]]

        For reference, the corresponding dot bracket notation can be
        displayed as below.

        >>> print(app.get_dot_bracket())
        (((.((((.......)).)))))....
        """
        return base_pairs_from_dot_bracket(self._dotbracket)

    @staticmethod
    def compute_secondary_structure(sequence, bin_path="RNAfold"):
        """
        Compute the minimum free energy secondary structure of a 
        ribonucleic acid sequence using *ViennaRNA's* *RNAfold* software.

        This is a convenience function, that wraps the
        :class:`RNAfoldApp` execution.

        Parameters
        ----------
        sequence : NucleotideSequence
            The RNA sequence.
        bin_path : str, optional
            Path of the *RNAfold* binary.

        Returns
        -------
        dotbracket : str
            The secondary structure in dot bracket notation.
        mfe : float
            The minimum free energy.
        """
        app = RNAfoldApp(sequence, bin_path=bin_path)
        app.start()
        app.join()
        return app.get_dot_bracket(), app.get_mfe()
