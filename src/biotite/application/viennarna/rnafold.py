# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.viennarna"
__author__ = "Tom David Müller, Patrick Kunzmann"
__all__ = ["RNAfoldApp"]

from tempfile import NamedTemporaryFile
import numpy as np
from biotite.application.application import AppState, requires_state
from biotite.application.localapp import LocalApp, cleanup_tempfile
from biotite.application.viennarna.util import build_constraint_string
from biotite.sequence.io.fasta import FastaFile, set_sequence
from biotite.structure.dotbracket import base_pairs_from_dot_bracket


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
    >>> print(app.get_free_energy())
    -1.3
    >>> print(app.get_dot_bracket())
    (((.((((.......)).)))))....
    """

    def __init__(self, sequence, temperature=37, bin_path="RNAfold"):
        self._sequence = sequence.copy()
        self._temperature = str(temperature)
        self._constraints = None
        self._enforce = None
        self._in_file = NamedTemporaryFile("w", suffix=".fa", delete=False)
        super().__init__(bin_path)

    def run(self):
        # Insert no line breaks
        # -> Extremely high value for characters per line
        fasta_file = FastaFile(chars_per_line=np.iinfo(np.int32).max)
        set_sequence(fasta_file, self._sequence)
        if self._constraints is not None:
            fasta_file.lines.append(self._constraints)
        fasta_file.write(self._in_file)
        self._in_file.flush()

        options = [
            "--noPS",
            "-T",
            self._temperature,
        ]
        if self._enforce is True:
            options.append("--enforceConstraint")
        if self._constraints is not None:
            options.append("-C")

        self.set_arguments(options + [self._in_file.name])
        super().run()

    def evaluate(self):
        super().evaluate()
        lines = self.get_stdout().splitlines()
        content = lines[2]
        dotbracket, free_energy = content.split(" ", maxsplit=1)
        free_energy = float(free_energy[1:-1])

        self._free_energy = free_energy
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

    @requires_state(AppState.CREATED)
    def set_constraints(
        self,
        pairs=None,
        paired=None,
        unpaired=None,
        downstream=None,
        upstream=None,
        enforce=False,
    ):
        """
        Add constraints of known paired or unpaired bases to the folding
        algorithm.

        Constraints forbid pairs conflicting with the respective
        constraint.

        Parameters
        ----------
        pairs : ndarray, shape=(n,2), dtype=int, optional
            Positions of constrained base pairs.
        paired : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any other base.
        unpaired : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are unpaired.
        downstream : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any downstream base.
        upstream : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any upstream base.
        enforce : bool, optional
            If set to true, the given constraints are enforced, i.e. a
            the respective base pairs must form.
            By default (false), a constraint does only forbid formation
            of a pair that would conflict with this constraint.
        """
        self._constraints = build_constraint_string(
            len(self._sequence), pairs, paired, unpaired, downstream, upstream
        )
        self._enforce = enforce

    @requires_state(AppState.JOINED)
    def get_free_energy(self):
        """
        Get the free energy (kcal/mol) of the suggested
        secondary structure.

        Returns
        -------
        free_energy : float
            The free energy.

        Examples
        --------

        >>> sequence = NucleotideSequence("CGACGTAGATGCTAGCTGACTCGATGC")
        >>> app = RNAfoldApp(sequence)
        >>> app.start()
        >>> app.join()
        >>> print(app.get_free_energy())
        -1.3
        """
        return self._free_energy

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
        free_energy : float
            The free energy.
        """
        app = RNAfoldApp(sequence, bin_path=bin_path)
        app.start()
        app.join()
        return app.get_dot_bracket(), app.get_free_energy()
