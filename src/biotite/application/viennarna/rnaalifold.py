# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.viennarna"
__author__ = "Tom David Müller"
__all__ = ["RNAalifoldApp"]

from .basefold import BaseFoldApp
from ..application import AppState, requires_state
from ...sequence.io.fasta import FastaFile, set_alignment
from ...structure.dotbracket import base_pairs_from_dot_bracket
from ...structure.bonds import BondList

class RNAalifoldApp(BaseFoldApp):
    """
    Predict the secondary structure of a ribonucleic acid sequence using
    *ViennaRNA's* *RNAalifold* software.

    In contrast to :class:`RNAfoldApp`, the energy function includes
    a term that includes coevolution information extracted from an
    alignment in addition to the physical free energy term.

    Internally this creates a :class:`Popen` instance, which handles
    the execution.

    Parameters
    ----------
    alignment : Alignment
        An alignment of RNA sequences.
    temperature : int, optional
        The temperature (°C) to be assumed for the energy parameters.
    bin_path : str, optional
        Path of the *RNAalifold* binary.
    """

    def __init__(self, alignment, temperature=37, bin_path="RNAalifold"):
        fasta_file = FastaFile()
        set_alignment(
            fasta_file, alignment,
            seq_names=[str(i) for i in range(len(alignment.sequences))]
        )
        super().__init__(fasta_file, temperature, bin_path)
        self._alignment = alignment
        self._temperature = str(temperature)

    @staticmethod
    def accepts_stdin():
        return False

    def run(self):
        self.set_arguments(["--noPS"])
        super().run()

    def evaluate(self):
        super().evaluate()
        lines = self.get_stdout().splitlines()
        content = lines[1]
        dotbracket, total_energy = content.split(" ", maxsplit=1)
        # Energy has the form:
        # (<total> = <free> + <covariance>)
        total_energy = total_energy[1:-1]
        energy_contributions = total_energy.split("=")[1].split("+")
        self._free_energy = float(energy_contributions[0])
        self._covariance_energy = float(energy_contributions[1])
        self._dotbracket = dotbracket

    @requires_state(AppState.JOINED)
    def get_free_energy(self):
        """
        Get the free energy (kcal/mol) of the suggested consensus
        secondary structure.

        Returns
        -------
        free_energy : float
            The free energy.
        
        Notes
        -----
        The total energy of the secondary structure regarding the
        minimization objective is the sum of the free energy and the
        covariance term.
        
        See also
        --------
        get_covariance_energy
        """
        return self._free_energy
    
    @requires_state(AppState.JOINED)
    def get_covariance_energy(self):
        """
        Get the energy of the artificial covariance term (kcal/mol) of
        the suggested consensus secondary structure.

        Returns
        -------
        covariance_energy : float
            The energy of the covariance term.
        
        Notes
        -----
        The total energy of the secondary structure regarding the
        minimization objective is the sum of the free energy and the
        covariance term.
        
        See also
        --------
        get_free_energy
        """
        return self._covariance_energy

    @requires_state(AppState.JOINED)
    def get_dot_bracket(self):
        """
        Get the consensus secondary structure in dot bracket notation.

        Returns
        -------
        dotbracket : str
            The secondary structure in dot bracket notation.
        """
        return self._dotbracket

    @requires_state(AppState.JOINED)
    def get_base_pairs(self, sequence_index=None):
        """
        Get the base pairs from the suggested secondary structure.

        Parameters
        ----------
        sequence_index : int, optional
            By default, the base pairs point to positions in the
            alignment.
            If `sequence_index` is set, the returned base pairs point to
            positions in the given sequence, instead.
            The sequence is specified as index in the alignment.
            For example, if the alignment comprises three sequences,
            `sequence_index` is in range 0-2.

        Returns
        -------
        base_pairs : ndarray, shape=(n,2)
            Each row corresponds to the positions of the bases in the
            alignment.
            If `sequence_index` is set, the positions correspond to the
            given sequence.
        """
        base_pairs = base_pairs_from_dot_bracket(self._dotbracket)
        if sequence_index is not None:
            trace = self._alignment.trace[:, sequence_index]
            # Map base pairs that point to consensus to base pairs that
            # point to given sequence, which is only a subsequence
            # (without gaps) of consensus sequence
            # This is not trivial:
            # The pairs that are not part of the subsequence must be
            # removed and all other pairs need to be shifted
            # To solve this problem a BondList is 'misused', since it
            # is build to solve the same problem on the level of atoms
            # Here the 'bonds' in the BondList are base pairs and the indices
            # are base positions
            pair_list = BondList(len(self._alignment), base_pairs)
            # Remove all pairs that appear in gaps of given sequence
            pair_list = pair_list[trace != -1]
            # Convert back to array of base pairs,
            # remove unused BondType column
            base_pairs = pair_list.as_array()[:,:2]
        return base_pairs

    @staticmethod
    def compute_secondary_structure(alignment, bin_path="RNAalifold"):
        """
        Predict the secondary structure of a ribonucleic acid sequence
        using *ViennaRNA's* *RNAalifold* software.

        This is a convenience function, that wraps the
        :class:`RNAalifoldApp` execution.

        Parameters
        ----------
        alignment : Alignment
            An alignment of RNA sequences.
        bin_path : str, optional
            Path of the *RNAalifold* binary.

        Returns
        -------
        dotbracket : str
            The secondary structure in dot bracket notation.
        free_energy : float
            The free energy.
        covariance_energy : float
            The energy of the covariance term.
        """

        app = RNAalifoldApp(alignment, bin_path=bin_path)
        app.start()
        app.join()
        return (
            app.get_dot_bracket(),
            app.get_free_energy(),
            app.get_covariance_energy()
        )
