# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.viennarna"
__author__ = "Tom David Müller"
__all__ = ["RNAalifoldApp"]

import copy
from tempfile import NamedTemporaryFile
import numpy as np
from biotite.application.application import AppState, requires_state
from biotite.application.localapp import LocalApp, cleanup_tempfile
from biotite.application.viennarna.util import build_constraint_string
from biotite.sequence.io.fasta import FastaFile, set_alignment
from biotite.structure.bonds import BondList
from biotite.structure.dotbracket import base_pairs_from_dot_bracket


class RNAalifoldApp(LocalApp):
    """
    Predict the consensus secondary structure from a ribonucleic acid alignment
    using *ViennaRNA's* *RNAalifold* software.

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
        super().__init__(bin_path)
        self._alignment = copy.deepcopy(alignment)
        self._temperature = str(temperature)
        self._constraints = None
        self._enforce = None
        self._in_file = NamedTemporaryFile("w", suffix=".fa", delete=False)
        self._constraints_file = NamedTemporaryFile(
            "w+", suffix=".constraints", delete=False
        )

    def run(self):
        # Insert no line breaks
        # -> Extremely high value for characters per line
        fasta_file = FastaFile(chars_per_line=np.iinfo(np.int32).max)
        set_alignment(
            fasta_file,
            self._alignment,
            seq_names=[str(i) for i in range(len(self._alignment.sequences))],
        )
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
            self._constraints_file.write(self._constraints)
            self._constraints_file.flush()
            self._constraints_file.seek(0)
            self.set_stdin(self._constraints_file)

        self.set_arguments(options + [self._in_file.name])
        super().run()

    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._in_file)
        cleanup_tempfile(self._constraints_file)

    def evaluate(self):
        super().evaluate()
        lines = self.get_stdout().splitlines()
        self._consensus = lines[0].strip()
        result = lines[1]
        dotbracket, total_energy = result.split(" ", maxsplit=1)
        # Energy has the form:
        # (<total> = <free> + <covariance>)
        total_energy = total_energy[1:-1]
        energy_contributions = total_energy.split("=")[1].split("+")
        self._free_energy = float(energy_contributions[0])
        self._covariance_energy = float(energy_contributions[1])
        self._dotbracket = dotbracket

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

        Warnings
        --------
        If a constraint is given for a gap position in the consensus sequence,
        the software may find no base pairs at all.
        """
        self._constraints = build_constraint_string(
            len(self._alignment), pairs, paired, unpaired, downstream, upstream
        )
        self._enforce = enforce

    @requires_state(AppState.JOINED)
    def get_free_energy(self):
        """
        Get the free energy (kcal/mol) of the suggested consensus
        secondary structure.

        Returns
        -------
        free_energy : float
            The free energy.

        See Also
        --------
        get_covariance_energy : Get the energy of the artificial covariance term.

        Notes
        -----
        The total energy of the secondary structure regarding the
        minimization objective is the sum of the free energy and the
        covariance term.
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

        See Also
        --------
        get_free_energy : Get the free energy.

        Notes
        -----
        The total energy of the secondary structure regarding the
        minimization objective is the sum of the free energy and the
        covariance term.
        """
        return self._covariance_energy

    @requires_state(AppState.JOINED)
    def get_consensus_sequence_string(self):
        """
        Get the consensus sequence.

        As the consensus may contain gaps, the sequence is returned as
        string.

        Returns
        -------
        consensus : str
            The consensus sequence.
        """
        return self._consensus

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
            base_pairs = pair_list.as_array()[:, :2]
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
            app.get_covariance_energy(),
        )
