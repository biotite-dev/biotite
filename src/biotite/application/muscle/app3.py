# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.muscle"
__author__ = "Patrick Kunzmann"
__all__ = ["MuscleApp"]

import numbers
import warnings
from collections.abc import Sequence
from tempfile import NamedTemporaryFile
from biotite.application.application import AppState, VersionError, requires_state
from biotite.application.localapp import cleanup_tempfile, get_version
from biotite.application.msaapp import MSAApp
from biotite.sequence.phylo.tree import Tree


class MuscleApp(MSAApp):
    """
    Perform a multiple sequence alignment using MUSCLE version 3.

    Parameters
    ----------
    sequences : list of Sequence
        The sequences to be aligned.
    bin_path : str, optional
        Path of the MUSCLE binary.
    matrix : SubstitutionMatrix, optional
        A custom substitution matrix.

    See Also
    --------
    Muscle5App : Interface to MUSCLE version ``>=5``.

    Examples
    --------

    >>> seq1 = ProteinSequence("BIQTITE")
    >>> seq2 = ProteinSequence("TITANITE")
    >>> seq3 = ProteinSequence("BISMITE")
    >>> seq4 = ProteinSequence("IQLITE")
    >>> app = MuscleApp([seq1, seq2, seq3, seq4])
    >>> app.start()
    >>> app.join()
    >>> alignment = app.get_alignment()
    >>> print(alignment)
    BIQT-ITE
    TITANITE
    BISM-ITE
    -IQL-ITE
    """

    def __init__(self, sequences, bin_path="muscle", matrix=None):
        major_version = get_version(bin_path, "-version")[0]
        if major_version != 3:
            raise VersionError(f"Muscle 3 is required, got version {major_version}")

        super().__init__(sequences, bin_path, matrix)
        self._gap_open = None
        self._gap_ext = None
        self._terminal_penalty = None
        self._tree1 = None
        self._tree2 = None
        self._out_tree1_file = NamedTemporaryFile("r", suffix=".tree", delete=False)
        self._out_tree2_file = NamedTemporaryFile("r", suffix=".tree", delete=False)

    def run(self):
        args = [
            "-quiet",
            "-in",
            self.get_input_file_path(),
            "-out",
            self.get_output_file_path(),
            "-tree1",
            self._out_tree1_file.name,
            "-tree2",
            self._out_tree2_file.name,
        ]
        if self.get_seqtype() == "protein":
            args += ["-seqtype", "protein"]
        else:
            args += ["-seqtype", "dna"]
        if self.get_matrix_file_path() is not None:
            args += ["-matrix", self.get_matrix_file_path()]
        if self._gap_open is not None and self._gap_ext is not None:
            args += ["-gapopen", f"{self._gap_open:.1f}"]
            args += ["-gapextend", f"{self._gap_ext:.1f}"]
            # When the gap penalty is set,
            # use the penalty also for hydrophobic regions
            args += ["-hydrofactor", "1.0"]
            # Use the recommendation of the documentation
            args += ["-center", "0.0"]
        self.set_arguments(args)
        super().run()

    def evaluate(self):
        super().evaluate()

        newick = self._out_tree1_file.read().replace("\n", "")
        if len(newick) > 0:
            self._tree1 = Tree.from_newick(newick)
        else:
            warnings.warn("MUSCLE did not write a tree file from the first iteration")

        newick = self._out_tree2_file.read().replace("\n", "")
        if len(newick) > 0:
            self._tree2 = Tree.from_newick(newick)
        else:
            warnings.warn("MUSCLE did not write a tree file from the second iteration")

    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._out_tree1_file)
        cleanup_tempfile(self._out_tree2_file)

    @requires_state(AppState.CREATED)
    def set_gap_penalty(self, gap_penalty):
        """
        Set the gap penalty for the alignment.

        Parameters
        ----------
        gap_penalty : float or (tuple, dtype=int)
            If a float is provided, the value will be interpreted as
            general gap penalty.
            If a tuple is provided, an affine gap penalty is used.
            The first value in the tuple is the gap opening penalty,
            the second value is the gap extension penalty.
            The values need to be negative.
        """
        # Check if gap penalty is general or affine
        if isinstance(gap_penalty, numbers.Real):
            if gap_penalty > 0:
                raise ValueError("Gap penalty must be negative")
            self._gap_open = gap_penalty
            self._gap_ext = gap_penalty
        elif isinstance(gap_penalty, Sequence):
            if gap_penalty[0] > 0 or gap_penalty[1] > 0:
                raise ValueError("Gap penalty must be negative")
            self._gap_open = gap_penalty[0]
            self._gap_ext = gap_penalty[1]
        else:
            raise TypeError("Gap penalty must be either float or tuple")

    @requires_state(AppState.JOINED)
    def get_guide_tree(self, iteration="identity"):
        """
        Get the guide tree created for the progressive alignment.

        Parameters
        ----------
        iteration : {'kmer', 'identity'}
            If 'kmer', the first iteration tree is returned.
            This tree uses the sequences common *k*-mers as distance
            measure.
            If 'identity' the second iteration tree is returned.
            This tree uses distances based on the pairwise sequence
            identity after the first progressive alignment iteration.

        Returns
        -------
        tree : Tree
            The guide tree.
        """
        if iteration == "kmer":
            return self._tree1
        elif iteration == "identity":
            return self._tree2
        else:
            raise ValueError("Iteration must be 'kmer' or 'identity'")

    @staticmethod
    def supports_nucleotide():
        return True

    @staticmethod
    def supports_protein():
        return True

    @staticmethod
    def supports_custom_nucleotide_matrix():
        return False

    @staticmethod
    def supports_custom_protein_matrix():
        return True

    @classmethod
    def align(cls, sequences, bin_path=None, matrix=None, gap_penalty=None):
        """
        Perform a multiple sequence alignment.

        This is a convenience function, that wraps the :class:`MuscleApp`
        execution.

        Parameters
        ----------
        sequences : iterable object of Sequence
            The sequences to be aligned.
        bin_path : str, optional
            Path of the MSA software binary. By default, the default path
            will be used.
        matrix : SubstitutionMatrix, optional
            A custom substitution matrix.
        gap_penalty : float or (tuple, dtype=int), optional
            If a float is provided, the value will be interpreted as
            general gap penalty.
            If a tuple is provided, an affine gap penalty is used.
            The first value in the tuple is the gap opening penalty,
            the second value is the gap extension penalty.
            The values need to be negative.

        Returns
        -------
        alignment : Alignment
            The global multiple sequence alignment.
        """
        if bin_path is None:
            app = cls(sequences, matrix=matrix)
        else:
            app = cls(sequences, bin_path, matrix=matrix)
        if gap_penalty is not None:
            app.set_gap_penalty(gap_penalty)
        app.start()
        app.join()
        return app.get_alignment()
