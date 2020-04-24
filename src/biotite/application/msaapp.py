# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application"
__author__ = "Patrick Kunzmann"
__all__ = ["MSAApp"]

import abc
from tempfile import NamedTemporaryFile
from collections import OrderedDict
import numpy as np
from .localapp import LocalApp
from .application import AppState, requires_state
from ..sequence.sequence import Sequence
from ..sequence.seqtypes import NucleotideSequence, ProteinSequence
from ..sequence.io.fasta.file import FastaFile
from ..sequence.align.alignment import Alignment
from ..sequence.align.matrix import SubstitutionMatrix


class MSAApp(LocalApp, metaclass=abc.ABCMeta):
    """
    This is an abstract base class for multiple sequence alignment
    software.
    
    It handles conversion of :class:`Sequence` objects to FASTA input
    and FASTA output to an :class:`Alignment` object.
    Inheriting subclasses only need to incorporate the file path
    of these FASTA files into the program arguments.

    Furthermore, this class can handle custom substitution matrices,
    if the underlying program supports these.

    MSA software that supports alignment of protein sequences and custom
    substitution matrices, can be used to align exotic, normally
    unsupported sequence types:
    At first the exotic sequences are mapped into protein sequences and
    the custom substitution matrix is converted into a protein sequence
    substitution matrix.
    Then the protein sequences are aligned and finally the protein
    sequences are mapped back into the original sequence types.
    The mapping does not work, when the alphabet of the exotic
    sequences is larger than the amino acid alphabet.
    
    Internally this creates a :class:`Popen` instance, which handles
    the execution.
    
    Parameters
    ----------
    sequences : iterable object of Sequence
        The sequences to be aligned.
    bin_path : str, optional
        Path of the MSA software binary. By default, the default path
        will be used.
    matrix : SubstitutionMatrix, optional
        A custom substitution matrix.
    """
    
    def __init__(self, sequences, bin_path, matrix=None):
        super().__init__(bin_path)
        
        if len(sequences) < 2:
            raise ValueError("At least two sequences are required")
        # Check if all sequences share the same alphabet
        alphabet = sequences[0].get_alphabet()
        for seq in sequences:
            if seq.get_alphabet() != alphabet:
                raise ValueError("Alphabets of the sequences are not equal")
        
        self._matrix = None
        # Check whether the program supports the alignment for the given
        # sequence type
        if ProteinSequence.alphabet.extends(alphabet) \
            and self.supports_protein():
                self._is_mapped = False
                self._seqtype = "protein"
                if matrix is not None:
                    if not self.supports_custom_protein_matrix():
                        raise TypeError(
                            "The software does not support custom "
                            "substitution matrices for protein sequences"
                        )
                    self._matrix = matrix
        elif NucleotideSequence.alphabet_amb.extends(alphabet) \
            and self.supports_nucleotide():
                self._is_mapped = False
                self._seqtype = "nucleotide"
                if matrix is not None:
                    if not self.supports_custom_nucleotide_matrix():
                        raise TypeError(
                            "The software does not support custom "
                            "substitution matrices for nucleotide sequences"
                        )
                    self._matrix = matrix
        else:
            # For all other sequence types, try to map the sequence into
            # a protein sequence
            if not self.supports_protein():
                # Alignment of a custom sequence type requires mapping
                # into a protein sequence
                raise TypeError(
                    f"The software cannot align sequences of type "
                    f"{type(sequences[0]).__name__}: "
                    f"No support for alignment of the mapped sequences"
                )
            if not self.supports_custom_protein_matrix():
                # Alignment of a custom sequence type requires a custom
                # substitution matrix
                raise TypeError(
                    f"The software cannot align sequences of type "
                    f"{type(sequences[0]).__name__}: "
                    f"No support for custom substitution matrices"
                )
            self._is_mapped = True
            self._sequences = sequences
            # Sequence masquerades as protein
            self._seqtype = "protein"
            self._mapped_sequences = MSAApp._map_sequences(sequences, alphabet)
            self._matrix = MSAApp._map_matrix(matrix)

        self._sequences = sequences
        self._in_file     = NamedTemporaryFile("w", suffix=".fa")
        self._out_file    = NamedTemporaryFile("r", suffix=".fa")
        self._matrix_file = NamedTemporaryFile("w", suffix=".mat")

    def run(self):
        sequences = self._sequences if not self._is_mapped \
                    else self._mapped_sequences
        sequences_file = FastaFile()
        for i, seq in enumerate(sequences):
            sequences_file[str(i)] = str(seq)
        sequences_file.write(self._in_file)
        self._in_file.flush()
        if self._matrix is not None:
            self._matrix_file.write(str(self._matrix))
            self._matrix_file.flush()
        super().run()
    
    def evaluate(self):
        super().evaluate()
        alignment_file = FastaFile.read(self._out_file)
        seq_dict = OrderedDict(alignment_file)
        # Get alignment
        out_seq_str = [None] * len(seq_dict)
        for i in range(len(self._sequences)):
            out_seq_str[i] = seq_dict[str(i)]
        trace = Alignment.trace_from_strings(out_seq_str)
        self._alignment = Alignment(self._sequences, trace, None)
        # Also obtain original order
        self._order = np.zeros(len(seq_dict), dtype=int)
        for i, seq_index in enumerate(seq_dict):
             self._order[i] = int(seq_index)
    
    def clean_up(self):
        super().clean_up()
        self._in_file.close()
        self._out_file.close()
        self._matrix_file.close()
    
    @requires_state(AppState.JOINED)
    def get_alignment(self):
        """
        Get the resulting multiple sequence alignment.
        
        Returns
        -------
        alignment : Alignment
            The global multiple sequence alignment.
        """
        return self._alignment
    
    @requires_state(AppState.JOINED)
    def get_alignment_order(self):
        """
        Get the order of the resulting multiple sequence alignment.

        Usually the order of sequences in the output file is
        different from the input file, e.g. the sequences are ordered
        according to the guide tree.
        After running an MSA software, the output sequence order of
        the alignment rearranged so that it is the same as the input
        order.
        This method returns the order of the sequences intended by the
        MSA software.
        
        Returns
        -------
        order : ndarray, dtype=int
            The sequence order intended by the MSA software.
        
        Examples
        --------
        Align sequences and restore the original order:

        app = ClustalOmegaApp(sequences)
        app.start()
        app.join()
        alignment = app.get_alignment()
        order = app.get_alignment_order()
        alignment = alignment[:, order]
        """
        return self._order
    
    def get_input_file_path(self):
        """
        Get input file path (FASTA format).
        
        PROTECTED: Do not call from outside.
        
        Returns
        -------
        path : str
            Path of input file.
        """
        return self._in_file.name
    
    def get_output_file_path(self):
        """
        Get output file path (FASTA format).
        
        PROTECTED: Do not call from outside.
        
        Returns
        -------
        path : str
            Path of output file.
        """
        return self._out_file.name
    
    def get_matrix_file_path(self):
        """
        Get file path for custom substitution matrix.
        
        PROTECTED: Do not call from outside.
        
        Returns
        -------
        path : str or None
            Path of substitution matrix.
            None if no matrix was given.
        """
        return self._matrix_file.name if self._matrix is not None else None
    
    def get_seqtype(self):
        """
        Get the type of aligned sequences.

        When a custom sequence type (neither nucleotide nor protein)
        is mapped onto a protein sequence, the return value is also
        ``'protein'``.
        
        PROTECTED: Do not call from outside.
        
        Returns
        -------
        seqtype : {'nucleotide', 'protein'}
            Type of sequences to be aligned.
        """
        return self._seqtype
    
    @staticmethod
    @abc.abstractmethod
    def supports_nucleotide():
        """
        Check whether this class supports nucleotide sequences for
        alignment.

        Returns
        -------
        support : bool
            True, if the class has support, false otherwise.
        
        PROTECTED: Override when inheriting.
        """
        pass
    
    @staticmethod
    @abc.abstractmethod
    def supports_protein():
        """
        Check whether this class supports nucleotide sequences for
        alignment.

        Returns
        -------
        support : bool
            True, if the class has support, false otherwise.
        
        PROTECTED: Override when inheriting.
        """
        pass
    
    @staticmethod
    @abc.abstractmethod
    def supports_custom_nucleotide_matrix():
        """
        Check whether this class supports custom substitution matrices
        for protein sequence alignment.

        Returns
        -------
        support : bool
            True, if the class has support, false otherwise.
        
        PROTECTED: Override when inheriting.
        """
        pass
    
    @staticmethod
    @abc.abstractmethod
    def supports_custom_protein_matrix():
        """
        Check whether this class supports custom substitution matrices
        for nucleotide sequence alignment.

        Returns
        -------
        support : bool
            True, if the class has support, false otherwise.
        
        PROTECTED: Override when inheriting.
        """
        pass
    
    @staticmethod
    def _map_sequences(sequences, alphabet):
        if len(alphabet) > len(ProteinSequence.alphabet):
            # Cannot map into a protein sequence if the alphabet
            # has more symbols
            raise TypeError(
                f"The software cannot align sequences of type "
                f"{type(sequences[0]).__name__}: "
                f"Alphabet is too large to be converted into amino "
                f"acid alphabet"
            )
        mapped_sequences = []
        for seq in sequences:
            # Mapping is done by simply taking over the sequence
            # code of the original sequence
            prot_seq = ProteinSequence()
            prot_seq.code = seq.code
            mapped_sequences.append(prot_seq)
        return mapped_sequences
    
    @staticmethod
    def _map_matrix(matrix):
        if matrix is None:
            raise TypeError(
                "A substitution matrix must be provided for custom "
                "sequence types"
            )
        if not matrix.is_symmetric():
            raise ValueError(
                "A symmetric matrix is required for "
                "multiple sequence alignments"
            )
        # Create a protein substitution matrix with the values taken
        # from the original matrix
        # All trailing symbols are filled with zeros
        old_length = len(matrix.get_alphabet1())
        new_length = len(ProteinSequence.alphabet)
        new_score_matrix = np.zeros((new_length, new_length))
        new_score_matrix[:old_length, :old_length] = matrix.score_matrix()
        return SubstitutionMatrix(
            ProteinSequence.alphabet, ProteinSequence.alphabet,
            new_score_matrix
        )
    
    @classmethod
    def align(cls, sequences, bin_path=None, matrix=None):
        """
        Perform a multiple sequence alignment.
        
        This is a convenience function, that wraps the :class:`MSAApp`
        execution.
        
        Parameters
        ----------
        sequences : iterable object of Sequence
            The sequences to be aligned
        bin_path : str, optional
            Path of the MSA software binary. By default, the default
            path will be used.
        matrix : SubstitutionMatrix, optional
        A custom substitution matrix.
        
        Returns
        -------
        alignment : Alignment
            The global multiple sequence alignment.
        """
        if bin_path is None:
            app = cls(sequences, matrix=matrix)
        else:
            app = cls(sequences, bin_path, matrix=matrix)
        app.start()
        app.join()
        return app.get_alignment()
