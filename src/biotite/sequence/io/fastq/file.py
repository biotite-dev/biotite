# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.io.fastq"
__author__ = "Patrick Kunzmann"

from numbers import Integral
import textwrap
from collections import OrderedDict
from collections.abc import MutableMapping
import numpy as np
from ...seqtypes import NucleotideSequence
from ....file import TextFile, InvalidFileError

__all__ = ["FastqFile"]


_OFFSETS = {
    "Sanger"       : 33,
    "Solexa"       : 64,
    "Illumina-1.3" : 64,
    "Illumina-1.5" : 64,
    "Illumina-1.8" : 33,
}


class FastqFile(TextFile, MutableMapping):
    """
    This class represents a file in FASTQ format.

    A FASTQ file stores one or multiple sequences (base calls) along
    with sequencing quality scores.
    Each sequence is associated with an identifer string,
    beginning with an ``@``.

    The quality scores are encoded as ASCII characters,
    with each actual score being the ASCII code subtracted by an
    `offset` value.
    The offset is format dependent.
    As the offset is not reliably deducible from the file contets, it
    must be provided explicitly, either as number or format
    (e.g. ``'Illumina-1.8'``).

    Similar to the :class:`FastaFile` class, this class implements the
    :class:`MutableMapping` interface:
    An identifier string (without the leading ``@``) is used as index
    to get and set the corresponding sequence and quality.
    ``del`` removes an entry in the file.
    
    Parameters
    ----------
    offset : int or {'Sanger', 'Solexa', 'Illumina-1.3', 'Illumina-1.5', 'Illumina-1.8'}
        This value that is added to the quality score to obtain the
        ASCII code.
        Can either directly the value, or a string that indicates
        the score format.
    chars_per_line : int, optional
        The number characters in a line containing sequence data
        after which a line break is inserted.
        Only relevant, when adding sequences to a file.
        By default each sequence (and score string)
        is put into one line.
    
    Examples
    --------
    
    >>> import os.path
    >>> file = FastqFile(offset="Sanger")
    >>> file["seq1"] = NucleotideSequence("ATACT"), [0,3,10,7,12]
    >>> file["seq2"] = NucleotideSequence("TTGTAGG"), [15,13,24,21,28,38,35]
    >>> print(file)
    @seq1
    ATACT
    +
    !$+(-
    @seq2
    TTGTAGG
    +
    0.96=GD
    >>> sequence, scores = file["seq1"]
    >>> print(sequence)
    ATACT
    >>> print(scores)
    [ 0  3 10  7 12]
    >>> del file["seq1"]
    >>> print(file)
    @seq2
    TTGTAGG
    +
    0.96=GD
    >>> file.write(os.path.join(path_to_directory, "test.fastq"))
    """
    
    def __init__(self, offset, chars_per_line=None):
        super().__init__()
        self._chars_per_line = chars_per_line
        self._entries = OrderedDict()
        if isinstance(offset, Integral):
            self._offset = offset
        elif isinstance(offset, str):
            self._offset = _OFFSETS[offset]
        else:
            raise TypeError(
                f"The offset must be either an integer or a string "
                f"indicating the format, not {type(offset).__name__}"
            )
    
    def read(self, file):
        super().read(file)
        # Remove leading and trailing whitespace in all lines
        self.lines = [line.strip() for line in self.lines]
        # Filter out empty lines
        self.lines = [line for line in self.lines if len(line) != 0]
        if len(self.lines) == 0:
            raise InvalidFileError("File is empty")
        self._find_entries()
    
    def get_sequence(self, identifier):
        """
        Get the sequence for the specified identifier.

        Parameters
        ----------
        identifier : str
            The identifier of the sequence.
        
        Returns
        -------
        sequence : NucleotideSequence
            The sequence corresponding to the identifier.
        """
        if not isinstance(identifier, str):
            raise IndexError(
                "'FastqFile' only supports identifier strings as keys"
            )
        seq_start, seq_stop, score_start, score_stop \
            = self._entries[identifier]
        # Concatenate sequence string from the sequence lines
        sequence = NucleotideSequence("".join(
            self.lines[seq_start : seq_stop]
        ))
        return sequence
    
    def get_quality(self, identifier):
        """
        Get the quality scores for the specified identifier.

        Parameters
        ----------
        identifier : str
            The identifier of the quality scores.
        
        Returns
        -------
        sequence : NucleotideSequence
            The quality scores corresponding to the identifier.
        """
        if not isinstance(identifier, str):
            raise IndexError(
                "'FastqFile' only supports identifier strings as keys"
            )
        seq_start, seq_stop, score_start, score_stop \
            = self._entries[identifier]
        # Concatenate sequence string from the score lines
        scores = np.frombuffer(
            bytearray(
                "".join(self.lines[score_start : score_stop]), encoding="ascii"
            ),
            dtype=np.int8
        )
        scores -= self._offset
        return scores
        
    def __setitem__(self, identifier, item):
        sequence, scores = item
        if len(sequence) != len(scores):
            raise ValueError(
                f"Sequence has length {len(sequence)}, "
                f"but score length is {len(scores)}"
            )
        if not isinstance(identifier, str):
            raise IndexError(
                "'FastqFile' only supports header strings as keys"
            )
        if not isinstance(sequence, NucleotideSequence):
            raise IndexError("Can only set 'NucleotideSequence' objects")
        # Delete lines of entry corresponding to the identifier,
        # if already existing
        if identifier in self:
            del self[identifier]
        # Append identifier line
        self.lines += ["@" + identifier.replace("\n","").strip()]
        # Append new lines with sequence string (with line breaks)
        if self._chars_per_line is None:
            self.lines.append(str(sequence))
        else:
            self.lines += textwrap.wrap(
                str(sequence), width=self._chars_per_line
            )
        # Append sequence-score separator
        self.lines += ["+"]
        # Append scores
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        scores = scores + self._offset
        score_chars = scores.astype(np.int8, copy=False) \
                            .tobytes() \
                            .decode("ascii")
        if self._chars_per_line is None:
            self.lines.append(score_chars)
        else:
            self.lines += textwrap.wrap(
                score_chars, width=self._chars_per_line
            )
        self._find_entries()
    
    def __getitem__(self, identifier):
        return self.get_sequence(identifier), self.get_quality(identifier)
    
    def __delitem__(self, identifier):
        seq_start, seq_stop, score_start, score_stop \
            = self._entries[identifier]
        del self.lines[seq_start-1 : score_stop]
        del self._entries[identifier]
        self._find_entries()
    
    def __len__(self):
        return len(self._entries)
    
    def __iter__(self):
        return self._entries.__iter__()
    
    def __contains__(self, identifer):
        return identifer in self._entries
    
    def _find_entries(self):
        self._entries = OrderedDict()
        in_sequence = False
        # Record if the parser is currently in a quality score section,
        # as the '@' character at the start of a line may also be a
        # score instead of the start of an identifier
        in_scores = False
        seq_len = 0
        score_len = 0
        seq_start_i = None
        seq_stop_i = None
        score_start_i = None
        score_stop_i = None
        identifier = None
        for i, line in enumerate(self.lines):
            if not in_scores and not in_sequence and line[0] == "@":
                # Identifier line
                identifier = line[1:]
                seq_start_i = i+1
                # Next line is sequence
                in_sequence = True
                # Reset
                seq_len = 0
                score_len = 0
            elif in_sequence:
                if line[0] == "+":
                    # End of sequence start of scores
                    in_sequence = False
                    in_scores = True
                    seq_stop_i = i
                    score_start_i = i+1
                else:
                    # Still in sequence
                    seq_len += len(line)
            elif in_scores:
                score_len += len(line)
                if score_len < seq_len:
                    # Scores have not ended yet
                    pass
                elif score_len == seq_len:
                    # End of scores
                    # -> End of entry
                    score_stop_i = i + 1
                    in_scores = False
                    # Record this entry
                    self._entries[identifier] = (
                        seq_start_i, seq_stop_i, score_start_i, score_stop_i
                    )
                else: # score_len > seq_len
                    raise InvalidFileError(
                        f"The amount of scores is not equal to the sequence "
                        f"length for the sequence in line {seq_start_i+1} "
                    )
            else:
                raise InvalidFileError(f"Line {i+1} in FASTQ file is invalid")
        # At the end of the file, the last sequence or score block
        # must have properly ended
        if in_sequence or in_scores:
            raise InvalidFileError("The last entry in the file is incomplete")