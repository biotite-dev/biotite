# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

import textwrap
from collections import OrderedDict
from collections.abc import MutableMapping
import numpy as np
from ...seqtypes import NucleotideSequence
from ....file import TextFile, InvalidFileError

__all__ = ["FastqFile"]


class FastqFile(TextFile, MutableMapping):
    
    def __init__(self, offset, chars_per_line=None):
        super().__init__()
        self._chars_per_line = chars_per_line
        self._entries = OrderedDict()
        self._offset = offset
    
    def read(self, file):
        super().read(file)
        # Remove leading and trailing whitespace in all lines
        self.lines = [line.strip() for line in self.lines]
        # Filter out empty lines
        self.lines = [line for line in self.lines if len(line) != 0]
        self._find_entries()
    
    def get_sequence(self, identifier):
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
        if not isinstance(identifier, str):
            raise IndexError(
                "'FastqFile' only supports identifier strings as keys"
            )
        seq_start, seq_stop, score_start, score_stop \
            = self._entries[identifier]
        # Concatenate sequence string from the score lines
        scores = np.fromstring("".join(
            self.lines[score_start : score_stop]
        ), dtype=np.int8)
        scores -= self._offset
        return scores
        
    def __setitem__(self, identifier, item):
        sequence, scores = item
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
            self.lines += str(sequence)
        else:
            self.lines += textwrap.wrap(
                str(sequence), width=self._chars_per_line
            )
        # Append sequence-score separator
        self.lines += ["+"]
        # Append scores
        score_chars = scores.tobytes().decode("ascii")
        if self._chars_per_line is None:
            self.lines += score_chars
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
    