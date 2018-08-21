# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Tuple, Dict, Union, Optional, Iterable
from .file import FastaFile
from ...seqtypes import ProteinSequence, NucleotideSequence
from ...align.alignment import Alignment


def get_sequence(
    fasta_file: FastaFile,
    header: Optional[str] = None
) -> Union[ProteinSequence, NucleotideSequence]: ...

def set_sequence(
    fasta_file: FastaFile,
    sequence: Union[ProteinSequence, NucleotideSequence],
    header: Optional[str] = None
) -> None: ...

def get_sequences(
    fasta_file: FastaFile
) -> Dict[str, Union[ProteinSequence, NucleotideSequence]]: ...

def set_sequences(
    fasta_file: FastaFile,
    sequence_dict: Dict[str, Union[ProteinSequence, NucleotideSequence]]
) -> None: ...

def get_alignment(
    fasta_file: FastaFile,
    additional_gap_chars: Tuple[str] = ('_',)
) -> Alignment: ...

def set_alignment(
    fasta_file: FastaFile,
    alignment: Alignment,
    seq_names: Iterable[str]
) -> None: ...