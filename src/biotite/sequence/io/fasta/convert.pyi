from biotite.sequence.align.alignment import Alignment
from biotite.sequence.io.fasta.file import FastaFile
from biotite.sequence.seqtypes import (
    NucleotideSequence,
    ProteinSequence,
)
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)


def _convert_to_sequence(
    seq_str: str
) -> Union[ProteinSequence, NucleotideSequence]: ...


def _convert_to_string(sequence: NucleotideSequence) -> str: ...


def get_alignment(
    fasta_file: FastaFile,
    additional_gap_chars: Tuple[str] = ('_',)
) -> Alignment: ...


def get_sequence(
    fasta_file: FastaFile,
    header: None = None
) -> Union[ProteinSequence, NucleotideSequence]: ...


def get_sequences(
    fasta_file: FastaFile
) -> Dict[str, Union[ProteinSequence, NucleotideSequence]]: ...


def set_alignment(
    fasta_file: FastaFile,
    alignment: Alignment,
    seq_names: List[str]
) -> None: ...


def set_sequence(
    fasta_file: FastaFile,
    sequence: NucleotideSequence,
    header: None = None
) -> None: ...


def set_sequences(
    fasta_file: FastaFile,
    sequence_dict: Dict[str, NucleotideSequence]
) -> None: ...
