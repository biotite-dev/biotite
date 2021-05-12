# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module contains a convenience function for loading sequences from
general sequence files.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["load_sequence", "save_sequence",
           "load_sequences", "save_sequences"]

import itertools
import os.path
import io
from collections import OrderedDict
import numpy as np
from ..seqtypes import NucleotideSequence, ProteinSequence
from ..alphabet import Alphabet


def load_sequence(file_path):
    """
    Load a sequence from a sequence file without the need
    to manually instantiate a :class:`File` object.
    
    Internally this function uses a :class:`File` object, based on the
    file extension.
    
    Parameters
    ----------
    file_path : str
        The path to the sequence file.
    
    Returns
    -------
    sequence : Sequence
        The first sequence in the file.
    """
    # We only need the suffix here
    filename, suffix = os.path.splitext(file_path)
    if suffix in [".fasta", ".fa", ".mpfa", ".fna", ".fsa"]:
        from .fasta import FastaFile, get_sequence
        file = FastaFile.read(file_path)
        return get_sequence(file)
    elif suffix in [".fastq", ".fq"]:
        from .fastq import FastqFile
        # Quality scores are irrelevant for this function
        # -> Offset is irrelevant
        file = FastqFile.read(file_path, offset="Sanger")
        # Get first sequence
        for seq_str, scores in file.values():
            sequence = NucleotideSequence(seq_str)
            break
        return sequence
    elif suffix in [".gb", ".gbk", ".gp"]:
        from .genbank import GenBankFile, get_sequence
        format = "gp" if suffix == ".gp" else "gb"
        file = GenBankFile.read(file_path)
        return get_sequence(file, format)
    else:
        raise ValueError(f"Unknown file format '{suffix}'")


def save_sequence(file_path, sequence):
    """
    Save a sequence into a sequence file without the need
    to manually instantiate a :class:`File` object.
    
    Internally this function uses a :class:`File` object, based on the
    given file extension.
    
    Parameters
    ----------
    file_path : str
        The path to structure file.
    sequence : Sequence
        The sequence to be saved.
    """
    # We only need the suffix here
    filename, suffix = os.path.splitext(file_path)
    if suffix in [".fasta", ".fa", ".mpfa", ".fna", ".fsa"]:
        from .fasta import FastaFile, set_sequence
        file = FastaFile()
        set_sequence(file, sequence)
        file.write(file_path)
    elif suffix in [".fastq", ".fq"]:
        from .fastq import FastqFile
        # Quality scores are irrelevant for this function
        # -> Offset is irrelevant
        file = FastqFile(offset="Sanger")
        # Scores are set to 0 since no score information is supplied
        scores = np.zeros(len(sequence))
        file["sequence"] = str(sequence), scores
        file.write(file_path)
    elif suffix in [".gb", ".gbk", ".gp"]:
        from .genbank import GenBankFile, set_locus, set_sequence
        file = GenBankFile()
        set_locus(file, "sequence", len(sequence))
        set_sequence(file, sequence)
        file.write(file_path)
    else:
        raise ValueError(f"Unknown file format '{suffix}'")


def load_sequences(file_path):
    """
    Load multiple sequences from a sequence file without the need
    to manually instantiate a :class:`File` object.
    
    Internally this function uses a :class:`File` object, based on the
    file extension.
    
    Parameters
    ----------
    file_path : str
        The path to the sequence file.
    
    Returns
    -------
    sequences : dict of (str, Sequence)
        The sequences in the file.
        This dictionary maps each header name to
        the respective sequence. 
    """
    # We only need the suffix here
    filename, suffix = os.path.splitext(file_path)
    if suffix in [".fasta", ".fa", ".mpfa", ".fna", ".fsa"]:
        from .fasta import FastaFile, get_sequences
        file = FastaFile.read(file_path)
        return get_sequences(file)
    elif suffix in [".fastq", ".fq"]:
        from .fastq import FastqFile
        # Quality scores are irrelevant for this function
        # -> Offset is irrelevant
        file = FastqFile.read(file_path, offset="Sanger")
        return {identifier : NucleotideSequence(seq_str)
                for identifier, (seq_str, scores) in file.items()}
    elif suffix in [".gb", ".gbk", ".gp"]:
        from .genbank import MultiFile, get_definition, get_sequence
        file = MultiFile.read(file_path)
        format = "gp" if suffix == ".gp" else "gb"
        sequences = OrderedDict()
        for f in file:
            sequences[get_definition(f)] = get_sequence(f, format)
        return sequences
    else:
        raise ValueError(f"Unknown file format '{suffix}'")


def save_sequences(file_path, sequences):
    """
    Save multiple sequences into a sequence file without the need
    to manually instantiate a :class:`File` object.
    
    Internally this function uses a :class:`File` object, based on the
    given file extension.
    
    Parameters
    ----------
    file_path : str
        The path to structure file.
    sequences : dict of (str, Sequence)
        The sequences to be saved. The dictionary maps a header name
        to asequence.
    """
    # We only need the suffix here
    filename, suffix = os.path.splitext(file_path)
    if suffix in [".fasta", ".fa", ".mpfa", ".fna", ".fsa"]:
        from .fasta import FastaFile, set_sequences
        file = FastaFile()
        set_sequences(file, sequences)
        file.write(file_path)
    elif suffix in [".fastq", ".fq"]:
        from .fastq import FastqFile
        # Quality scores are irrelevant for this function
        # -> Offset is irrelevant
        file = FastqFile(offset="Sanger")
        for identifier, sequence in sequences.items():
            # Scores are set to 0 since no score information is supplied
            scores = np.zeros(len(sequence))
            file["identifer"] = str(sequence), scores
        file.write(file_path)
    elif suffix in [".gb", ".gbk", ".gp"]:
        raise NotImplementedError(
            "Writing GenBank files containing multiple records is currently "
            "not supported"
        )
    else:
        raise ValueError(f"Unknown file format '{suffix}'")
