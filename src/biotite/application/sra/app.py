# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.sra"
__author__ = "Patrick Kunzmann"
__all__ = ["FastqDumpApp"]

import glob
from tempfile import NamedTemporaryFile, gettempdir
from ..localapp import LocalApp, cleanup_tempfile
from ..application import AppState, requires_state
from ...sequence.io.fastq.file import FastqFile
from ...sequence.io.fastq.convert import get_sequences


class FastqDumpApp(LocalApp):
    """
    Fetch sequencing data as FASTQ from the *NCBI sequence read archive*
    (SRA) using *sra-tools*.

    Parameters
    ----------
    uid : str
        A *unique identifier* (UID) of the file to be downloaded.
    output_path_prefix : str, optional
        The prefix of the path to store the downloaded FASTQ file.
        ``.fastq`` is appended to this prefix if the run contains
        a single read per spot.
        ``_1.fastq``, ``_2.fastq``, etc. is appended if it contains
        multiple reads per spot.
        By default, the files are created in a temporary directory and
        deleted after the files have been read.
    bin_path : str, optional
        Path to the ``fasterq-dump`` binary.
    offset : int or {'Sanger', 'Solexa', 'Illumina-1.3', 'Illumina-1.5', 'Illumina-1.8'}, optional
        This value is subtracted from the FASTQ ASCII code to obtain the
        quality score.
        Can either be directly the value, or a string that indicates
        the score format.
    """
    
    def __init__(self, uid, output_path_prefix=None, bin_path="fasterq-dump",
                 offset="Sanger"):
        super().__init__(bin_path)
        self._uid = uid
        self._offset = offset
        if output_path_prefix is None:
            # NamedTemporaryFile is only created to obtain prefix
            # for FASTQ files
            self._out_file = NamedTemporaryFile("r")
            self._prefix = self._out_file.name
        else:
            self._out_file = None
            self._prefix = output_path_prefix

    def run(self):
        self.set_arguments([
            "-o", self._prefix + ".fastq",
            "-t", gettempdir(),
            "-f",
            self._uid
        ])
        super().run()
    
    def evaluate(self):
        super().evaluate()
        self._file_names = (
            # For entries with one read per spot
            glob.glob(self._prefix +   ".fastq") + 
            # For entries with multiple reads per spot
            glob.glob(self._prefix + "_*.fastq")
        )
        # Only load FASTQ files into memory when needed
        self._fastq_files = None
    
    def clean_up(self):
        super().clean_up()
        if self._out_file is not None:
            # This file was only created to reserve a unique file name
            # Now it is not needed anymore
            self._out_file.close()
    
    @requires_state(AppState.JOINED)
    def get_file_paths(self):
        """
        Get the file paths to the downloaded FASTQ files.
        
        Returns
        -------
        paths : list of str
            The file paths to the downloaded files.
        """
        return self._file_names
    
    @requires_state(AppState.JOINED)
    def get_sequences(self):
        """
        Get the sequences and score values from the downloaded file(s).
        
        Returns
        -------
        sequences_and_scores : list of dict (str -> (NucleotideSequence, ndarray))
            This list contains the reads for each spot:
            The first item contains the first read for each spot, the
            second item contains the second read for each spot (if existing),
            etc.
            Each item in the list is a dictionary mapping identifiers to its
            corresponding sequence and score values.
        """
        fastq_files = self.get_fastq()
        return [get_sequences(fastq_file) for fastq_file in fastq_files]

    @requires_state(AppState.JOINED)
    def get_fastq(self):
        """
        Get the `FastqFile` objects from the downloaded file(s).

        Returns
        -------
        fastq_files : list of FastqFile
            This list contains the reads for each spot:
            The first item contains the first read for each spot, the
            second item contains the second read for each spot (if existing),
            etc.
        """
        if self._fastq_files is None:
            self._fastq_files = [
                FastqFile.read(file_name, offset=self._offset)
                for file_name in self.get_file_paths()
            ]
        return self._fastq_files
    
    @staticmethod
    def fetch(uid, output_path_prefix=None, bin_path="fasterq-dump",
              offset="Sanger"):
        """
        Get the sequences and score values belonging to the UID from the
        *NCBI sequence read archive* (SRA).

        Parameters
        ----------
        uid : str
            A *unique identifier* (UID) of the file to be downloaded.
        output_path_prefix : str, optional
            The prefix of the path to store the downloaded FASTQ file.
            ``.fastq`` is appended to this prefix if the run contains
            a single read per spot.
            ``_1.fastq``, ``_2.fastq``, etc. is appended if it contains
            multiple reads per spot.
            By default, the files are created in a temporary directory and
            deleted after the files have been read.
        bin_path : str, optional
            Path to the ``fasterq-dump`` binary.
        offset : int or {'Sanger', 'Solexa', 'Illumina-1.3', 'Illumina-1.5', 'Illumina-1.8'}, optional
            This value is subtracted from the FASTQ ASCII code to obtain the
            quality score.
            Can either be directly the value, or a string that indicates
            the score format.
        
        Returns
        -------
        sequences_and_scores : list of dict (str -> (NucleotideSequence, ndarray))
            This list contains the reads for each spot:
            The first item contains the first read for each spot, the
            second item contains the second read for each spot (if existing),
            etc.
            Each item in the list is a dictionary mapping identifiers to its
            corresponding sequence and score values.
        """
        app = FastqDumpApp(uid, output_path_prefix, bin_path, offset)
        app.start()
        app.join()
        return app.get_sequences()
