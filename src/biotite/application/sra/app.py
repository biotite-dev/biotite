# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.sra"
__author__ = "Patrick Kunzmann"
__all__ = ["FastaDumpApp", "FastqDumpApp"]

import abc
import glob
from os.path import join
from subprocess import PIPE, Popen, SubprocessError, TimeoutExpired
from tempfile import TemporaryDirectory
from biotite.application.application import (
    Application,
    AppState,
    AppStateError,
    requires_state,
)
from biotite.sequence.io.fasta.convert import get_sequences
from biotite.sequence.io.fasta.file import FastaFile
from biotite.sequence.io.fastq.convert import get_sequences as get_sequences_and_scores
from biotite.sequence.io.fastq.file import FastqFile
from biotite.sequence.seqtypes import NucleotideSequence


# Do not use LocalApp, as two programs are executed
class _DumpApp(Application, metaclass=abc.ABCMeta):
    """
    Fetch sequencing data from the *NCBI sequence read archive*
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
    prefetch_path, fasterq_dump_path : str, optional
        Path to the ``prefetch_path`` and ``fasterq-dump`` binary,
        respectively.
    """

    def __init__(
        self,
        uid,
        output_path_prefix=None,
        prefetch_path="prefetch",
        fasterq_dump_path="fasterq-dump",
    ):
        super().__init__()
        self._prefetch_path = prefetch_path
        self._fasterq_dump_path = fasterq_dump_path
        self._uid = uid
        self._sra_dir = TemporaryDirectory(suffix="_sra")
        if output_path_prefix is None:
            self._prefix = join(self._sra_dir.name, self._uid)
        else:
            self._prefix = output_path_prefix
        self._prefetch_process = None
        self._fasterq_dump_process = None

    @requires_state(AppState.RUNNING | AppState.FINISHED)
    def join(self, timeout=None):
        # Override method as repetitive calls of 'is_finished()'
        # are not necessary as 'communicate()' already waits for the
        # finished application
        try:
            _, self._stderr = self._process.communicate(timeout=timeout)
        except TimeoutExpired:
            self.cancel()
            raise TimeoutError(f"The application expired its timeout ({timeout:.1f} s)")
        self._state = AppState.FINISHED

        try:
            self.evaluate()
        except AppStateError:
            raise
        except:
            self._state = AppState.CANCELLED
            raise
        else:
            self._state = AppState.JOINED
        self.clean_up()

    def run(self):
        # Prefetch into a temp directory with file name equaling UID
        # This ensures that the ID in the header is not the temp prefix
        sra_file_name = join(self._sra_dir.name, self._uid)
        command = (
            f"{self._prefetch_path} -q -O {self._sra_dir.name} "
            f"{self.get_prefetch_options()} {self._uid}; "
            f"{self._fasterq_dump_path} -q -o {self._prefix}.fastq "
            f"{self.get_fastq_dump_options()} {sra_file_name}"
        )
        self._process = Popen(
            command, stdout=PIPE, stderr=PIPE, shell=True, encoding="UTF-8"
        )

    def is_finished(self):
        code = self._process.poll()
        if code is None:
            return False
        else:
            _, self._stderr = self._process.communicate()
            return True

    def evaluate(self):
        super().evaluate()
        # Check if applicaion terminated correctly
        exit_code = self._process.returncode
        if exit_code != 0:
            err_msg = self._stderr.replace("\n", " ")
            raise SubprocessError(
                f"'prefetch' or 'fasterq-dump' returned with exit code "
                f"{exit_code}: {err_msg}"
            )

        self._file_names = (
            # For entries with one read per spot
            glob.glob(self._prefix + ".fastq")
            +
            # For entries with multiple reads per spot
            glob.glob(self._prefix + "_*.fastq")
        )
        # Only load FASTQ files into memory when needed
        self._fastq_files = None

    def wait_interval(self):
        # Not used in this implementation of 'join()'
        raise NotImplementedError()

    def clean_up(self):
        if self.get_app_state() == AppState.CANCELLED:
            self._process.kill()
        # Directory with temp files does not need to be deleted,
        # as temp dir is automatically deleted upon object destruction

    @requires_state(AppState.CREATED)
    def get_prefetch_options(self):
        """
        Get additional options for the `prefetch` call.

        PROTECTED: Override when inheriting.

        Returns
        -------
        options: str
            The additional options.
        """
        return ""

    @requires_state(AppState.CREATED)
    def get_fastq_dump_options(self):
        """
        Get additional options for the `fasterq-dump` call.

        PROTECTED: Override when inheriting.

        Returns
        -------
        options: str
            The additional options.
        """
        return ""

    @requires_state(AppState.JOINED)
    def get_file_paths(self):
        """
        Get the file paths to the downloaded files.

        Returns
        -------
        paths : list of str
            The file paths to the downloaded files.
        """
        return self._file_names

    @requires_state(AppState.JOINED)
    @abc.abstractmethod
    def get_sequences(self):
        """
        Get the sequences from the downloaded file(s).

        Returns
        -------
        sequences : list of dict (str -> NucleotideSequence)
            This list contains the reads for each spot:
            The first item contains the first read for each spot, the
            second item contains the second read for each spot (if existing),
            etc.
            Each item in the list is a dictionary mapping identifiers to its
            corresponding sequence.
        """
        pass


class FastqDumpApp(_DumpApp):
    """
    Fetch sequencing data from the *NCBI sequence read archive*
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
    prefetch_path, fasterq_dump_path : str, optional
        Path to the ``prefetch_path`` and ``fasterq-dump`` binary,
        respectively.
    offset : int or {'Sanger', 'Solexa', 'Illumina-1.3', 'Illumina-1.5', 'Illumina-1.8'}, optional
        This value is subtracted from the FASTQ ASCII code to obtain the
        quality score.
        Can either be directly the value, or a string that indicates
        the score format.
    """

    def __init__(
        self,
        uid,
        output_path_prefix=None,
        prefetch_path="prefetch",
        fasterq_dump_path="fasterq-dump",
        offset="Sanger",
    ):
        super().__init__(uid, output_path_prefix, prefetch_path, fasterq_dump_path)
        self._offset = offset
        self._fastq_files = None

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

    @requires_state(AppState.JOINED)
    def get_sequences(self):
        return [
            {
                header: NucleotideSequence(seq_str.replace("U", "T").replace("X", "N"))
                for header, (seq_str, _) in fastq_file.items()
            }
            for fastq_file in self.get_fastq()
        ]

    @requires_state(AppState.JOINED)
    def get_sequences_and_scores(self):
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
        return [get_sequences_and_scores(fastq_file) for fastq_file in self.get_fastq()]

    @classmethod
    def fetch(
        cls,
        uid,
        output_path_prefix=None,
        prefetch_path="prefetch",
        fasterq_dump_path="fasterq-dump",
        offset="Sanger",
    ):
        """
        Get the sequences belonging to the UID from the
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
        prefetch_path, fasterq_dump_path : str, optional
            Path to the ``prefetch_path`` and ``fasterq-dump`` binary,
            respectively.
        offset : int or {'Sanger', 'Solexa', 'Illumina-1.3', 'Illumina-1.5', 'Illumina-1.8'}, optional
            This value is subtracted from the FASTQ ASCII code to obtain the
            quality score.
            Can either be directly the value, or a string that indicates
            the score format.

        Returns
        -------
        sequences : list of dict (str -> NucleotideSequence)
            This list contains the reads for each spot:
            The first item contains the first read for each spot, the
            second item contains the second read for each spot (if existing),
            etc.
            Each item in the list is a dictionary mapping identifiers to its
            corresponding sequence.
        """
        app = cls(uid, output_path_prefix, prefetch_path, fasterq_dump_path, offset)
        app.start()
        app.join()
        return app.get_sequences()


class FastaDumpApp(_DumpApp):
    """
    Fetch sequencing data from the *NCBI sequence read archive*
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
    prefetch_path, fasterq_dump_path : str, optional
        Path to the ``prefetch_path`` and ``fasterq-dump`` binary,
        respectively.
    """

    def __init__(
        self,
        uid,
        output_path_prefix=None,
        prefetch_path="prefetch",
        fasterq_dump_path="fasterq-dump",
    ):
        super().__init__(uid, output_path_prefix, prefetch_path, fasterq_dump_path)
        self._fasta_files = None

    @requires_state(AppState.CREATED)
    def get_prefetch_options(self):
        return
        # TODO: Use '--eliminate-quals'
        # when https://github.com/ncbi/sra-tools/issues/883 is resolved
        # return "--eliminate-quals"

    @requires_state(AppState.CREATED)
    def get_fastq_dump_options(self):
        return "--fasta"

    @requires_state(AppState.JOINED)
    def get_fasta(self):
        """
        Get the `FastaFile` objects from the downloaded file(s).

        Returns
        -------
        fasta_files : list of FastaFile
            This list contains the reads for each spot:
            The first item contains the first read for each spot, the
            second item contains the second read for each spot (if existing),
            etc.
        """
        if self._fasta_files is None:
            self._fasta_files = [
                FastaFile.read(file_name) for file_name in self.get_file_paths()
            ]
        return self._fasta_files

    @requires_state(AppState.JOINED)
    def get_sequences(self):
        return [get_sequences(fasta_file) for fasta_file in self.get_fasta()]

    @classmethod
    def fetch(
        cls,
        uid,
        output_path_prefix=None,
        prefetch_path="prefetch",
        fasterq_dump_path="fasterq-dump",
    ):
        """
        Get the sequences belonging to the UID from the
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
        prefetch_path, fasterq_dump_path : str, optional
            Path to the ``prefetch_path`` and ``fasterq-dump`` binary,
            respectively.

        Returns
        -------
        sequences : list of dict (str -> NucleotideSequence)
            This list contains the reads for each spot:
            The first item contains the first read for each spot, the
            second item contains the second read for each spot (if existing),
            etc.
            Each item in the list is a dictionary mapping identifiers to its
            corresponding sequence.
        """
        app = cls(uid, output_path_prefix, prefetch_path, fasterq_dump_path)
        app.start()
        app.join()
        return app.get_sequences()
