# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.sra"
__author__ = "Patrick Kunzmann"
__all__ = ["FastqDumpApp"]

import glob
from tempfile import NamedTemporaryFile, gettempdir
from ..localapp import LocalApp
from ..application import AppState, requires_state
from ...sequence.io.fastq.file import FastqFile
from ...sequence.io.fastq.convert import get_sequences


class FastqDumpApp(LocalApp):
    """
    """
    
    def __init__(self, uid, output_path_prefix=None, bin_path="fasterq-dump",
                 offset="Sanger"):
        super().__init__(bin_path)
        self._uid = uid
        self._offset = offset
        if output_path_prefix is None:
            self._out_file = NamedTemporaryFile("r")
            self._prefix = self._out_file.name
        else:
            self._prefix = output_path_prefix
            self._out_file = None

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
        file_names = (
            # For entries with one read per spot
            glob.glob(self._prefix +   ".fastq") + 
            # For entries with multiple reads per spot
            glob.glob(self._prefix + "_*.fastq")
        )
        self._fastq_files = [
            FastqFile.read(file_name, offset=self._offset)
            for file_name in file_names
        ]
    
    def clean_up(self):
        super().clean_up()
        if self._out_file is not None:
            self._out_file.close()
    
    @requires_state(AppState.JOINED)
    def get_sequences(self):
        return [get_sequences(fastq_file) for fastq_file in self._fastq_files]

    @requires_state(AppState.JOINED)
    def get_fastq(self):
        return self._fastq_files
    
    @staticmethod
    def fetch(uid, output_path_prefix=None, bin_path="fasterq-dump"):
        app = FastqDumpApp(uid, output_path_prefix, bin_path)
        app.start()
        app.join()
        return app.get_sequences()
