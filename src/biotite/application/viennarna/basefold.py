# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application"
__author__ = "Patrick Kunzmann"
__all__ = ["BaseFoldApp"]

import abc
from tempfile import NamedTemporaryFile
from ..localapp import LocalApp, cleanup_tempfile
from ..application import AppState, requires_state


class BaseFoldApp(LocalApp, metaclass=abc.ABCMeta):
    """
    This is an abstract base class for *ViennaRNA* folding applications
    
    It handles file input as well as common options for *ViennaRNA*'s various
    folding tools such as temperature dependence.
    
    Parameters
    ----------
    fasta_file : FastaFile
        The :class:`FastaFile` containing the input sequences or alignment
    temperature : int, optional
        The temperature (°C) to be assumed for the energy parameters.
    bin_path : str, optional
        Path of the tool binary.
    """
    
    def __init__(self, fasta_file, temperature, bin_path):
        super().__init__(bin_path)
        self._temperature = str(temperature)
        self._in_file = NamedTemporaryFile(
            "w+", suffix=".fa", delete=False
        )
        fasta_file.write(self._in_file)
        self._in_file.flush()
        self._in_file.seek(0)
        self.set_stdin(self._in_file)

    def run(self):
        super().run()
    
    @requires_state(AppState.CREATED)
    def set_temperature(self, temperature):
        """
        Adjust the energy parameters according to a temperature in
        degrees Celsius.

        Parameters
        ----------
        temperature : int
            The temperature.
        """
        self._temperature = str(temperature)

    @requires_state(AppState.CREATED)
    def set_arguments(self, arguments):
        base_arguments = ["-T", self._temperature]
        super().set_arguments(base_arguments + arguments)
    
    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._in_file)