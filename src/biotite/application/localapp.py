# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application"
__author__ = "Patrick Kunzmann"
__all__ = ["LocalApp"]

import abc
import copy
import re
import subprocess
from os import chdir, getcwd, remove
from pathlib import Path
from subprocess import PIPE, Popen, SubprocessError, TimeoutExpired
from biotite.application.application import (
    Application,
    AppState,
    AppStateError,
    requires_state,
)


class LocalApp(Application, metaclass=abc.ABCMeta):
    """
    The base class for all locally installed applications, that are used
    via the command line.

    Internally this creates a :class:`Popen` instance, which handles
    the execution.

    Parameters
    ----------
    bin_path : str
        Path of the application represented by this class.
    """

    def __init__(self, bin_path):
        super().__init__()
        self._bin_path = bin_path
        self._arguments = []
        self._options = []
        self._exec_dir = getcwd()
        self._process = None
        self._command = None
        self._stdin_file = None

    @requires_state(AppState.CREATED)
    def set_arguments(self, arguments):
        """
        Set command line arguments for the application run.

        PROTECTED: Do not call from outside.

        Parameters
        ----------
        arguments : list of str
            A list of strings representing the command line options.
        """
        self._arguments = copy.copy(arguments)

    @requires_state(AppState.CREATED)
    def set_stdin(self, file):
        """
        Set a file as standard input for the application run.

        PROTECTED: Do not call from outside.

        Parameters
        ----------
        file : file object
            The file for the standard input.
            Must have a valid file descriptor, e.g. file-like objects
            such as `StringIO` are invalid.
        """
        self._stdin_file = file

    @requires_state(AppState.CREATED)
    def add_additional_options(self, options):
        """
        Add additional options for the command line program.
        These options are put before the arguments automatically
        determined by the respective :class:`LocalApp` subclass.

        This method is focused on advanced users, who have knowledge on
        the available options of the command line program and the
        options already used by the :class:`LocalApp` subclasses.
        Ignoring the already used options may result in conflicting
        CLI arguments and potential unexpected results.
        It is recommended to use this method only, when the respective
        :class:`LocalApp` subclass does not provide a method to set the
        desired option.

        Parameters
        ----------
        options : list of str
            A list of strings representing the command line options.

        Notes
        -----
        In order to see which options the command line execution used,
        try the :meth:`get_command()` method.

        Examples
        --------

        >>> seq1 = ProteinSequence("BIQTITE")
        >>> seq2 = ProteinSequence("TITANITE")
        >>> seq3 = ProteinSequence("BISMITE")
        >>> seq4 = ProteinSequence("IQLITE")
        >>> # Run application without additional arguments
        >>> app = ClustalOmegaApp([seq1, seq2, seq3, seq4])
        >>> app.start()
        >>> app.join()
        >>> print(app.get_command())
        clustalo --in ...fa --out ...fa --force --output-order=tree-order --seqtype Protein --guidetree-out ...tree
        >>> # Run application with additional argument
        >>> app = ClustalOmegaApp([seq1, seq2, seq3, seq4])
        >>> app.add_additional_options(["--full"])
        >>> app.start()
        >>> app.join()
        >>> print(app.get_command())
        clustalo --full --in ...fa --out ...fa --force --output-order=tree-order --seqtype Protein --guidetree-out ...tree
        """
        self._options += options

    @requires_state(
        AppState.RUNNING | AppState.CANCELLED | AppState.FINISHED | AppState.JOINED
    )
    def get_command(self):
        """
        Get the executed command.

        Cannot be called until the application has been started.

        Returns
        -------
        command : str
            The executed command.

        Examples
        --------

        >>> seq1 = ProteinSequence("BIQTITE")
        >>> seq2 = ProteinSequence("TITANITE")
        >>> seq3 = ProteinSequence("BISMITE")
        >>> seq4 = ProteinSequence("IQLITE")
        >>> app = ClustalOmegaApp([seq1, seq2, seq3, seq4])
        >>> app.start()
        >>> print(app.get_command())
        clustalo --in ...fa --out ...fa --force --output-order=tree-order --seqtype Protein --guidetree-out ...tree
        """
        return " ".join(self._command)

    @requires_state(AppState.CREATED)
    def set_exec_dir(self, exec_dir):
        """
        Set the directory where the application should be executed.
        If not set, it will be executed in the working directory at the
        time the application was created.

        PROTECTED: Do not call from outside.

        Parameters
        ----------
        exec_dir : str
            The execution directory.
        """
        self._exec_dir = exec_dir

    @requires_state(AppState.RUNNING | AppState.FINISHED)
    def get_process(self):
        """
        Get the `Popen` instance.

        PROTECTED: Do not call from outside.

        Returns
        -------
        process : Popen
            The `Popen` instance.
        """
        return self._process

    @requires_state(AppState.FINISHED | AppState.JOINED)
    def get_exit_code(self):
        """
        Get the exit code of the process.

        PROTECTED: Do not call from outside.

        Returns
        -------
        code : int
            The exit code.
        """
        return self._process.returncode

    @requires_state(AppState.FINISHED | AppState.JOINED)
    def get_stdout(self):
        """
        Get the STDOUT pipe content of the process.

        PROTECTED: Do not call from outside.

        Returns
        -------
        stdout : str
            The standard output.
        """
        return self._stdout

    @requires_state(AppState.FINISHED | AppState.JOINED)
    def get_stderr(self):
        """
        Get the STDERR pipe content of the process.

        PROTECTED: Do not call from outside.

        Returns
        -------
        stdout : str
            The standard error.
        """
        return self._stderr

    def run(self):
        cwd = getcwd()
        chdir(self._exec_dir)
        self._command = [self._bin_path] + self._options + self._arguments
        self._process = Popen(
            self._command,
            stdin=self._stdin_file,
            stdout=PIPE,
            stderr=PIPE,
            encoding="UTF-8",
        )
        chdir(cwd)

    def is_finished(self):
        code = self._process.poll()
        if code is None:
            return False
        else:
            self._stdout, self._stderr = self._process.communicate()
            return True

    @requires_state(AppState.RUNNING | AppState.FINISHED)
    def join(self, timeout=None):
        # Override method as repetitive calls of 'is_finished()'
        # are not necessary as 'communicate()' already waits for the
        # finished application
        try:
            self._stdout, self._stderr = self._process.communicate(timeout=timeout)
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

    def wait_interval(self):
        # Not used in this implementation of 'join()'
        raise NotImplementedError()

    def evaluate(self):
        super().evaluate()
        # Check if applicaion terminated correctly
        exit_code = self.get_exit_code()
        if exit_code != 0:
            err_msg = self.get_stderr().replace("\n", " ")
            raise SubprocessError(
                f"'{self._bin_path}' returned with exit code {exit_code}: {err_msg}"
            )

    def clean_up(self):
        if self.get_app_state() == AppState.CANCELLED:
            self._process.kill()


def cleanup_tempfile(temp_file):
    """
    Close a :class:`NamedTemporaryFile` and delete it manually,
    if `delete` is set to ``False``.
    This function is a small helper function intended for usage in
    `LocalApp` subclasses.

    The manual deletion is necessary, as Windows does not allow to open
    a :class:`NamedTemporaryFile` as second time
    (e.g. by the file name), if `delete` is set to ``True``.

    Parameters
    ----------
    temp_file : NamedTemporaryFile
        The temporary file to be closed and deleted.
    """
    temp_file.close()
    try:
        remove(temp_file.name)
    except FileNotFoundError:
        # File was already deleted, e.g. due to `TemporaryFile(delete=True)`
        pass


def get_version(bin_path, version_option="--version"):
    """
    Get the version of a locally installed application.

    Parameters
    ----------
    bin_path : str or Path
        Path of the application.
    version_option : str, optional
        The command line option to get the version.

    Returns
    -------
    major, minor : int
        The major and minor version number.
    """
    output = subprocess.run(
        [bin_path, version_option], capture_output=True, text=True
    ).stdout
    # Find matches for version string containing major and minor version
    match = re.search(r"\d+\.\d+", output)
    if match is None:
        raise subprocess.SubprocessError(
            f"Could not determine '{Path(bin_path).name}' version "
            f"from the string '{output}'"
        )
    version_string = match.group(0)
    splitted = version_string.split(".")
    return int(splitted[0]), int(splitted[1])
