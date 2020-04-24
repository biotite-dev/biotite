# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application"
__author__ = "Patrick Kunzmann"
__all__ = ["LocalApp"]

import abc
import copy
import time
import io
from os import chdir, getcwd
from .application import Application, AppState, requires_state
from subprocess import Popen, PIPE, SubprocessError

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
        self._stdout_file_path = None
        self._stdout_file = None
        self._command = None
    
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
        try the `get_command()` method.

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
        AppState.RUNNING | \
        AppState.CANCELLED | \
        AppState.FINISHED | \
        AppState.JOINED
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
            The `Popen` instance
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
        return self._code
    
    @requires_state(AppState.FINISHED | AppState.JOINED)
    def get_stdout(self):
        """
        Get the STDOUT pipe content of the process.
        
        PROTECTED: Do not call from outside.
        
        Returns
        -------
        stdout : str
            The standard outpout.
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
            self._command, stdout=PIPE, stderr=PIPE, encoding="UTF-8"
        )
        chdir(cwd)
    
    def is_finished(self):
        code = self._process.poll()
        if code == None:
            return False
        else:
            self._code = code
            self._stdout, self._stderr = self._process.communicate()
            return True
    
    def wait_interval(self):
        return 0.01
    
    def evaluate(self):
        super().evaluate()
        # Check if applicaion terminated correctly
        exit_code = self.get_exit_code()
        if exit_code != 0:
            err_msg = self.get_stderr().replace("\n", " ")
            raise SubprocessError(
                f"'{self._bin_path}' returned with exit code {exit_code}: "
                f"{err_msg}"
            )
    
    def clean_up(self):
        if self.get_app_state() == AppState.CANCELLED:
            self._process.kill()
