# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["LocalApp"]

import abc
import time
import io
from os import chdir, getcwd
from .application import Application, AppState, requires_state
from subprocess import Popen, PIPE, SubprocessError

class LocalApp(Application, metaclass=abc.ABCMeta):
    """
    The base class for all locally installed applications.
    
    Internally this creates a `Popen` instance, which handles
    the execution.
    
    Parameters
    ----------
    bin_path : str
        Path of the application represented by this class.
    """
    
    def __init__(self, bin_path):
        super().__init__()
        self._bin_path = bin_path
        self._options = []
        self._exec_dir = getcwd()
        self._process = None
        self._stdout_file_path = None
        self._stdout_file = None
    
    @requires_state(AppState.CREATED)
    def set_options(self, options):
        """
        Set command line options for the application run.
        
        PROTECTED: Do not call from outside.
        
        Parameters
        ----------
        options : list
            A list of strings representing the command line options.
        """
        self._options = options
    
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
    
    @requires_state(AppState.JOINED)
    def get_stdout(self):
        return self._stdout
    
    @requires_state(AppState.JOINED)
    def get_stderr(self):
        return self._stderr

    def run(self):
        cwd = getcwd()
        chdir(self._exec_dir) 
        self._process = Popen([self._bin_path] + self._options,
                              stdout=PIPE, stderr=PIPE, encoding="UTF-8") 
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
