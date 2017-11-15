# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import abc
import time
from os import chdir, getcwd
from .application import Application, AppState, requires_state
from subprocess import Popen, DEVNULL

__all__ = ["LocalApp"]

class LocalApp(Application, metaclass=abc.ABCMeta):
    """
    The base class for all locally installed applications.
    
    Internally this creates a `Popen` instance, which handles
    the execution.
    
    Parameters
    ----------
    bin_path : str
        Path of the application represented by this class.
    mute : bool, optional
        If true, the console output of the application goes into
        DEVNULL. (Default: True)
    """
    
    def __init__(self, bin_path, mute=True):
        super().__init__()
        self._bin_path = bin_path
        self._mute = mute
        self._options = []
        self._exec_dir = getcwd()
        self._process = None
    
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

    def run(self):
        cwd = getcwd()
        chdir(self._exec_dir) 
        if self._mute:
            std_out = DEVNULL
            std_err = DEVNULL
        else:
            std_out = None
            std_err = None
        self._process = Popen([self._bin_path] + self._options,
                              stdout=std_out, stderr=std_err) 
        chdir(cwd)
    
    def is_finished(self):
        code = self._process.poll()
        if code == None:
            return False
        else:
            self._code = code
            return True
    
    def wait_interval(self):
        return 0.01
    
    def evaluate(self):
        pass
    
    def clean_up(self):
        if self.get_app_state() == AppState.CANCELLED:
            self._process.kill()
