# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import abc
import time
from os import chdir, getcwd
from .application import Application, AppState, requires_state
from subprocess import Popen, DEVNULL

__all__ = ["LocalApp"]

class LocalApp(Application, metaclass=abc.ABCMeta):
    
    def __init__(self, bin_path, mute=True):
        super().__init__()
        self._bin_path = bin_path
        self._mute = mute
        self._options = []
        self._exec_dir = getcwd()
        self._process = None
    
    @requires_state(AppState.CREATED)
    def set_options(self, options):
        self._options = options
    
    @requires_state(AppState.CREATED)
    def set_exec_dir(self, exec_dir):
        self._exec_dir = exec_dir
    
    @requires_state(AppState.RUNNING | AppState.FINISHED)
    def get_process(self):
        return self._process

    def run(self):
        cwd = getcwd()
        chdir(self._exec_dir) 
        if self._mute:
            std_out = DEVNULL
        else:
            std_out = None
        self._process = Popen([self._bin_path] + self._options, stdout=std_out) 
        chdir(cwd)
    
    def is_finished(self):
        code = self._process.poll()
        if code == None:
            return False
        else:
            return True
    
    def wait_interval(self):
        return 0.1
    
    def evaluate(self):
        pass
    
    def clean_up(self):
        if self.get_app_state() == AppState.RUNNING:
            self._process.kill()