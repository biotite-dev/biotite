# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import abc
import time
from enum import IntEnum

__all__ = ["Application", "AppStateError", "AppState"]


class AppState(IntEnum):
    CREATED = auto()
    RUNNING = auto()
    FINISHED = auto()
    CANCELLED = auto()


class Application(metaclass=abc.ABCMeta):
    
    def __init__(self):
        self._state = AppState.CREATED
    
    @requires_state(AppState.CREATED)
    def start(self):
        self.run()
        self._start_time = time.time()
        self._state = AppState.RUNNING
    
    @requires_state(AppState.RUNNING)
    def join(self, timeout=None):
        time.sleep(wait_interval())
        while self.get_app_state() != AppState.FINISHED:
            if timeout is not None and time.time()-self._start_time > timeout:
                raise TimeoutError("The application expired its timeout")
            else:
                time.sleep(wait_interval())
        time.sleep(wait_interval())
        self.evaluate()
        self._state = AppState.FINISHED
        self.tidy_up()
    
    @requires_state(AppState.RUNNING)
    def cancel(self):
        self._state = AppState.CANCELLED
        self.tidy_up()
    
    def get_app_state(self):
        if self._state == AppState.RUNNING:
            if is_finished():
                self._state = AppState.FINISHED
        return self._state
    
    @abc.abstractmethod
    def run(self):
        pass
    
    @abc.abstractmethod
    def is_finished(self):
        pass
    
    @abc.abstractmethod
    def wait_interval(self):
        pass
    
    @abc.abstractmethod
    def evaluate(self):
        pass
    
    def tidy_up(self):
        pass 


def requires_state(app_state):
    def decorator(func):
        def wrapper(*args):
            instance = args[0]
            if instance.get_app_state() != app_state:
                raise AppStateError("The application object is in {:}, "
                                    "but the method requires {:}".format(
                                        instance.get_app_state(), app_state)
                                    )
        return wrapper
    return decorator


class AppStateError(Exception):
    """
    Indicate that the application lifecycle was violated.
    """
    pass


class TimeoutError(Exception):
    """
    Indicate that the application's timeout expired.
    """
    pass
