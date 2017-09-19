# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import abc
import time
from enum import Flag, auto

__all__ = ["Application", "AppStateError", "AppState"]


class AppState(Flag):
    CREATED = auto()
    RUNNING = auto()
    FINISHED = auto()
    JOINED = auto()
    CANCELLED = auto()


def requires_state(app_state):
    def decorator(func):
        def wrapper(*args):
            instance = args[0]
            if (instance._state & app_state).value == 0:
                raise AppStateError("The application is in {:} state, "
                                    "but {:} state is required".format(
                                        str(instance.get_app_state()),
                                        str(app_state))
                                    )
            return func(*args)
        return wrapper
    return decorator


class Application(metaclass=abc.ABCMeta):
    
    def __init__(self):
        self._state = AppState.CREATED
    
    @requires_state(AppState.CREATED)
    def start(self):
        self.run()
        self._start_time = time.time()
        self._state = AppState.RUNNING
    
    @requires_state(AppState.RUNNING | AppState.FINISHED)
    def join(self, timeout=None):
        time.sleep(self.wait_interval())
        while self.get_app_state() != AppState.FINISHED:
            if timeout is not None and time.time()-self._start_time > timeout:
                raise TimeoutError("The application expired its timeout")
            else:
                time.sleep(self.wait_interval())
        time.sleep(self.wait_interval())
        self.evaluate()
        self._state = AppState.JOINED
        self.clean_up()
    
    @requires_state(AppState.RUNNING | AppState.FINISHED)
    def cancel(self):
        self._state = AppState.CANCELLED
        self.clean_up()
    
    def get_app_state(self):
        if self._state == AppState.RUNNING:
            if self.is_finished():
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
    
    def clean_up(self):
        pass


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
