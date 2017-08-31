# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import abc

__all__ = ["Application", "ApplicationError"]


class Application(metaclass=abc.ABCMeta):
    
    def __init__(self):
        self._is_joined = False
        self._is_started = False
    
    @abc.abstractmethod
    def run(self):
        self._is_started = True
    
    @abc.abstractmethod
    def join(self):
        self._is_joined = True
    
    @abc.abstractmethod
    def is_finished(self):
        pass
    
    def is_started(self):
        return self._is_started


def evaluation(func):
    def wrapper(*args):
        instance = args[0]
        if not instance._is_joined:
            raise EvaluationError("Cannot access run results yet, "
                                  "join first")
        return func(*args)
    return wrapper


class ApplicationError(Exception):
    """
    Indicates that the application lifecycle was violated:
    Either it was tried to join the application, before it was started,
    or it was tried to access the application results,
    before the application run was finished and joined.
    """