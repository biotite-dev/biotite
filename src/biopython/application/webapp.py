# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import abc
from .application import Application

__all__ = ["WebApp", "RuleViolationError"]


class WebApp(Application, metaclass=abc.ABCMeta):
    
    def __init__(self, app_url, obey_rules=True):
        super().__init__()
        self._obey_rules = obey_rules
        self._app_url = app_url
    
    def violate_rule(self, msg=None):
        if self._obey_rules:
            if msg is None:
                raise RuleViolationError("The user guidelines "
                                         "would be violated")
            else:
                raise RuleViolationError(msg)
    
    def app_url(self):
        return self._app_url


class RuleViolationError(Exception):
    """
    Indicates that the user guidelines of the web application would be
    violated, if the program continued.
    """
    pass