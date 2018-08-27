# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["WebApp", "RuleViolationError"]

import abc
from .application import Application


class WebApp(Application, metaclass=abc.ABCMeta):
    """
    The base class for all web based applications.
    
    It allows for getting and setting the URL of the app and raises
    an `RuleViolationError` when a subclass calls `violate_rule()`
    (e.g. when the server was contacted too often.)
    
    Be careful, when calling `get_app_state()`. This may involve a
    server contact and therefore frequent calls may raise a
    `RuleViolationError`.
    
    Parameters
    ----------
    app_url : str
        URL of the web app.
    obey_rules : bool, optional
        If true, the application raises an `RuleViolationError`, if
        the server rules are violated. (Default: True)
    """
    
    def __init__(self, app_url, obey_rules=True):
        super().__init__()
        self._obey_rules = obey_rules
        self._app_url = app_url
    
    def violate_rule(self, msg=None):
        """
        Indicate that a server rule was violated, i.e. this raises a
        `RuleViolationError` unless `obey_rules` is false.
        
        PROTECTED: Do not call from outside.
        
        Parameters
        ----------
        msg : str, optional
            A custom message for the `RuleViolationError`.
        """
        if self._obey_rules:
            if msg is None:
                raise RuleViolationError(
                    "The user guidelines would be violated"
                )
            else:
                raise RuleViolationError(msg)
    
    def app_url(self):
        """
        Get the URL of the web app.
        
        Returns
        -------
        url : str
            URL of the web app.
        """
        return self._app_url


class RuleViolationError(Exception):
    """
    Indicates that the user guidelines of the web application would be
    violated, if the program continued.
    """
    pass