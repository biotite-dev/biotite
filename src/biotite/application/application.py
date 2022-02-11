# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application"
__author__ = "Patrick Kunzmann"
__all__ = ["Application", "AppStateError", "TimeoutError", "VersionError",
           "AppState", "requires_state"]

import abc
import time
from functools import wraps
from enum import Flag, auto


class AppState(Flag):
    """
    This enum type represents the app states of an application. 
    """
    CREATED = auto()
    RUNNING = auto()
    FINISHED = auto()
    JOINED = auto()
    CANCELLED = auto()


def requires_state(app_state):
    """
    A decorator for methods of :class:`Application` subclasses that
    raises an :class:`AppStateError` in case the method is called, when
    the :class:`Application` is not in the specified :class:`AppState`
    `app_state`.
    
    Parameters
    ----------
    app_state : AppState
        The required app state.
    
    Examples
    --------
    Raises :class:`AppStateError` when `function` is called,
    if :class:`Application` is not in one of the specified states:
    
    >>> @requires_state(AppState.RUNNING | AppState.FINISHED)
    ... def function(self):
    ...     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # First parameter of method is always 'self'
            instance = args[0]
            if not instance._state & app_state:
                raise AppStateError(
                    f"The application is in {instance.get_app_state()} state, "
                    f"but {app_state} state is required"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class Application(metaclass=abc.ABCMeta):
    """
    This class is a wrapper around an external piece of runnable
    software in any sense. Subclasses of this abstract base class
    specify the respective kind of software and the way of interacting
    with it.
    
    Every :class:`Application` runs through a different app states
    (instances of enum :class:`AppState`) from its creation until its
    termination:
    Directly after its instantiation the app is in the *CREATED* state.
    In this state further parameters can be set for the application run.
    After the user calls the :func:`start()` method, the app state is
    set to *RUNNING* and the :class:`Application` type specific
    :func:`run()` method is called.
    When the application finishes the AppState changes to *FINISHED*.
    This is checked via the :class:`Application` type specific
    :func:`is_finished()` method.
    The user can now call the :func:`join()` method, concluding the
    application in the *JOINED* state and making the results of the
    application accessible by executing the :class:`Application`
    type specific :func:`evaluate()` method.
    Furthermore this executes the :class:`Application` type specific
    :func:`clean_up()` method.
    :func:`join()` can even be called in the *RUNNING* state:
    This will constantly check :func:`is_finished()` and will directly
    go into the *JOINED* state as soon as the application reaches the
    *FINISHED* state.
    Calling the :func:`cancel()` method while the application is
    *RUNNING* or *FINISHED* leaves the application in the *CANCELLED*
    state.
    This triggers the :func:`clean_up()` method, too, but there are no
    accessible results.
    If a method is called in an unsuitable app state, an
    :class:`AppStateError` is called.
    
    The application run behaves like an additional thread: Between the
    call of :func:`start()` and :func:`join()` other Python code can be
    executed, while the application runs in the background.
    """
    
    def __init__(self):
        self._state = AppState.CREATED
    
    @requires_state(AppState.CREATED)
    def start(self):
        """
        Start the application run and set its state to *RUNNING*.
        This can only be done from the *CREATED* state.
        """
        self.run()
        self._start_time = time.time()
        self._state = AppState.RUNNING
    
    @requires_state(AppState.RUNNING | AppState.FINISHED)
    def join(self, timeout=None):
        """
        Conclude the application run and set its state to *JOINED*.
        This can only be done from the *RUNNING* or *FINISHED* state.
        
        If the application is *FINISHED* the joining process happens
        immediately, if otherwise the application is *RUNNING*, this
        method waits until the application is *FINISHED*.
        
        Parameters
        ----------
        timeout : float, optional
            If this parameter is specified, the :class:`Application`
            only waits for finishing until this value (in seconds) runs
            out.
            After this time is exceeded a :class:`TimeoutError` is
            raised and the application is cancelled.
        
        Raises
        ------
        TimeoutError
            If the joining process exceeds the `timeout` value.
        """
        time.sleep(self.wait_interval())
        while self.get_app_state() != AppState.FINISHED:
            if timeout is not None and time.time()-self._start_time > timeout:
                self.cancel()
                raise TimeoutError(
                    f"The application expired its timeout "
                    f"({timeout:.1f} s)"
                )
            else:
                time.sleep(self.wait_interval())
        time.sleep(self.wait_interval())
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
    
    @requires_state(AppState.RUNNING | AppState.FINISHED)
    def cancel(self):
        """
        Cancel the application when in *RUNNING* or *FINISHED* state.
        """
        self._state = AppState.CANCELLED
        self.clean_up()
    
    def get_app_state(self):
        """
        Get the current app state.
        
        Returns
        -------
        app_state : AppState
            The current app state.
        """
        if self._state == AppState.RUNNING:
            if self.is_finished():
                self._state = AppState.FINISHED
        return self._state
    
    @abc.abstractmethod
    def run(self):
        """
        Commence the application run. Called in :func:`start()`.
        
        PROTECTED: Override when inheriting.
        """
        pass
    
    @abc.abstractmethod
    def is_finished(self):
        """
        Check if the application has finished.
        
        PROTECTED: Override when inheriting.
        
        Returns
        -------
        finished : bool
            True of the application has finished, false otherwise
        """
        pass
    
    @abc.abstractmethod
    def wait_interval(self):
        """
        The time interval of :func:`is_finished()` calls in the joining
        process.
        
        PROTECTED: Override when inheriting.
        
        Returns
        -------
        interval : float
            Time (in seconds) between calls of :func:`is_finished()` in
            :func:`join()`
        """
        pass
    
    @abc.abstractmethod
    def evaluate(self):
        """
        Evaluate application results. Called in :func:`join()`.
        
        PROTECTED: Override when inheriting.
        """
        pass
    
    def clean_up(self):
        """
        Do clean up work after the application terminates.
        
        PROTECTED: Optionally override when inheriting.
        """
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


class VersionError(Exception):
    """
    Indicate that the application's version is invalid.
    """
    pass