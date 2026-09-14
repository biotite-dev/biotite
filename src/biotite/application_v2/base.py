# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2"
__author__ = "Patrick Kunzmann"
__all__ = [
    "Application",
    "Future",
    "CompositeFuture",
    "PostprocessFuture",
    "VersionError",
    "CancelledError",
]

import abc
import time
from collections.abc import Callable
from concurrent.futures import CancelledError
from typing import Any, Generic, TypeVar, TypeVarTuple, Unpack, cast

T = TypeVar("T")
U = TypeVar("U")
Ts = TypeVarTuple("Ts")


class Application(abc.ABC):
    """
    Subclasses of this base class represent wrappers around external runnable software.

    A run of an :class:`Application` is started by calling the respective method
    producing a :class:`Future` that represents the run.
    """


class Future(abc.ABC, Generic[T]):
    """
    A handle to an application run that may still be in progress.

    Mirrors :class:`concurrent.futures.Future`.

    A :class:`Future` is usually not created directly, but is returned by the
    respective method of an :class:`Application`.

    A subclass must implement :meth:`_evaluate` to parse the finished run
    into its result object and :meth:`_cleanup` to release the resources
    of the run.
    """

    def __init__(self) -> None:
        self._result: T | None = None
        self._exception: Exception | None = None
        self._finished: bool = False
        self._cancelled: bool = False
        self._cleaned_up: bool = False

    @abc.abstractmethod
    def _wait(self, timeout: float | None) -> None:
        """
        Block until the underlying run finishes.

        Parameters
        ----------
        timeout : float, optional
            The maximum time to wait for the run to finish, in seconds.

        Raises
        ------
        TimeoutError
            If `timeout` (in seconds) is exceeded.

        PROTECTED: Override when inheriting.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _poll(self) -> bool:
        """
        Return whether the underlying run has finished (non-blocking).

        PROTECTED: Override when inheriting.

        Returns
        -------
        bool
            True if the run has finished, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _terminate(self) -> None:
        """
        Abort the underlying run.

        PROTECTED: Override when inheriting.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _evaluate(self) -> T:
        """
        Parse the finished run into its result object.

        PROTECTED: Override when inheriting.

        Returns
        -------
        result : object
            The result of the run.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _cleanup(self) -> None:
        """
        Release the resources of the run, e.g. temporary files.

        Called exactly once: after evaluation, on cancellation or on
        destruction, whichever comes first.

        PROTECTED: Override when inheriting.
        """
        raise NotImplementedError

    def _check(self) -> None:
        """
        Validate the finished run before evaluation, e.g. by inspecting
        an exit code, raising an appropriate exception on failure.

        PROTECTED: Optionally override when inheriting.
        """

    def cancel(self) -> bool:
        """
        Abort the run.

        Returns
        -------
        cancelled : bool
            True, if the run was cancelled, or False if it had already
            finished and could not be cancelled anymore.
        """
        if self._cancelled:
            return True
        if self._finished or self._poll():
            return False
        self._terminate()
        self._cancelled = True
        self._do_cleanup()
        return True

    def cancelled(self) -> bool:
        """
        Return True, if the run was cancelled.

        Returns
        -------
        bool
            True if the run was cancelled, False otherwise.
        """
        return self._cancelled

    def running(self) -> bool:
        """
        Return True, if the run is currently in progress.

        Returns
        -------
        bool
            True if the run is currently in progress, False otherwise.
        """
        return not self._cancelled and not self._finished and not self._poll()

    def done(self) -> bool:
        """
        Return True, if the run was cancelled or has finished.

        Returns
        -------
        bool
            True if the run was cancelled or has finished, False otherwise.

        Notes
        -----
        A finished run may still raise an exception in :meth:`result`.
        """
        return self._cancelled or self._finished or self._poll()

    def result(self, timeout: float | None = None) -> T:
        """
        Wait for the run to finish and return its result.

        Parameters
        ----------
        timeout : float, optional
            The maximum number of seconds to wait.
            By default, the method waits indefinitely.

        Returns
        -------
        result : object
            The result object of the run.

        Raises
        ------
        TimeoutError
            If the run does not finish within `timeout` seconds.
        CancelledError
            If the run was cancelled.
        """
        self._resolve(timeout)
        if self._exception is not None:
            raise self._exception
        return self._result  # type: ignore[return-value]

    def exception(self, timeout: float | None = None) -> Exception | None:
        """
        Wait for the run to finish and return the exception it raised.

        Parameters
        ----------
        timeout : float, optional
            The maximum number of seconds to wait.
            By default, the method waits indefinitely.

        Returns
        -------
        exception : Exception or None
            The exception raised by the run, or None if it finished
            without error.

        Raises
        ------
        TimeoutError
            If the run does not finish within `timeout` seconds.
        CancelledError
            If the run was cancelled.
        """
        self._resolve(timeout)
        return self._exception

    def _resolve(self, timeout: float | None) -> None:
        """
        Ensure the run is finished and evaluated, caching its result or
        exception.
        """
        if self._cancelled:
            raise CancelledError("The application run was cancelled")
        if self._finished:
            return
        # A timeout leaves the run in progress, so it stays retrievable
        # -> raise before entering the finished state
        self._wait(timeout)
        try:
            self._check()
            self._result = self._evaluate()
        except Exception as exception:
            self._exception = exception
        finally:
            self._finished = True
            self._do_cleanup()

    def _do_cleanup(self) -> None:
        if not self._cleaned_up:
            self._cleaned_up = True
            self._cleanup()

    def __del__(self) -> None:
        try:
            self._do_cleanup()
        except Exception:
            # Never raise during garbage collection
            pass


class CompositeFuture(Future[tuple[Unpack[Ts]]]):
    """
    A future combining the results of multiple futures.

    The composite is done when all input futures are done.
    Cancelling it propagates the cancellation request to every input future that is
    still running.
    Its result is a tuple containing the input results in the same order as the input
    futures.

    Parameters
    ----------
    *futures : Future
        The futures to combine. Their order determines the order of the
        returned results.
    """

    def __init__(self, *futures: Future[Any]) -> None:
        super().__init__()
        self._futures = futures
        self._child_exceptions: tuple[Exception | None, ...] = ()

    def _wait(self, timeout: float | None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        exceptions = []
        for future in self._futures:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            try:
                # `exception()` blocks until the underlying future finishes
                exception = future.exception(remaining)
            except CancelledError as e:
                # Treat cancellation like any other child failure
                # This lets the composite finish resolving its remaining children and
                # report the error through its own result.
                exception = e
            exceptions.append(exception)
        self._child_exceptions = tuple(exceptions)

    def _poll(self) -> bool:
        return all(future.done() for future in self._futures)

    def _terminate(self) -> None:
        for future in self._futures:
            future.cancel()

    def _check(self) -> None:
        for exception in self._child_exceptions:
            if exception is not None:
                raise exception

    def _evaluate(self) -> tuple[Unpack[Ts]]:
        results = tuple(future.result() for future in self._futures)
        return cast(tuple[Unpack[Ts]], results)

    def _cleanup(self) -> None:
        pass


class PostprocessFuture(Future[U], Generic[T, U]):
    """
    A future applying a function to the result of another future.

    The postprocessing function is called lazily and at most once when the postprocessed
    result is requested.

    Parameters
    ----------
    future : Future
        The input future.
    postprocess_fn : callable
        A function receiving the input result and returning the postprocessed
        result.

    Notes
    -----
    `postprocess_fn` is run in the same process as the caller, i.e. its call is blocking.
    """

    def __init__(
        self,
        future: Future[T],
        postprocess_fn: Callable[[T], U],
    ) -> None:
        super().__init__()
        self._input_future = future
        self._postprocess_fn = postprocess_fn
        self._input_exception: Exception | None = None

    def _wait(self, timeout: float | None) -> None:
        try:
            # `exception()` blocks until the underlying future finishes
            self._input_exception = self._input_future.exception(timeout)
        except CancelledError as exception:
            self._input_exception = exception

    def _poll(self) -> bool:
        return self._input_future.done()

    def _terminate(self) -> None:
        self._input_future.cancel()

    def _check(self) -> None:
        if self._input_exception is not None:
            raise self._input_exception

    def _evaluate(self) -> U:
        return self._postprocess_fn(self._input_future.result())

    def _cleanup(self) -> None:
        pass


class VersionError(Exception):
    """
    Indicate that the version of the wrapped software is not supported.
    """
