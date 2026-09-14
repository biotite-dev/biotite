# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
from biotite.application_v2 import CompositeFuture, Future, PostprocessFuture


class _ManualFuture(Future):
    """
    A controllable future for testing future composition.
    """

    def __init__(self, result=None, exception=None, done=False):
        super().__init__()
        self._value = result
        self._given_exception = exception
        self._done = done
        self.terminated = False
        self.cleaned_up = False

    def finish(self):
        self._done = True

    def _wait(self, timeout):
        if not self._done:
            raise TimeoutError

    def _poll(self):
        return self._done

    def _terminate(self):
        self.terminated = True

    def _check(self):
        if self._given_exception is not None:
            raise self._given_exception

    def _evaluate(self):
        return self._value

    def _cleanup(self):
        self.cleaned_up = True


def test_composite_future_result():
    """
    A composite future becomes done with its last input and returns all
    results in input order.
    """
    first = _ManualFuture(result="first", done=True)
    second = _ManualFuture(result=2)
    future = CompositeFuture(first, second)

    assert not future.done()
    second.finish()
    assert future.done()
    assert future.result() == ("first", 2)


def test_composite_future_cancellation():
    """
    Cancelling a composite future propagates to all running inputs.
    """
    children = [_ManualFuture(), _ManualFuture()]
    future = CompositeFuture(*children)

    assert future.cancel()
    assert future.cancelled()
    assert all(child.cancelled() for child in children)
    assert all(child.terminated for child in children)


def test_composite_future_exception():
    """
    An input exception is propagated after all inputs have been resolved.
    """
    exception = ValueError("invalid result")
    first = _ManualFuture(exception=exception, done=True)
    second = _ManualFuture(result=2, done=True)

    with pytest.raises(ValueError, match="invalid result"):
        CompositeFuture(first, second).result()
    assert second.cleaned_up


def test_postprocess_future_result():
    """
    Postprocessing is applied lazily and exactly once.
    """
    calls = []

    def postprocess(value):
        calls.append(value)
        return value * 2

    future = PostprocessFuture(
        _ManualFuture(result=3, done=True),
        postprocess,
    )

    assert future.result() == 6
    assert future.result() == 6
    assert calls == [3]


def test_postprocess_future_cancellation():
    """
    Cancelling a postprocessing future propagates to its input future.
    """
    input_future = _ManualFuture()
    future = PostprocessFuture(input_future, lambda value: value)

    assert future.cancel()
    assert input_future.cancelled()
    assert input_future.terminated


def test_postprocess_future_exception():
    """
    An input exception prevents the postprocessing function from running.
    """
    was_called = False

    def postprocess(value):
        nonlocal was_called
        was_called = True
        return value

    input_future = _ManualFuture(
        exception=ValueError("invalid result"),
        done=True,
    )
    with pytest.raises(ValueError, match="invalid result"):
        PostprocessFuture(input_future, postprocess).result()
    assert not was_called
