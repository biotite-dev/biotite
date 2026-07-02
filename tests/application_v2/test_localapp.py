# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import gc
import weakref
from pathlib import Path
import pytest
from packaging.version import Version
from biotite.application_v2 import VersionError
from biotite.application_v2.clustalo import ClustalOmegaApp
from biotite.application_v2.localapp import (
    CLIArgument,
    CLIFlag,
    CLIOption,
    CommandSetup,
    LocalApp,
    command,
)
from biotite.application_v2.mafft import MafftApp
from biotite.application_v2.muscle import Muscle3App, Muscle5App
from tests.util import is_not_installed

# Each implemented `LocalApp` together with the name of its executable
APPS = [
    (Muscle3App, "muscle"),
    (Muscle5App, "muscle"),
    (MafftApp, "mafft"),
    (ClustalOmegaApp, "clustalo"),
]

# A trivial executable that ignores its arguments and exits successfully,
# used to construct a dummy `LocalApp` for testing base-class behavior
DUMMY_BIN = "true"


class _DummyApp(LocalApp):
    """
    A minimal :class:`LocalApp` subclass for testing base-class behavior.
    """

    @command(allowed_options=["some_option", "some_multi_option", "some_flag"])
    def run(self):
        return CommandSetup(evaluate=lambda stdout, stderr: None)


class _PresetApp(LocalApp):
    """
    A :class:`LocalApp` whose command already sets an option and a flag
    that are also listed as allowed options, used to test collision
    detection with extra keyword arguments.
    """

    @command(allowed_options=["preset_option", "preset_flag"])
    def run(self):
        return CommandSetup(
            parameters=[CLIOption("preset_option", 1), CLIFlag("preset_flag")],
            evaluate=lambda stdout, stderr: None,
        )


class _Resource:
    """
    An object whose lifetime can be observed through a weak reference.
    """


class _ResourceApp(LocalApp):
    """
    A :class:`LocalApp` accepting a resource as a command line argument.
    """

    def _format_value(self, value):
        if isinstance(value, _Resource):
            return "resource"
        return super()._format_value(value)

    @command
    def run(self, resource):
        return CommandSetup(
            parameters=[CLIArgument(resource)],
            evaluate=lambda stdout, stderr: None,
        )


def _dummy_or_skip():
    """
    Return a `_DummyApp` or skip the test if its executable is missing.
    """
    if is_not_installed(DUMMY_BIN):
        pytest.skip(f"'{DUMMY_BIN}' is not installed")
    return _DummyApp(DUMMY_BIN)


@pytest.mark.parametrize(
    ["app_class", "bin_path"],
    APPS,
    ids=[app_class.__name__ for app_class, _ in APPS],
)
def test_version(app_class, bin_path):
    """
    The ``version`` property returns the version of the software as a
    :class:`Version`.
    """
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    try:
        app = app_class(bin_path)
    except VersionError:
        pytest.skip("Invalid software version")

    version = app.version
    assert isinstance(version, Version)
    # A meaningful version was parsed from the software output
    assert version.release


@pytest.mark.parametrize(
    ["key", "expected"],
    [
        ("gap_open", "--gap-open"),
        ("x", "-x"),
        ("gap open", ValueError),
        ("--gap", ValueError),
    ],
)
def test_key_formatting(key, expected):
    """
    The default ``_format_key`` uses double dashes, converts interior
    underscores to dashes and rejects invalid keys.
    """
    app = _dummy_or_skip()

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            app._format_key(key)
    else:
        assert app._format_key(key) == expected


@pytest.mark.parametrize(
    ["value", "expected"],
    [
        ("foo", "foo"),
        (5, "5"),
        (1.5, "1.5"),
        (True, "1"),
        (False, "0"),
        (Path("/tmp/x"), str(Path("/tmp/x"))),
        (["not", "supported"], TypeError),
    ],
)
def test_value_formatting(value, expected):
    """
    The default ``_format_value`` supports str, number, bool and Path,
    and rejects other types.
    """
    app = _dummy_or_skip()

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            app._format_value(value)
    else:
        assert app._format_value(value) == expected


def test_invalid_path():
    """
    An unknown executable is reported at construction time.
    """
    with pytest.raises(FileNotFoundError):
        _DummyApp("this_executable_does_not_exist")


def test_non_executable_path(tmp_path):
    """
    An existing but non-executable file is reported at construction time.
    """
    non_executable = tmp_path / "not_a_binary"
    non_executable.write_text("")
    with pytest.raises(PermissionError):
        _DummyApp(non_executable)


def test_extra_option():
    """
    Options passed as additional keyword arguments to a command method
    end up in the executed command.
    """
    app = _dummy_or_skip()

    future = app.run(some_option=2, some_multi_option=["foo", "bar"], some_flag=None)
    assert future.command == (
        f"{app.path} "
        "--some-flag "
        "--some-option 2 "
        "--some-multi-option foo --some-multi-option bar"
    )
    # The run still completes successfully
    future.result()


def test_disallowed_option():
    """
    A keyword argument whose key is not listed in ``allowed_options``
    raises a :class:`ValueError`.
    """
    app = _dummy_or_skip()

    with pytest.raises(ValueError, match="not an allowed"):
        app.run(not_allowed=1)


@pytest.mark.parametrize("key", ["preset_option", "preset_flag"])
def test_extra_option_collision(key):
    """
    A keyword argument whose key is already set as a :class:`CLIOption`
    or :class:`CLIFlag` by the application raises a :class:`ValueError`.
    """
    if is_not_installed(DUMMY_BIN):
        pytest.skip(f"'{DUMMY_BIN}' is not installed")
    app = _PresetApp(DUMMY_BIN)

    with pytest.raises(ValueError, match="already set"):
        app.run(**{key: 42})


def test_parameters_are_retained_for_future_lifetime():
    """
    Command line parameter objects remain alive as long as the future exists.
    """
    if is_not_installed(DUMMY_BIN):
        pytest.skip(f"'{DUMMY_BIN}' is not installed")
    app = _ResourceApp(DUMMY_BIN)
    resource = _Resource()
    resource_reference = weakref.ref(resource)

    future = app.run(resource)
    del resource
    gc.collect()
    assert resource_reference() is not None

    future.result()
    gc.collect()
    assert resource_reference() is not None

    del future
    gc.collect()
    assert resource_reference() is None
