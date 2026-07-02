# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2"
__author__ = "Patrick Kunzmann"
__all__ = [
    "LocalApp",
    "LocalProcessFuture",
    "CLIArgument",
    "CLIOption",
    "CLIFlag",
    "CLIParameter",
    "CommandSetup",
    "command",
]

import functools
import inspect
import numbers
import re
import shutil
import string
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from io import IOBase
from os import PathLike, remove
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired
from tempfile import TemporaryDirectory
from typing import Any, Generic, TypeAlias, TypeVar, overload
from packaging.version import Version
from biotite.application_v2.base import Application, Future

T = TypeVar("T")


@dataclass(frozen=True)
class CLIArgument:
    """
    A positional command line argument.

    Attributes
    ----------
    value : object
        The argument value.
    """

    value: Any


@dataclass(frozen=True)
class CLIOption:
    """
    A key-value command line option.

    Attributes
    ----------
    key : str
        The option name.
    value : object
        The option value.
    """

    key: str
    value: Any


@dataclass(frozen=True)
class CLIFlag:
    """
    A valueless command line flag.

    Attributes
    ----------
    key : str
        The flag name.
    """

    key: str


CLIParameter: TypeAlias = CLIFlag | CLIOption | CLIArgument


class LocalProcessFuture(Future[T]):
    """
    A :class:`Future` backed by a locally running :class:`Popen`
    process.

    Parameters
    ----------
    process : Popen
        The already launched process.
    command : list of str
        The command line the process was launched with.
        Used for error messages and exposed via :attr:`command`.
    evaluate : callable
        A callable that parses the finished run and returns the result object.
        It is passed the captured ``STDOUT``/``STDERR`` and is invoked lazily and at
        most once, on the first call to :meth:`result` or :meth:`exception`.
    cleanup : callable, optional
        A zero-argument callable that releases resources (e.g. temporary files) of the
        run.
        It is invoked exactly once: after evaluation, on cancellation or on destruction,
        whichever comes first.
    parameters : sequence of CLIParameter, optional
        The original command line parameters.
        They are retained for the lifetime of the future so resources represented by an
        argument or option remain alive.

    Attributes
    ----------
    command : str
        The command line the process was launched with.
    """

    def __init__(
        self,
        process: Popen[bytes],
        command: list[str],
        evaluate: Callable[[bytes, bytes], T],
        cleanup: Callable[[], None] | None = None,
        parameters: Sequence[CLIParameter] = (),
    ) -> None:
        super().__init__()
        self._process = process
        self._command = command
        self._evaluate_fn = evaluate
        self._cleanup_fn = cleanup
        self._parameters = tuple(parameters)
        # Empty until the run finishes and the streams are captured
        self._stdout = b""
        self._stderr = b""

    def _evaluate(self) -> T:
        return self._evaluate_fn(self._stdout, self._stderr)

    def _cleanup(self) -> None:
        if self._cleanup_fn is not None:
            self._cleanup_fn()

    def _wait(self, timeout: float | None) -> None:
        try:
            # Use `communicate()` to avoid deadlocks if the output is larger
            # than the buffer size of the pipe
            self._stdout, self._stderr = self._process.communicate(timeout=timeout)
        except TimeoutExpired:
            # Leave the process running so that it stays retrievable
            raise TimeoutError(
                f"'{self._command[0]}' did not finish within {timeout} seconds"
            )

    def _poll(self) -> bool:
        return self._process.poll() is not None

    def _terminate(self) -> None:
        self._process.kill()

    def _check(self) -> None:
        exit_code = self._process.returncode
        if exit_code != 0:
            error_message = self._stderr.decode("UTF-8", errors="replace")
            error_message = error_message.replace("\n", " ")
            raise subprocess.SubprocessError(
                f"'{self._command[0]}' returned with exit code {exit_code}: "
                f"{error_message}"
            )

    @property
    def command(self) -> str:
        return " ".join(self._command)

    def stdout(self, timeout: float | None = None) -> bytes:
        """
        Wait for the run to finish and return the captured standard
        output.

        Parameters
        ----------
        timeout : float, optional
            The maximum number of seconds to wait.
            By default, the method waits indefinitely.

        Returns
        -------
        stdout : bytes
            The captured standard output as raw bytes.

        Raises
        ------
        TimeoutError
            If the run does not finish within `timeout` seconds.
        CancelledError
            If the run was cancelled.
        """
        self._resolve(timeout)
        return self._stdout

    def stderr(self, timeout: float | None = None) -> bytes:
        """
        Wait for the run to finish and return the captured standard
        error.

        Parameters
        ----------
        timeout : float, optional
            The maximum number of seconds to wait.
            By default, the method waits indefinitely.

        Returns
        -------
        stderr : bytes
            The captured standard error as raw bytes.

        Raises
        ------
        TimeoutError
            If the run does not finish within `timeout` seconds.
        CancelledError
            If the run was cancelled.
        """
        self._resolve(timeout)
        return self._stderr


class LocalApp(Application):
    """
    The base class for all locally installed command line applications.

    A run is launched via a method decorated with :func:`command`.

    Parameters
    ----------
    path : str or Path
        Either the name of the executable (resolved via the ``PATH``
        environment variable) or a path to it.
        The given value is resolved to an absolute path and exposed
        read-only via the :attr:`path` attribute.

    Attributes
    ----------
    version : Version
        The version of the wrapped software.
    path : Path
        The absolute path to the executable.
    """

    def __init__(self, path: PathLike[str] | str) -> None:
        path = str(path)
        # `shutil.which()` resolves a bare command name via the `PATH`
        # environment variable and, in either case, ensures that the
        # target exists and is executable
        resolved = shutil.which(path)
        if resolved is None:
            if Path(path).is_file():
                raise PermissionError(f"'{path}' exists, but is not executable")
            raise FileNotFoundError(f"The executable '{path}' does not exist")
        # Make the path absolute, but do NOT resolve symlinks: some
        # executables are multi-call binaries (e.g. sra-tools) that select
        # their tool based on the invocation (symlink) name
        self._path = Path(resolved).absolute()

    def _format_key(self, key: str) -> str:
        """
        Format an option name for the command line.

        By default, a leading double dash is added (a single dash for a
        one-character name) and interior underscores are replaced with
        dashes.

        PROTECTED: Override when the software uses a different convention.

        Parameters
        ----------
        key : str
            The option name.

        Returns
        -------
        formatted : str
            The formatted option name.

        Raises
        ------
        ValueError
            If `key` contains whitespace or has leading or trailing
            punctuation.
        """
        if any(char in string.whitespace for char in key):
            raise ValueError(f"Key '{key}' contains whitespace")
        if key.strip(string.punctuation) != key:
            raise ValueError(f"Key '{key}' contains leading or trailing punctuation")
        prefix = "-" if len(key) == 1 else "--"
        return prefix + key.replace("_", "-")

    def _format_value(self, value: Any) -> str:
        """
        Format an argument or option value for the command line.

        By default, :class:`str`, :class:`int`, :class:`float`,
        :class:`bool` (as ``0`` or ``1``), and (temporary) paths are supported.

        PROTECTED: Override to support additional value types.

        Parameters
        ----------
        value : object
            The value to format.

        Returns
        -------
        formatted : str
            The formatted value.

        Raises
        ------
        TypeError
            If the type of `value` is not supported.
        """
        match value:
            case bool():
                return "1" if value else "0"
            case numbers.Number():
                return str(value)
            case str():
                return value
            case Enum():
                return self._format_value(value.value)
            case Path():
                return str(value)
            case TemporaryDirectory():
                return value.name
            case IOBase():
                name = getattr(value, "name", None)
                if name is not None:
                    return str(name)
                raise TypeError(
                    f"Unsupported value type '{type(value).__name__}' without a name"
                )
            case _:
                # `NamedTemporaryFile()` returns a wrapper instead of an `IOBase`
                # instance, but exposes the wrapped file and its name.
                if isinstance(getattr(value, "file", None), IOBase):
                    return str(value.name)
                raise TypeError(f"Unsupported value type '{type(value).__name__}'")

    def _format_command(
        self,
        parameters: Sequence[CLIParameter],
        subcommand: str | None = None,
    ) -> list[str]:
        """
        Assemble the full command line for the run.

        By default, the executable is followed by the optional subcommand
        and the flags, the options and finally the arguments.
        The keys and values are formatted via :meth:`_format_key` and
        :meth:`_format_value`, respectively.

        PROTECTED: Override when the software expects a different command
        layout.

        Parameters
        ----------
        parameters : sequence of CLIParameter
            The command line flags, options and arguments.
        subcommand : str, optional
            A subcommand appended to the executable, before the
            parameters.

        Returns
        -------
        command : list of str
            The command line, starting with the executable.
        """
        command = [str(self._path)]
        if subcommand is not None:
            command.append(subcommand)
        for parameter in parameters:
            if isinstance(parameter, CLIFlag):
                command.append(self._format_key(parameter.key))
        for parameter in parameters:
            if isinstance(parameter, CLIOption):
                command.append(self._format_key(parameter.key))
                command.append(self._format_value(parameter.value))
        for parameter in parameters:
            if isinstance(parameter, CLIArgument):
                command.append(self._format_value(parameter.value))
        return command

    @property
    def version(self) -> Version:
        """
        The version of the wrapped software.

        The default implementation runs the executable with a ``version``
        flag, formatted via :meth:`_format_key`, and parses the version
        number from its standard output or standard error.
        Override this property if the software requires a different flag
        or its output does not match this format.

        Returns
        -------
        version : Version
            The version of the wrapped software.
        """
        command = self._format_command([CLIFlag("version")])
        process = subprocess.run(command, capture_output=True, text=True)
        output = process.stdout + process.stderr
        match = re.search(r"\d+\.\d+(?:\.\d+)*", output)
        if match is None:
            raise subprocess.SubprocessError(
                f"Could not determine the version of '{self._path.name}' "
                f"from the output '{output}'"
            )
        return Version(match.group())

    @property
    def path(self) -> Path:
        """
        The absolute path to the executable.

        Returns
        -------
        path : Path
            The absolute path to the executable.
        """
        return self._path

    def _execute(
        self,
        parameters: Sequence[CLIParameter],
        evaluate: Callable[[bytes, bytes], T],
        cleanup: Callable[[], None] | None = None,
        input: bytes | None = None,
        exec_dir: PathLike[str] | str | None = None,
        subcommand: str | None = None,
    ) -> LocalProcessFuture[T]:
        """
        Launch the executable and return a :class:`Future` for the run.

        PROTECTED: Called by the :func:`command` decorator to run the
        application.

        Parameters
        ----------
        parameters : sequence of CLIParameter
            The command line flags, options and arguments.
        evaluate : callable
            A callable parsing the finished run into its result object.
            It is passed the captured standard output and standard error
            as raw bytes.
        cleanup : callable, optional
            A zero-argument callable releasing resources of the run.
        input : bytes, optional
            Bytes written to the standard input of the run.
        exec_dir : str or Path, optional
            The working directory of the run.
            By default, the current working directory is used.
        subcommand : str, optional
            A subcommand appended to the executable.

        Returns
        -------
        future : LocalProcessFuture
            A handle to the launched run.

        Notes
        -----
        The standard output and standard error are always captured as
        raw bytes.
        Decoding them into text is the responsibility of `evaluate`.
        """
        command = self._format_command(parameters, subcommand)
        process = Popen(
            command,
            stdin=PIPE if input is not None else None,
            stdout=PIPE,
            stderr=PIPE,
            cwd=exec_dir,
        )
        if input is not None:
            process.stdin.write(input)  # type: ignore[union-attr]
            process.stdin.close()  # type: ignore[union-attr]
            # Detach so the later `communicate()` call does not access the closed stream
            process.stdin = None
        return LocalProcessFuture(
            process,
            command,
            evaluate,
            cleanup,
            parameters=parameters,
        )


@dataclass(frozen=True, kw_only=True)
class CommandSetup(Generic[T]):
    """
    The specification of a single command line run.

    A :class:`LocalApp` method decorated with :func:`command` returns a
    :class:`CommandSetup`, which the decorator turns into a launched
    :class:`LocalProcessFuture`.

    Attributes
    ----------
    evaluate : callable
        A callable parsing the finished run into its result object.
        It is passed the captured standard output and standard error as
        raw bytes.
    parameters : sequence of CLIParameter, optional
        The command line flags, options and arguments.
    cleanup : callable, optional
        A zero-argument callable releasing resources of the run.
    input : bytes, optional
        Bytes written to the standard input of the run.
    exec_dir : str or Path, optional
        The working directory of the run.
    """

    evaluate: Callable[[bytes, bytes], T]
    parameters: Sequence[CLIParameter] = ()
    cleanup: Callable[[], None] | None = None
    input: bytes | None = None
    exec_dir: PathLike[str] | str | None = None


@overload
def command(
    method: Callable[..., CommandSetup[T]],
) -> Callable[..., LocalProcessFuture[T]]: ...


@overload
def command(
    *,
    subcommand: str | None = None,
    allowed_options: list[str] | None = None,
) -> Callable[
    [Callable[..., CommandSetup[T]]], Callable[..., LocalProcessFuture[T]]
]: ...


def command(
    method: Callable[..., CommandSetup[T]] | None = None,
    *,
    subcommand: str | None = None,
    allowed_options: list[str] | None = None,
) -> Callable[..., Any]:
    """
    Turn a :class:`LocalApp` method into a runnable command.

    The decorated method must return a :class:`CommandSetup`.
    The resulting method accepts additional keyword arguments beyond its
    own parameters, which are added as command line options, and returns
    a :class:`LocalProcessFuture` for the launched run.
    An additional keyword argument is turned into a :class:`CLIFlag` if
    its value is ``None``, into multiple :class:`CLIOption` if its value
    is a list, and into a single :class:`CLIOption` otherwise.
    Only keyword arguments whose key is listed in `allowed_options` are
    accepted.

    Parameters
    ----------
    method : callable, optional
        The decorated method.
        Supplied automatically when the decorator is applied without
        parentheses.
    subcommand : str, optional
        A subcommand appended to the executable, before the options and
        arguments.
    allowed_options : list of str, optional
        The keys of the command line options that may be passed as extra keyword
        arguments.
        By default, no extra options are allowed.

    Returns
    -------
    decorated : callable
        The decorated method or, if `method` is not given, a decorator.

    Raises
    ------
    ValueError
        If an extra keyword argument not listed in `allowed_options` is used.

    Notes
    -----
    An option should only be added to `allowed_options`, if all of the
    following conditions are met:

    1. It is not already set by the command method.
    2. It does not counteract or is not mutually exclusive with an option
       set by the command method.
    3. It does not break the logic of the command method, i.e. it must not
       change the expected input format or the formatting of the output.
    """
    allowed = frozenset(allowed_options) if allowed_options is not None else frozenset()

    def decorator(
        method: Callable[..., CommandSetup[T]],
    ) -> Callable[..., LocalProcessFuture[T]]:
        parameter_names = set(inspect.signature(method).parameters)

        @functools.wraps(method)
        def wrapper(self: LocalApp, *args: Any, **kwargs: Any) -> LocalProcessFuture[T]:
            method_kwargs = {
                key: value for key, value in kwargs.items() if key in parameter_names
            }
            option_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in parameter_names
            }
            setup = method(self, *args, **method_kwargs)
            existing_keys = {
                parameter.key
                for parameter in setup.parameters
                if isinstance(parameter, (CLIFlag, CLIOption))
            }
            extra_parameters: list[CLIParameter] = []
            for key, value in option_kwargs.items():
                if key not in allowed:
                    raise ValueError(f"'{key}' is not an allowed command line option")
                if key in existing_keys:
                    raise ValueError(
                        f"The command line option '{key}' is already set by the "
                        f"application and cannot be given as keyword argument"
                    )
                if value is None:
                    extra_parameters.append(CLIFlag(key))
                elif isinstance(value, (list, tuple)):
                    extra_parameters.extend(
                        CLIOption(key, element) for element in value
                    )
                else:
                    extra_parameters.append(CLIOption(key, value))
            parameters = [*setup.parameters, *extra_parameters]
            return self._execute(
                parameters,
                setup.evaluate,
                setup.cleanup,
                input=setup.input,
                exec_dir=setup.exec_dir,
                subcommand=subcommand,
            )

        return wrapper

    if method is None:
        return decorator
    return decorator(method)


def cleanup_tempfile(temp_file: Any) -> None:
    """
    Close a :class:`NamedTemporaryFile` and delete it manually,
    if `delete` is set to ``False``.
    This function is a small helper function intended for usage in
    :class:`LocalApp` subclasses.

    The manual deletion is necessary, as Windows does not allow opening
    a :class:`NamedTemporaryFile` a second time
    (e.g. by the file name), if `delete` is set to ``True``.

    Parameters
    ----------
    temp_file : NamedTemporaryFile
        The temporary file to be closed and deleted.
    """
    temp_file.close()
    try:
        remove(temp_file.name)
    except FileNotFoundError:
        # File was already deleted, e.g. due to `TemporaryFile(delete=True)`
        pass
