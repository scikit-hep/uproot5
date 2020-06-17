# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Utilities for internal use.
"""

from __future__ import absolute_import

import os
import sys
import numbers
import re
import glob

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

import numpy


py2 = sys.version_info[0] <= 2
py26 = py2 and sys.version_info[1] <= 6
py27 = py2 and not py26
py35 = not py2 and sys.version_info[1] <= 5
win = os.name == "nt"


# to silence flake8 F821 errors
if py2:
    unicode = eval("unicode")
else:
    unicode = None


def isint(x):
    """
    Returns True if and only if `x` is an integer (including NumPy, not
    including bool).
    """
    return isinstance(x, (int, numbers.Integral, numpy.integer)) and not isinstance(
        x, (numpy.bool, numpy.bool_)
    )


def isnum(x):
    """
    Returns True if and only if `x` is a number (including NumPy, not
    including bool).
    """
    return isinstance(x, (int, float, numbers.Real, numpy.number)) and not isinstance(
        x, (numpy.bool, numpy.bool_)
    )


def isstr(x):
    if py2:
        return isinstance(x, (bytes, unicode))
    else:
        return isinstance(x, str)


def ensure_str(x):
    if not py2 and isinstance(x, bytes):
        return x.decode(errors="surrogateescape")
    elif py2 and isinstance(x, unicode):
        return x.encode()
    elif isinstance(x, str):
        return x
    else:
        raise TypeError("expected a string, not {0}".format(type(x)))


_regularize_filter_regex = re.compile("^/(.*)/([iLmsux]*)$")


def _regularize_filter_regex_flags(flags):
    flagsbyte = 0
    for flag in flags:
        if flag == "i":
            flagsbyte += re.I
        elif flag == "L":
            flagsbyte += re.L
        elif flag == "m":
            flagsbyte += re.M
        elif flag == "s":
            flagsbyte += re.S
        elif flag == "u":
            flagsbyte += re.U
        elif flag == "x":
            flagsbyte += re.X
    return flagsbyte


def no_filter(x):
    return True


def exact_filter(filter):
    if filter is None:
        return False
    elif callable(filter):
        return False
    if isstr(filter):
        m = _regularize_filter_regex.match(filter)
        if m is not None:
            return False
        elif "*" in filter or "?" in filter or "[" in filter:
            return False
        else:
            return True
    else:
        raise TypeError(
            "filter must be callable, a regex string between slashes, or a "
            "glob pattern, not {0}".format(repr(filter))
        )


def regularize_filter(filter):
    if filter is None:
        return no_filter
    elif callable(filter):
        return filter
    elif isstr(filter):
        m = _regularize_filter_regex.match(filter)
        if m is not None:
            regex, flags = m.groups()
            return (
                lambda x: re.match(regex, x, _regularize_filter_regex_flags(flags))
                is not None
            )
        elif "*" in filter or "?" in filter or "[" in filter:
            return lambda x: glob.fnmatch.fnmatchcase(x, filter)
        else:
            return lambda x: x == filter
    elif isinstance(filter, Iterable) and not isinstance(filter, bytes):
        filters = [regularize_filter(f) for f in filter]
        return lambda x: any(f(x) for f in filters)
    else:
        raise TypeError(
            "filter must be callable, a regex string between slashes, or a "
            "glob pattern, not {0}".format(repr(filter))
        )


_windows_absolute_path_pattern = re.compile(r"^[A-Za-z]:\\")
_windows_absolute_path_pattern_slash = re.compile(r"^/[A-Za-z]:\\")


def path_to_source_class(file_path, options):
    if isinstance(file_path, getattr(os, "PathLike", ())):
        file_path = os.fspath(file_path)

    elif hasattr(file_path, "__fspath__"):
        file_path = file_path.__fspath__()

    elif file_path.__class__.__module__ == "pathlib":
        import pathlib

        if isinstance(file_path, pathlib.Path):
            file_path = str(file_path)

    windows_absolute_path = None

    if os.name == "nt":
        if _windows_absolute_path_pattern.match(file_path) is not None:
            windows_absolute_path = file_path

    parsed_url = urlparse(file_path)

    if os.name == "nt" and windows_absolute_path is None:
        if _windows_absolute_path_pattern.match(parsed_url.path) is not None:
            windows_absolute_path = parsed_url.path
        elif _windows_absolute_path_pattern_slash.match(parsed_url.path) is not None:
            windows_absolute_path = parsed_url.path[1:]

    if (
        parsed_url.scheme == "file"
        or len(parsed_url.scheme) == 0
        or windows_absolute_path
    ):
        if windows_absolute_path is None:
            if parsed_url.netloc == "localhost":
                file_path = parsed_url.path
            else:
                file_path = parsed_url.netloc + parsed_url.path
        else:
            file_path = windows_absolute_path

        return options["file_handler"], os.path.expanduser(file_path)

    elif parsed_url.scheme == "root":
        return options["xrootd_handler"], file_path

    elif parsed_url.scheme == "http" or parsed_url.scheme == "https":
        return options["http_handler"], file_path

    else:
        raise ValueError("URI scheme not recognized: {0}".format(file_path))


def memory_size(data):
    """
    Normalizes strings like '## kB' and plain integer number of bytes to
    an integer number of bytes.
    """
    if isinstance(data, str):
        m = re.match(
            r"^\s*([+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?)\s*([kmgtpezy]?b)\s*$",
            data,
            re.I,
        )
        if m is not None:
            target, unit = float(m.group(1)), m.group(5).upper()
            if unit == "KB":
                target *= 1024
            elif unit == "MB":
                target *= 1024 ** 2
            elif unit == "GB":
                target *= 1024 ** 3
            elif unit == "TB":
                target *= 1024 ** 4
            elif unit == "PB":
                target *= 1024 ** 5
            elif unit == "EB":
                target *= 1024 ** 6
            elif unit == "ZB":
                target *= 1024 ** 7
            elif unit == "YB":
                target *= 1024 ** 8
            return int(target)

    elif isint(data):
        return int(data)

    else:
        raise TypeError(
            "number of bytes or memory size string with units "
            "required, not {0}".format(repr(data))
        )


def new_class(name, bases, members):
    out = type(ensure_str(name), bases, members)
    out.__module__ = "<dynamic>"
    return out
