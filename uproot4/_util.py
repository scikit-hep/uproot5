# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Utilities for internal use.
"""

from __future__ import absolute_import

import os
import sys
import numbers
import re

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

import numpy


py2 = sys.version_info[0] <= 2
py26 = py2 and sys.version_info[1] <= 6
py27 = py2 and not py26
py35 = not py2 and sys.version_info[1] <= 5


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


_windows_absolute_path_pattern = re.compile(r"^[A-Za-z]:\\")


def path_to_source_class(file_path, options):
    if isinstance(file_path, getattr(os, "PathLike", ())):
        file_path = os.fspath(file_path)
    elif hasattr(file_path, "__fspath__"):
        file_path = file_path.__fspath__()
    elif file_path.__class__.__module__ == "pathlib":
        import pathlib

        if isinstance(file_path, pathlib.Path):
            file_path = str(file_path)

    windows_absolute_path = (
        os.name == "nt" and _windows_absolute_path_pattern.match(file_path) is not None
    )
    parsed_url = urlparse(file_path)
    if (
        parsed_url.scheme == "file"
        or len(parsed_url.scheme) == 0
        or windows_absolute_path
    ):
        if not windows_absolute_path:
            file_path = parsed_url.netloc + parsed_url.path
        return options["file_handler"]

    elif parsed_url.scheme == "root":
        return options["xrootd_handler"]

    elif parsed_url.scheme == "http" or parsed_url.scheme == "https":
        return options["http_handler"]

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
