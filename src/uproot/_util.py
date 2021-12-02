# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines utilities for internal use. This is not a public interface
and may be changed without notice.
"""

from __future__ import absolute_import

import datetime
import glob
import numbers
import os
import platform
import re
import sys
import warnings

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
try:
    from urllib.parse import unquote, urlparse
except ImportError:
    from urlparse import urlparse, unquote

import numpy

py2 = sys.version_info[0] <= 2
py26 = py2 and sys.version_info[1] <= 6
py27 = py2 and not py26
py35 = not py2 and sys.version_info[1] <= 5
py36 = not py2 and sys.version_info[1] <= 6
win = platform.system().lower().startswith("win")


if py2:
    # to silence flake8 F821 errors
    unicode = eval("unicode")
    range = eval("xrange")
else:
    unicode = None
    range = eval("range")


def tobytes(array):
    """
    Calls ``array.tobytes()`` or its older equivalent, ``array.tostring()``,
    depending on what's available in this NumPy version.
    """
    if hasattr(array, "tobytes"):
        return array.tobytes()
    else:
        return array.tostring()


def isint(x):
    """
    Returns True if and only if ``x`` is an integer (including NumPy, not
    including bool).
    """
    return isinstance(x, (int, numbers.Integral, numpy.integer)) and not isinstance(
        x, (bool, numpy.bool_)
    )


def isnum(x):
    """
    Returns True if and only if ``x`` is a number (including NumPy, not
    including bool).
    """
    return isinstance(x, (int, float, numbers.Real, numpy.number)) and not isinstance(
        x, (bool, numpy.bool_)
    )


def isstr(x):
    """
    Returns True if and only if ``x`` is a string (including Python 2 unicode).
    """
    if py2:
        return isinstance(x, (bytes, unicode))
    else:
        return isinstance(x, str)


def ensure_str(x):
    """
    Ensures that ``x`` is a string (decoding with 'surrogateescape' if necessary).
    """
    if not py2 and isinstance(x, bytes):
        return x.decode(errors="surrogateescape")
    elif py2 and isinstance(x, unicode):
        return x.encode()
    elif isinstance(x, str):
        return x
    else:
        raise TypeError("expected a string, not {0}".format(type(x)))


def ensure_numpy(array, types=(numpy.bool_, numpy.integer, numpy.floating)):
    """
    Returns an ``np.ndarray`` if ``array`` can be converted to an array of the
    desired type and raises TypeError if it cannot.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", numpy.VisibleDeprecationWarning)
        try:
            out = numpy.asarray(array)
        except (ValueError, numpy.VisibleDeprecationWarning):
            raise TypeError("cannot be converted to a NumPy array")
        if not issubclass(out.dtype.type, types):
            raise TypeError(
                "cannot be converted to a NumPy array of type {0}".format(types)
            )
        return out


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
    """
    A filter that accepts anything (always returns True).
    """
    return True


def regularize_filter(filter):
    """
    Convert None, str, iterable of str, wildcards, and regular expressions into
    the standard form for a filter: a callable returning True or False.
    """
    if filter is None:
        return no_filter
    elif callable(filter):
        return filter
    elif isstr(filter):
        m = _regularize_filter_regex.match(filter)
        if m is not None:
            regex, flags = m.groups()
            matcher = re.compile(regex, _regularize_filter_regex_flags(flags))
            return lambda x: matcher.match(x) is not None
        elif "*" in filter or "?" in filter or "[" in filter:
            return lambda x: glob.fnmatch.fnmatchcase(x, filter)
        else:
            return lambda x: x == filter
    elif isinstance(filter, Iterable) and not isinstance(filter, bytes):
        filters = [regularize_filter(f) for f in filter]
        return lambda x: any(f(x) for f in filters)
    else:
        raise TypeError(
            "filter must be None, callable, a regex string between slashes, or a "
            "glob pattern, not {0}".format(repr(filter))
        )


def no_rename(x):
    """
    A renaming function that keeps all names the same (identity function).
    """
    return x


_regularize_filter_regex_rename = re.compile("^s?/(.*)/(.*)/([iLmsux]*)$")


def regularize_rename(rename):
    """
    Convert None, dict, and regular expression mappings into the standard form
    for renaming: a callable that maps strings to strings.
    """
    if rename is None:
        return no_rename

    elif callable(rename):
        return rename

    elif isstr(rename):
        m = _regularize_filter_regex_rename.match(rename)
        if m is not None:
            regex, trans, flags = m.groups()
            matcher = re.compile(regex, _regularize_filter_regex_flags(flags))
            return lambda x: matcher.sub(trans, x)
        else:
            raise TypeError("rename regular expressions must be in '/from/to/' form")

    elif isinstance(rename, dict):
        return lambda x: rename.get(x, x)

    elif isinstance(rename, Iterable) and not isinstance(rename, bytes):
        rules = []
        for x in rename:
            if isstr(x):
                m = _regularize_filter_regex_rename.match(x)
                if m is not None:
                    regex, trans, flags = m.groups()
                    rules.append(
                        (
                            re.compile(regex, _regularize_filter_regex_flags(flags)),
                            trans,
                        )
                    )
                else:
                    raise TypeError(
                        "rename regular expressions must be in '/from/to/' form"
                    )
            else:
                break
        else:

            def applyrules(x):
                for matcher, trans in rules:
                    if matcher.search(x) is not None:
                        return matcher.sub(trans, x)
                else:
                    return x

            return applyrules

    raise TypeError(
        "rename must be None, callable, a '/from/to/' regex, an iterable of "
        "regex rules, or a dict, not {0}".format(repr(rename))
    )


def regularize_path(path):
    """
    Converts pathlib Paths into plain string paths (for all versions of Python).
    """
    if isinstance(path, getattr(os, "PathLike", ())):
        path = os.fspath(path)

    elif hasattr(path, "__fspath__"):
        path = path.__fspath__()

    elif path.__class__.__module__ == "pathlib":
        import pathlib

        if isinstance(path, pathlib.Path):
            path = str(path)

    return path


_windows_drive_letter_ending = re.compile(r".*\b[A-Za-z]$")
_windows_absolute_path_pattern = re.compile(r"^[A-Za-z]:[\\/]")
_windows_absolute_path_pattern_slash = re.compile(r"^[\\/][A-Za-z]:[\\/]")
_might_be_port = re.compile(r"^[0-9].*")


def file_object_path_split(path):
    """
    Split a path with a colon into a file path and an object-in-file path.
    """
    path = regularize_path(path)

    try:
        index = path.rindex(":")
    except ValueError:
        return path, None
    else:
        file_path, object_path = path[:index], path[index + 1 :]

        if (
            _might_be_port.match(object_path) is not None
            and urlparse(file_path).path == ""
        ):
            return path, None

        file_path = file_path.rstrip()
        object_path = object_path.lstrip()

        if file_path.upper() in ("FILE", "HTTP", "HTTPS", "ROOT"):
            return path, None
        elif win and _windows_drive_letter_ending.match(file_path) is not None:
            return path, None
        else:
            return file_path, object_path


_remote_schemes = ["ROOT", "HTTP", "HTTPS"]
_schemes = ["FILE"] + _remote_schemes


def file_path_to_source_class(file_path, options):
    """
    Use a file path to get the :doc:`uproot.source.chunk.Source` class that would read it.
    """
    import uproot.source.chunk

    file_path = regularize_path(file_path)

    if (
        not isstr(file_path)
        and hasattr(file_path, "read")
        and hasattr(file_path, "seek")
    ):
        out = options["object_handler"]
        if not (isinstance(out, type) and issubclass(out, uproot.source.chunk.Source)):
            raise TypeError(
                "'object_handler' is not a class object inheriting from Source: "
                + repr(out)
            )
        return out, file_path

    windows_absolute_path = None
    if win:
        if _windows_absolute_path_pattern.match(file_path) is not None:
            windows_absolute_path = file_path

    parsed_url = urlparse(file_path)
    if parsed_url.scheme.upper() == "FILE":
        parsed_url_path = unquote(parsed_url.path)
    else:
        parsed_url_path = parsed_url.path

    if win and windows_absolute_path is None:
        if _windows_absolute_path_pattern.match(parsed_url_path) is not None:
            windows_absolute_path = parsed_url_path
        elif _windows_absolute_path_pattern_slash.match(parsed_url_path) is not None:
            windows_absolute_path = parsed_url_path[1:]

    if (
        parsed_url.scheme.upper() == "FILE"
        or len(parsed_url.scheme) == 0
        or windows_absolute_path is not None
    ):
        if windows_absolute_path is None:
            if parsed_url.netloc.upper() == "LOCALHOST":
                file_path = parsed_url_path
            else:
                file_path = parsed_url.netloc + parsed_url_path
        else:
            file_path = windows_absolute_path

        out = options["file_handler"]
        if not (isinstance(out, type) and issubclass(out, uproot.source.chunk.Source)):
            raise TypeError(
                "'file_handler' is not a class object inheriting from Source: "
                + repr(out)
            )
        return out, os.path.expanduser(file_path)

    elif parsed_url.scheme.upper() == "ROOT":
        out = options["xrootd_handler"]
        if not (isinstance(out, type) and issubclass(out, uproot.source.chunk.Source)):
            raise TypeError(
                "'xrootd_handler' is not a class object inheriting from Source: "
                + repr(out)
            )
        return out, file_path

    elif parsed_url.scheme.upper() == "HTTP" or parsed_url.scheme.upper() == "HTTPS":
        out = options["http_handler"]
        if not (isinstance(out, type) and issubclass(out, uproot.source.chunk.Source)):
            raise TypeError(
                "'http_handler' is not a class object inheriting from Source: "
                + repr(out)
            )
        return out, file_path

    else:
        raise ValueError("URI scheme not recognized: {0}".format(file_path))


if isinstance(__builtins__, dict):
    if "FileNotFoundError" in __builtins__:
        _FileNotFoundError = __builtins__["FileNotFoundError"]
    else:
        _FileNotFoundError = __builtins__["IOError"]
else:
    if hasattr(__builtins__, "FileNotFoundError"):
        _FileNotFoundError = __builtins__.FileNotFoundError
    else:
        _FileNotFoundError = __builtins__.IOError


def _file_not_found(files, message=None):
    if message is None:
        message = ""
    else:
        message = " (" + message + ")"

    return _FileNotFoundError(
        """file not found{0}

    {1}

Files may be specified as:
   * str/bytes: relative or absolute filesystem path or URL, without any colons
         other than Windows drive letter or URL schema.
         Examples: "rel/file.root", "C:\\abs\\file.root", "http://where/what.root"
   * str/bytes: same with an object-within-ROOT path, separated by a colon.
         Example: "rel/file.root:tdirectory/ttree"
   * pathlib.Path: always interpreted as a filesystem path or URL only (no
         object-within-ROOT path), regardless of whether there are any colons.
         Examples: Path("rel:/file.root"), Path("/abs/path:stuff.root")

Functions that accept many files (uproot.iterate, etc.) also allow:
   * glob syntax in str/bytes and pathlib.Path.
         Examples: Path("rel/*.root"), "/abs/*.root:tdirectory/ttree"
   * dict: keys are filesystem paths, values are objects-within-ROOT paths.
         Example: {{"/data_v1/*.root": "ttree_v1", "/data_v2/*.root": "ttree_v2"}}
   * already-open TTree objects.
   * iterables of the above.
""".format(
            message, repr(files)
        )
    )


def memory_size(data, error_message=None):
    """
    Regularizes strings like '## kB' and plain integer number of bytes to
    an integer number of bytes.
    """
    if isstr(data):
        m = re.match(
            r"^\s*([+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?)\s*([kmgtpezy]?b)\s*$",
            data,
            re.I,
        )
        if m is not None:
            target, unit = float(m.group(1)), m.group(5).upper()
            if unit == "KB":
                target *= 1000
            elif unit == "MB":
                target *= 1000 ** 2
            elif unit == "GB":
                target *= 1000 ** 3
            elif unit == "TB":
                target *= 1000 ** 4
            elif unit == "PB":
                target *= 1000 ** 5
            elif unit == "EB":
                target *= 1000 ** 6
            elif unit == "ZB":
                target *= 1000 ** 7
            elif unit == "YB":
                target *= 1000 ** 8
            elif unit == "KIB":
                target *= 1024
            elif unit == "MIB":
                target *= 1024 ** 2
            elif unit == "GIB":
                target *= 1024 ** 3
            elif unit == "TIB":
                target *= 1024 ** 4
            elif unit == "PIB":
                target *= 1024 ** 5
            elif unit == "EIB":
                target *= 1024 ** 6
            elif unit == "ZIB":
                target *= 1024 ** 7
            elif unit == "YIB":
                target *= 1024 ** 8
            return int(target)

    if isint(data):
        return int(data)

    if error_message is None:
        raise TypeError(
            "number of bytes or memory size string with units "
            "(such as '100 MB') required, not {0}".format(repr(data))
        )
    else:
        raise TypeError(error_message)


def new_class(name, bases, members):
    """
    Create a new class object with ``type(name, bases, members)`` and put it in
    the ``uproot.dynamic`` library.
    """
    import uproot.dynamic

    out = type(ensure_str(name), bases, members)
    out.__module__ = "uproot.dynamic"
    setattr(uproot.dynamic, out.__name__, out)
    return out


_primitive_awkward_form = {}


def awkward_form(
    model, file, index_format="i64", header=False, tobject_header=True, breadcrumbs=()
):
    """
    Utility function to get an ``ak.forms.Form`` for a :doc:`uproot.model.Model`.
    """
    import uproot

    awkward = uproot.extras.awkward()

    if isinstance(model, numpy.dtype):
        model = model.newbyteorder("=")

        if model not in _primitive_awkward_form:
            if model == numpy.dtype(numpy.bool_) or model == numpy.dtype(bool):
                _primitive_awkward_form[model] = awkward.forms.Form.fromjson('"bool"')
            elif model == numpy.dtype(numpy.int8):
                _primitive_awkward_form[model] = awkward.forms.Form.fromjson('"int8"')
            elif model == numpy.dtype(numpy.uint8):
                _primitive_awkward_form[model] = awkward.forms.Form.fromjson('"uint8"')
            elif model == numpy.dtype(numpy.int16):
                _primitive_awkward_form[model] = awkward.forms.Form.fromjson('"int16"')
            elif model == numpy.dtype(numpy.uint16):
                _primitive_awkward_form[model] = awkward.forms.Form.fromjson('"uint16"')
            elif model == numpy.dtype(numpy.int32):
                _primitive_awkward_form[model] = awkward.forms.Form.fromjson('"int32"')
            elif model == numpy.dtype(numpy.uint32):
                _primitive_awkward_form[model] = awkward.forms.Form.fromjson('"uint32"')
            elif model == numpy.dtype(numpy.int64):
                _primitive_awkward_form[model] = awkward.forms.Form.fromjson('"int64"')
            elif model == numpy.dtype(numpy.uint64):
                _primitive_awkward_form[model] = awkward.forms.Form.fromjson('"uint64"')
            elif model == numpy.dtype(numpy.float32):
                _primitive_awkward_form[model] = awkward.forms.Form.fromjson(
                    '"float32"'
                )
            elif model == numpy.dtype(numpy.float64):
                _primitive_awkward_form[model] = awkward.forms.Form.fromjson(
                    '"float64"'
                )
            else:
                raise AssertionError("{0}: {1}".format(repr(model), type(model)))

        return _primitive_awkward_form[model]

    else:
        return model.awkward_form(
            file, index_format, header, tobject_header, breadcrumbs
        )


def awkward_form_remove_uproot(awkward, form):
    """
    Remove the "uproot" parameters from an ``ak.forms.Form``.
    """
    parameters = dict(form.parameters)
    parameters.pop("uproot", None)
    if isinstance(form, awkward.forms.BitMaskedForm):
        return awkward.forms.BitMaskedForm(
            form.mask,
            awkward_form_remove_uproot(awkward, form.content),
            form.valid_when,
            form.lsb_order,
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.ByteMaskedForm):
        return awkward.forms.ByteMaskedForm(
            form.mask,
            awkward_form_remove_uproot(awkward, form.content),
            form.valid_when,
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.EmptyForm):
        return awkward.forms.EmptyForm(
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.IndexedForm):
        return awkward.forms.IndexedForm(
            form.index,
            awkward_form_remove_uproot(awkward, form.content),
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.IndexedOptionForm):
        return awkward.forms.IndexedOptionForm(
            form.index,
            awkward_form_remove_uproot(awkward, form.content),
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.ListForm):
        return awkward.forms.ListForm(
            form.starts,
            form.stops,
            awkward_form_remove_uproot(awkward, form.content),
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.ListOffsetForm):
        return awkward.forms.ListOffsetForm(
            form.offsets,
            awkward_form_remove_uproot(awkward, form.content),
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.NumpyForm):
        return awkward.forms.NumpyForm(
            form.inner_shape,
            form.itemsize,
            form.format,
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.RecordForm):
        return awkward.forms.RecordForm(
            dict(
                (k, awkward_form_remove_uproot(awkward, v))
                for k, v in form.contents.items()
                if not k.startswith("@")
            ),
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.RegularForm):
        return awkward.forms.RegularForm(
            awkward_form_remove_uproot(awkward, form.content),
            form.size,
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.UnionForm):
        return awkward.forms.UnionForm(
            form.tags,
            form.index,
            [awkward_form_remove_uproot(awkward, x) for x in form.contents],
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.UnmaskedForm):
        return awkward.forms.UnmaskedForm(
            awkward_form_remove_uproot(awkward, form.content),
            form.has_identities,
            parameters,
        )
    elif isinstance(form, awkward.forms.VirtualForm):
        return awkward.forms.VirtualForm(
            awkward_form_remove_uproot(awkward, form.form),
            form.has_length,
            form.has_identities,
            parameters,
        )
    else:
        raise RuntimeError("unrecognized form: {0}".format(type(form)))


# FIXME: Until we get Awkward reading these bytes directly, rather than
# going through ak.from_iter, the integer dtypes will be int64 and the
# floating dtypes will be float64 because that's what ak.from_iter makes.
def recursively_fix_awkward_form_of_iter(awkward, interpretation, form):
    """
    Given an interpretation of a TBranch, fixup any form corresponding to
    a component that cannot be read directly by awkward and needs to go
    through ak.from_iter
    """
    import uproot.interpretation

    if isinstance(interpretation, uproot.interpretation.grouped.AsGrouped):
        fixed = {}
        for key, subinterpretation in interpretation.subbranches.items():
            fixed[key] = recursively_fix_awkward_form_of_iter(
                awkward, subinterpretation, form.content(key)
            )
        return awkward.forms.RecordForm(
            fixed,
            form.has_identities,
            form.parameters,
        )
    elif isinstance(interpretation, uproot.interpretation.objects.AsObjects):
        if uproot.interpretation.objects.awkward_can_optimize(interpretation, form):
            return form
        else:
            return uproot._util.awkward_form_of_iter(awkward, form)
    else:
        return form


def awkward_form_of_iter(awkward, form):
    """
    Fix an ``ak.forms.Form`` object for a given iterable.

    (It might have been read with a different numeric type.)
    """
    if isinstance(form, awkward.forms.BitMaskedForm):
        return awkward.forms.BitMaskedForm(
            form.mask,
            awkward_form_of_iter(awkward, form.content),
            form.valid_when,
            form.lsb_order,
            form.has_identities,
            form.parameters,
        )
    elif isinstance(form, awkward.forms.ByteMaskedForm):
        return awkward.forms.ByteMaskedForm(
            form.mask,
            awkward_form_of_iter(awkward, form.content),
            form.valid_when,
            form.has_identities,
            form.parameters,
        )
    elif isinstance(form, awkward.forms.EmptyForm):
        return awkward.forms.EmptyForm(
            form.has_identities,
            form.parameters,
        )
    elif isinstance(form, awkward.forms.IndexedForm):
        return awkward.forms.IndexedForm(
            form.index,
            awkward_form_of_iter(awkward, form.content),
            form.has_identities,
            form.parameters,
        )
    elif isinstance(form, awkward.forms.IndexedOptionForm):
        return awkward.forms.IndexedOptionForm(
            form.index,
            awkward_form_of_iter(awkward, form.content),
            form.has_identities,
            form.parameters,
        )
    elif isinstance(form, awkward.forms.ListForm):
        return awkward.forms.ListForm(
            form.starts,
            form.stops,
            awkward_form_of_iter(awkward, form.content),
            form.has_identities,
            form.parameters,
        )
    elif isinstance(form, awkward.forms.ListOffsetForm):
        return awkward.forms.ListOffsetForm(
            form.offsets,
            awkward_form_of_iter(awkward, form.content),
            form.has_identities,
            form.parameters,
        )
    elif isinstance(form, awkward.forms.NumpyForm):
        if form.parameter("__array__") in ("char", "byte"):
            f = form
        else:
            d = form.to_numpy()
            if issubclass(d.type, numpy.integer):
                d = numpy.dtype(numpy.int64)
            elif issubclass(d.type, numpy.floating):
                d = numpy.dtype(numpy.float64)
            f = awkward.forms.Form.from_numpy(d)
        out = awkward.forms.NumpyForm(
            form.inner_shape,
            f.itemsize,
            f.format,
            form.has_identities,
            form.parameters,
        )
        return out
    elif isinstance(form, awkward.forms.RecordForm):
        return awkward.forms.RecordForm(
            dict(
                (k, awkward_form_of_iter(awkward, v)) for k, v in form.contents.items()
            ),
            form.has_identities,
            form.parameters,
        )
    elif isinstance(form, awkward.forms.RegularForm):
        return awkward.forms.RegularForm(
            awkward_form_of_iter(awkward, form.content),
            form.size,
            form.has_identities,
            form.parameters,
        )
    elif isinstance(form, awkward.forms.UnionForm):
        return awkward.forms.UnionForm(
            form.tags,
            form.index,
            [awkward_form_of_iter(awkward, x) for x in form.contents],
            form.has_identities,
            form.parameters,
        )
    elif isinstance(form, awkward.forms.UnmaskedForm):
        return awkward.forms.UnmaskedForm(
            awkward_form_of_iter(awkward, form.content),
            form.has_identities,
            form.parameters,
        )
    elif isinstance(form, awkward.forms.VirtualForm):
        return awkward.forms.VirtualForm(
            awkward_form_of_iter(awkward, form.form),
            form.has_length,
            form.has_identities,
            form.parameters,
        )
    else:
        raise RuntimeError("unrecognized form: {0}".format(type(form)))


def damerau_levenshtein(a, b, ratio=False):
    """
    Calculates the Damerau-Levenshtein distance of two strings.

    Used by :doc:`uproot.exceptions.KeyInFileError` to return the most likely
    misspellings of a failed attempt to get a key.
    """
    # Modified Damerau-Levenshtein distance. Adds a middling penalty
    # for capitalization.
    # https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    M = [[0] * (len(b) + 1) for i in range(len(a) + 1)]

    for i in range(len(a) + 1):
        M[i][0] = i
    for j in range(len(b) + 1):
        M[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:  # Same char
                cost = 0
            elif a[i - 1].lower() == b[j - 1].lower():  # Same if lowered
                cost = 0.5
            else:  # Different char
                cost = 2
            M[i][j] = min(
                M[i - 1][j] + 1,  # Addition
                M[i][j - 1] + 1,  # Removal
                M[i - 1][j - 1] + cost,  # Substitution
            )

            # Transposition
            if (
                i > 1
                and j > 1
                and a[i - 1].lower() == b[j - 2].lower()
                and a[i - 2].lower() == b[j - 2].lower()
            ):
                if a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                    # Transpose only
                    M[i][j] = min(M[i][j], M[i - 2][j - 2] + 1)
                else:
                    # Traspose and capitalization
                    M[i][j] = min(M[i][j], M[i - 2][j - 2] + 1.5)

    if not ratio:
        return M[len(a)][len(b)]
    else:
        return (len(a) + len(b)) - M[len(a)][len(b)] / (len(a) + len(b))


def code_to_datetime(code):
    """
    Converts a ROOT datime code into a Python datetime.
    """
    return datetime.datetime(
        ((code & 0b11111100000000000000000000000000) >> 26) + 1995,
        ((code & 0b00000011110000000000000000000000) >> 22),
        ((code & 0b00000000001111100000000000000000) >> 17),
        ((code & 0b00000000000000011111000000000000) >> 12),
        ((code & 0b00000000000000000000111111000000) >> 6),
        ((code & 0b00000000000000000000000000111111)),
    )


def datetime_to_code(dt):
    """
    Converts a Python datetime into a ROOT datime code.
    """
    return (
        ((dt.year - 1995) << 26)
        | (dt.month << 22)
        | (dt.day << 17)
        | (dt.hour << 12)
        | (dt.minute << 6)
        | (dt.second)
    )


def objectarray1d(items):
    """
    Converts a sized iterable into a 1D object array

    This avoids ``numpy.array``'s default behavior of turning nested iterables
    into n-d arrays when the shape is rectangular
    """
    out = numpy.empty(len(items), dtype=numpy.dtype(object))
    for i, x in enumerate(items):
        out[i] = x
    return out
