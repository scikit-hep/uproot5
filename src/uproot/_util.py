# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines utilities for internal use. This is not a public interface
and may be changed without notice.
"""

import datetime
import glob
import itertools
import numbers
import os
import platform
import re
import warnings
from collections.abc import Iterable
from urllib.parse import unquote, urlparse

import numpy
import packaging.version

win = platform.system().lower().startswith("win")


def tobytes(array):
    """
    Calls ``array.tobytes()`` or its older equivalent, ``array.tostring()``,
    depending on what's available in this NumPy version. (tobytes added in 1.9)
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
    return isinstance(x, str)


def ensure_str(x):
    """
    Ensures that ``x`` is a string (decoding with 'surrogateescape' if necessary).
    """
    if isinstance(x, bytes):
        return x.decode(errors="surrogateescape")
    elif isinstance(x, str):
        return x
    else:
        raise TypeError(f"expected a string, not {type(x)}")


def ensure_numpy(array, types=(numpy.bool_, numpy.integer, numpy.floating)):
    """
    Returns an ``np.ndarray`` if ``array`` can be converted to an array of the
    desired type and raises TypeError if it cannot.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", numpy.VisibleDeprecationWarning)
        try:
            out = numpy.asarray(array)
        except (ValueError, numpy.VisibleDeprecationWarning) as err:
            raise TypeError("cannot be converted to a NumPy array") from err
        if not issubclass(out.dtype.type, types):
            raise TypeError(f"cannot be converted to a NumPy array of type {types}")
        return out


def parse_version(version):
    """
    Converts a semver string into a Version object that can be compared with
    ``<``, ``>=``, etc.

    Currently implemented using ``packaging.Version``
    (exposing that library in the return type).
    """
    return packaging.version.parse(version)


def from_module(obj, module_name):
    """
    Returns True if ``obj`` is an instance of a class from a module
    given by name.

    This is like ``isinstance`` (in that it searches the whole ``mro``),
    except that the module providing the type to check against doesn't
    have to be imported and doesn't get imported (as a side effect) by
    this function.
    """
    try:
        mro = type(obj).mro()
    except TypeError:
        return False

    for t in mro:
        if t.__module__ == module_name or t.__module__.startswith(module_name + "."):
            return True
    return False


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


_regularize_filter_regex = re.compile("^/(.*)/([iLmsux]*)$")


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
            "glob pattern, not {}".format(repr(filter))
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
                        return matcher.sub(trans, x)  # noqa: B023
                else:
                    return x

            return applyrules

    raise TypeError(
        "rename must be None, callable, a '/from/to/' regex, an iterable of "
        "regex rules, or a dict, not {}".format(repr(rename))
    )


_fix_url_path = re.compile(r"^((file|https?|root):/)([^/])", re.I)


def regularize_path(path):
    """
    Converts pathlib Paths into plain string paths (for all versions of Python).
    """
    if isinstance(path, getattr(os, "PathLike", ())):
        path = _fix_url_path.sub(r"\1/\3", os.fspath(path))

    elif hasattr(path, "__fspath__"):
        path = _fix_url_path.sub(r"\1/\3", path.__fspath__())

    elif path.__class__.__module__ == "pathlib":
        import pathlib

        if isinstance(path, pathlib.Path):
            path = _fix_url_path.sub(r"\1/\3", str(path))

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

    elif parsed_url.scheme.upper() in {"HTTP", "HTTPS"}:
        out = options["http_handler"]
        if not (isinstance(out, type) and issubclass(out, uproot.source.chunk.Source)):
            raise TypeError(
                "'http_handler' is not a class object inheriting from Source: "
                + repr(out)
            )
        return out, file_path

    else:
        raise ValueError(f"URI scheme not recognized: {file_path}")


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
                target *= 1000**2
            elif unit == "GB":
                target *= 1000**3
            elif unit == "TB":
                target *= 1000**4
            elif unit == "PB":
                target *= 1000**5
            elif unit == "EB":
                target *= 1000**6
            elif unit == "ZB":
                target *= 1000**7
            elif unit == "YB":
                target *= 1000**8
            elif unit == "KIB":
                target *= 1024
            elif unit == "MIB":
                target *= 1024**2
            elif unit == "GIB":
                target *= 1024**3
            elif unit == "TIB":
                target *= 1024**4
            elif unit == "PIB":
                target *= 1024**5
            elif unit == "EIB":
                target *= 1024**6
            elif unit == "ZIB":
                target *= 1024**7
            elif unit == "YIB":
                target *= 1024**8
            return int(target)

    if isint(data):
        return int(data)

    if error_message is None:
        raise TypeError(
            "number of bytes or memory size string with units "
            f"(such as '100 MB') required, not {data!r}"
        )
    else:
        raise TypeError(error_message)


def trim_final(basket_arrays, entry_start, entry_stop, entry_offsets, library, branch):
    """
    Trims the output from a basket_array function and outputs the un-concatenated list of elements.
    """
    trimmed = []
    start = entry_offsets[0]
    for basket_num, stop in enumerate(entry_offsets[1:]):
        if start <= entry_start and entry_stop <= stop:
            local_start = entry_start - start
            local_stop = entry_stop - start
            trimmed.append(basket_arrays[basket_num][local_start:local_stop])

        elif start <= entry_start < stop:
            local_start = entry_start - start
            local_stop = stop - start
            trimmed.append(basket_arrays[basket_num][local_start:local_stop])

        elif start <= entry_stop <= stop:
            local_start = 0
            local_stop = entry_stop - start
            trimmed.append(basket_arrays[basket_num][local_start:local_stop])

        elif entry_start < stop and start <= entry_stop:
            trimmed.append(basket_arrays[basket_num])

        start = stop

    return trimmed


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


def awkward_form(model, file, context):
    """
    Utility function to get an ``ak.forms.Form`` for a :doc:`uproot.model.Model`.
    """
    import uproot

    awkward = uproot.extras.awkward()

    if isinstance(model, numpy.dtype):
        model = model.newbyteorder("=")

        if model not in _primitive_awkward_form:
            if model == numpy.dtype(numpy.bool_) or model == numpy.dtype(bool):
                _primitive_awkward_form[model] = awkward.forms.from_json('"bool"')
            elif model == numpy.dtype(numpy.int8):
                _primitive_awkward_form[model] = awkward.forms.from_json('"int8"')
            elif model == numpy.dtype(numpy.uint8):
                _primitive_awkward_form[model] = awkward.forms.from_json('"uint8"')
            elif model == numpy.dtype(numpy.int16):
                _primitive_awkward_form[model] = awkward.forms.from_json('"int16"')
            elif model == numpy.dtype(numpy.uint16):
                _primitive_awkward_form[model] = awkward.forms.from_json('"uint16"')
            elif model == numpy.dtype(numpy.int32):
                _primitive_awkward_form[model] = awkward.forms.from_json('"int32"')
            elif model == numpy.dtype(numpy.uint32):
                _primitive_awkward_form[model] = awkward.forms.from_json('"uint32"')
            elif model == numpy.dtype(numpy.int64):
                _primitive_awkward_form[model] = awkward.forms.from_json('"int64"')
            elif model == numpy.dtype(numpy.uint64):
                _primitive_awkward_form[model] = awkward.forms.from_json('"uint64"')
            elif model == numpy.dtype(numpy.float32):
                _primitive_awkward_form[model] = awkward.forms.from_json('"float32"')
            elif model == numpy.dtype(numpy.float64):
                _primitive_awkward_form[model] = awkward.forms.from_json('"float64"')
            else:
                raise AssertionError(f"{model!r}: {type(model)}")

        return _primitive_awkward_form[model]

    else:
        return model.awkward_form(file, context)


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
            list(fixed.values()),
            list(fixed.keys()),
            parameters=form.parameters,
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
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.ByteMaskedForm):
        return awkward.forms.ByteMaskedForm(
            form.mask,
            awkward_form_of_iter(awkward, form.content),
            form.valid_when,
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.EmptyForm):
        return awkward.forms.EmptyForm(
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.IndexedForm):
        return awkward.forms.IndexedForm(
            form.index,
            awkward_form_of_iter(awkward, form.content),
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.IndexedOptionForm):
        return awkward.forms.IndexedOptionForm(
            form.index,
            awkward_form_of_iter(awkward, form.content),
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.ListForm):
        return awkward.forms.ListForm(
            form.starts,
            form.stops,
            awkward_form_of_iter(awkward, form.content),
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.ListOffsetForm):
        return awkward.forms.ListOffsetForm(
            form.offsets,
            awkward_form_of_iter(awkward, form.content),
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.NumpyForm):
        if form.parameter("__array__") in ("char", "byte"):
            f = form
        else:
            d = awkward.types.numpytype.primitive_to_dtype(form.primitive)
            if issubclass(d.type, numpy.integer):
                d = numpy.dtype(numpy.int64)
            elif issubclass(d.type, numpy.floating):
                d = numpy.dtype(numpy.float64)
            f = awkward.forms.numpyform.from_dtype(d)
        out = awkward.forms.NumpyForm(
            f.primitive,
            form.inner_shape,
            parameters=form.parameters,
        )
        return out
    elif isinstance(form, awkward.forms.RecordForm):
        contents = {
            k: awkward_form_of_iter(awkward, v) for k, v in form.contents.items()
        }
        return awkward.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.RegularForm):
        return awkward.forms.RegularForm(
            awkward_form_of_iter(awkward, form.content),
            form.size,
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.UnionForm):
        return awkward.forms.UnionForm(
            form.tags,
            form.index,
            [awkward_form_of_iter(awkward, x) for x in form.contents],
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.UnmaskedForm):
        return awkward.forms.UnmaskedForm(
            awkward_form_of_iter(awkward, form.content),
            parameters=form.parameters,
        )
    else:
        raise RuntimeError(f"unrecognized form: {type(form)}")


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
        (code & 0b00000000000000000000000000111111),
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


_regularize_files_braces = re.compile(r"{([^}]*,)*([^}]*)}")
_regularize_files_isglob = re.compile(r"[\*\?\[\]{}]")


def _regularize_files_inner(files, parse_colon, counter, HasBranches):
    files2 = regularize_path(files)

    if isstr(files2) and not isstr(files):
        parse_colon = False
        files = files2

    if isstr(files):
        if parse_colon:
            file_path, object_path = file_object_path_split(files)
        else:
            file_path, object_path = files, None

        parsed_url = urlparse(file_path)

        if parsed_url.scheme.upper() in _remote_schemes:
            yield file_path, object_path

        else:
            expanded = os.path.expanduser(file_path)
            if _regularize_files_isglob.search(expanded) is None:
                yield file_path, object_path

            else:
                matches = list(_regularize_files_braces.finditer(expanded))
                if len(matches) == 0:
                    results = [expanded]
                else:
                    results = []
                    for combination in itertools.product(
                        *[match.group(0)[1:-1].split(",") for match in matches]
                    ):
                        tmp = expanded
                        for c, m in list(zip(combination, matches))[::-1]:
                            tmp = tmp[: m.span()[0]] + c + tmp[m.span()[1] :]
                        results.append(tmp)

                seen = set()
                for result in results:
                    for match in glob.glob(result):
                        if match not in seen:
                            yield match, object_path
                            seen.add(match)

    elif isinstance(files, HasBranches):
        yield files, None

    elif isinstance(files, dict):
        for key, object_path in files.items():
            for file_path, _ in _regularize_files_inner(
                key, False, counter, HasBranches
            ):
                yield file_path, object_path

    elif isinstance(files, Iterable):
        for file in files:
            counter[0] += 1
            for file_path, object_path in _regularize_files_inner(
                file, parse_colon, counter, HasBranches
            ):
                yield file_path, object_path

    else:
        raise TypeError(
            "'files' must be a file path/URL (string or Path), possibly with "
            "a glob pattern (for local files), a dict of "
            "{{path/URL: TTree/TBranch name}}, actual TTree/TBranch objects, or "
            "an iterable of such things, not {0}".format(repr(files))
        )


def regularize_files(files):
    """
    Common code for regularizing the possible file inputs accepted by uproot so they can be used by uproot internal functions.
    """
    from uproot.behaviors.TBranch import HasBranches

    out = []
    seen = set()
    counter = [0]
    for file_path, object_path in _regularize_files_inner(
        files, True, counter, HasBranches
    ):
        if isstr(file_path):
            key = (counter[0], file_path, object_path)
            if key not in seen:
                out.append((file_path, object_path))
                seen.add(key)
        else:
            out.append((file_path, object_path))

    if len(out) == 0:
        raise _file_not_found(files)

    return out


def regularize_object_path(
    file_path, object_path, custom_classes, allow_missing, options
):
    """
    Returns the TTree object from given object and file paths.
    """
    from uproot.behaviors.TBranch import HasBranches, _NoClose
    from uproot.reading import ReadOnlyFile

    if isinstance(file_path, HasBranches):
        return _NoClose(file_path)

    else:
        file = ReadOnlyFile(
            file_path,
            object_cache=None,
            array_cache=None,
            custom_classes=custom_classes,
            **options,  # NOTE: a comma after **options breaks Python 2
        ).root_directory
        if object_path is None:
            trees = file.keys(filter_classname="TTree", cycle=False)
            if len(trees) == 0:
                if allow_missing:
                    return None
                else:
                    raise ValueError(
                        """no TTrees found
in file {}""".format(
                            file_path
                        )
                    )
            elif len(trees) == 1:
                return file[trees[0]]
            else:
                raise ValueError(
                    """TTree object paths must be specified in the 'files' """
                    """as {{\"filenames*.root\": \"path\"}} if any files have """
                    """more than one TTree

    TTrees: {0}

in file {1}""".format(
                        ", ".join(repr(x) for x in trees), file_path
                    )
                )

        else:
            if allow_missing and object_path not in file:
                return None
            return file[object_path]


def _content_cls_from_name(awkward, name):
    if name.endswith("32") or name.endswith("64"):
        name = name[-2:]
    elif name.endswith("U32"):
        name = name[-3:]
    elif name.endswith("8_32") or name.endswith("8_64"):
        name = name[-4:]
    elif name.endswith("8_U32"):
        name = name[-5:]
    return getattr(awkward.contents, name)
