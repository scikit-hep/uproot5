# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import itertools
import json

import numpy

import uproot4.interpretation.jagged
import uproot4.interpretation.strings
import uproot4.interpretation.objects
import uproot4.containers


class Library(object):
    """
    Indicates the type of array to produce.

       * `imported`: The imported library or raises a helpful "how to"
             message if it could not be imported.
       * `empty(shape, dtype)`: Make (possibly temporary) multi-basket storage.
       * `finalize(array, branch, interpretation)`: Convert the internal
             storage form into one appropriate for this library (NumPy array,
             Pandas Series, etc.).
       * `group(arrays, expression_context, how)`: Combine arrays into a group,
             either a generic tuple or a grouping style appropriate for this
             library (NumPy array dict, Awkward RecordArray, etc.).
    """

    @property
    def imported(self):
        raise AssertionError

    def empty(self, shape, dtype):
        return numpy.empty(shape, dtype)

    def finalize(self, array, branch, interpretation):
        raise AssertionError

    def group(self, arrays, expression_context, how):
        if how is tuple:
            return tuple(arrays[name] for name, _ in expression_context)
        elif how is list:
            return [arrays[name] for name, _ in expression_context]
        elif how is dict or how is None:
            return dict((name, arrays[name]) for name, _ in expression_context)
        else:
            raise TypeError(
                "for library {0}, how must be tuple, list, dict, or None (for "
                "dict)".format(self.name)
            )

    def __repr__(self):
        return repr(self.name)

    def __eq__(self, other):
        return type(_libraries[self.name]) is type(_libraries[other.name])  # noqa: E721


class NumPy(Library):
    name = "np"

    @property
    def imported(self):
        import numpy

        return numpy

    def finalize(self, array, branch, interpretation):
        if isinstance(array, uproot4.interpretation.jagged.JaggedArray) and isinstance(
            array.content, uproot4.interpretation.objects.StridedObjectArray,
        ):
            out = numpy.zeros(len(array), dtype=numpy.object)
            for i, x in enumerate(array):
                out[i] = numpy.zeros(len(x), dtype=numpy.object)
                for j, y in enumerate(x):
                    out[i][j] = y
            return out

        elif isinstance(
            array,
            (
                uproot4.interpretation.jagged.JaggedArray,
                uproot4.interpretation.strings.StringArray,
                uproot4.interpretation.objects.ObjectArray,
                uproot4.interpretation.objects.StridedObjectArray,
            ),
        ):
            out = numpy.zeros(len(array), dtype=numpy.object)
            for i, x in enumerate(array):
                out[i] = x
            return out

        else:
            return array


def _strided_to_awkward(awkward1, path, interpretation, data):
    contents = []
    names = []
    for name, member in interpretation.members:
        if not name.startswith("@"):
            p = name
            if len(path) != 0:
                p = path + "/" + name
            if isinstance(member, uproot4.interpretation.objects.AsStridedObjects):
                contents.append(_strided_to_awkward(awkward1, p, member, data))
            else:
                contents.append(awkward1.layout.NumpyArray(numpy.array(data[p])))
            names.append(name)
    parameters = {
        "__record__": uproot4.model.classname_decode(interpretation.model.__name__)[0]
    }
    return awkward1.layout.RecordArray(
        contents, names, len(data), parameters=parameters
    )


# FIXME: _object_to_awkward_json and _awkward_json_to_array are slow functions
# with the right outputs to be replaced by compiled versions in awkward1._io.


def _object_to_awkward_json(form, obj):
    if form["class"] == "NumpyArray":
        return obj

    elif form["class"] == "RecordArray":
        out = {}
        for name, subform in form["contents"].items():
            if not name.startswith("@"):
                out[name] = _object_to_awkward_json(subform, obj.member(name))
        return out

    elif form["class"][:15] == "ListOffsetArray":
        if form["parameters"].get("__array__") == "string":
            return obj

        elif form["parameters"].get("__array__") == "sorted_map":
            key_form = form["content"]["contents"][0]
            value_form = form["content"]["contents"][1]
            return [
                (
                    _object_to_awkward_json(key_form, x),
                    _object_to_awkward_json(value_form, y),
                )
                for x, y in obj.items()
            ]

        else:
            subform = form["content"]
            return [_object_to_awkward_json(subform, x) for x in obj]

    elif form["class"] == "RegularArray":
        subform = form["content"]
        return [_object_to_awkward_json(subform, x) for x in obj]

    else:
        raise AssertionError(form["class"])


def _awkward_p(form):
    out = form["parameters"]
    out.pop("uproot", None)
    return out


def _awkward_json_to_array(awkward1, form, array):
    if form["class"] == "NumpyArray":
        return array

    elif form["class"] == "RecordArray":
        contents = []
        names = []
        for name, subform in form["contents"].items():
            if not name.startswith("@"):
                contents.append(_awkward_json_to_array(awkward1, subform, array[name]))
                names.append(name)
        return awkward1.layout.RecordArray(
            contents, names, len(array), parameters=_awkward_p(form)
        )

    elif form["class"][:15] == "ListOffsetArray":
        if form["parameters"].get("__array__") == "string":
            content = _awkward_json_to_array(awkward1, form["content"], array.content)
            return type(array)(array.offsets, content, parameters=_awkward_p(form))

        elif form["parameters"].get("__array__") == "sorted_map":
            key_form = form["content"]["contents"][0]
            value_form = form["content"]["contents"][1]
            keys = _awkward_json_to_array(awkward1, key_form, array.content["0"])
            values = _awkward_json_to_array(awkward1, value_form, array.content["1"])
            content = awkward1.layout.RecordArray(
                (keys, values),
                None,
                len(array.content),
                parameters=_awkward_p(form["content"]),
            )
            return type(array)(array.offsets, content, parameters=_awkward_p(form))

        else:
            content = _awkward_json_to_array(awkward1, form["content"], array.content)
            return type(array)(array.offsets, content, parameters=_awkward_p(form))

    elif form["class"] == "RegularArray":
        content = _awkward_json_to_array(awkward1, form["content"], array.content)
        return awkward1.layout.RegularArray(
            content, form["size"], parameters=_awkward_p(form)
        )

    else:
        raise AssertionError(form["class"])


class Awkward(Library):
    name = "ak"

    @property
    def imported(self):
        try:
            import awkward1
        except ImportError:
            raise ImportError(
                """install the 'awkward1' package with:

    pip install awkward1"""
            )
        else:
            return awkward1

    def finalize(self, array, branch, interpretation):
        awkward1 = self.imported

        if isinstance(array, uproot4.interpretation.objects.StridedObjectArray):
            return awkward1.Array(
                _strided_to_awkward(awkward1, "", array.interpretation, array.array)
            )

        elif isinstance(
            array, uproot4.interpretation.jagged.JaggedArray
        ) and isinstance(
            array.content, uproot4.interpretation.objects.StridedObjectArray
        ):
            content = _strided_to_awkward(
                awkward1, "", array.content.interpretation, array.content.array
            )
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward1.layout.Index32(array.offsets)
                layout = awkward1.layout.ListOffsetArray32(offsets, content)
            else:
                offsets = awkward1.layout.Index64(array.offsets)
                layout = awkward1.layout.ListOffsetArray64(offsets, content)
            return awkward1.Array(layout)

        elif isinstance(array, uproot4.interpretation.jagged.JaggedArray):
            content = awkward1.from_numpy(array.content, highlevel=False)
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward1.layout.Index32(array.offsets)
                layout = awkward1.layout.ListOffsetArray32(offsets, content)
            else:
                offsets = awkward1.layout.Index64(array.offsets)
                layout = awkward1.layout.ListOffsetArray64(offsets, content)
            return awkward1.Array(layout)

        elif isinstance(array, uproot4.interpretation.strings.StringArray):
            content = awkward1.layout.NumpyArray(
                numpy.frombuffer(array.content, dtype=numpy.dtype(numpy.uint8)),
                parameters={"__array__": "char"},
            )
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward1.layout.Index32(array.offsets)
                layout = awkward1.layout.ListOffsetArray32(
                    offsets, content, parameters={"__array__": "string"}
                )
            elif issubclass(array.offsets.dtype.type, numpy.uint32):
                offsets = awkward1.layout.IndexU32(array.offsets)
                layout = awkward1.layout.ListOffsetArrayU32(
                    offsets, content, parameters={"__array__": "string"}
                )
            elif issubclass(array.offsets.dtype.type, numpy.int64):
                offsets = awkward1.layout.Index64(array.offsets)
                layout = awkward1.layout.ListOffsetArray64(
                    offsets, content, parameters={"__array__": "string"}
                )
            else:
                raise AssertionError(repr(array.offsets.dtype))
            return awkward1.Array(layout)

        elif isinstance(interpretation, uproot4.interpretation.objects.AsObjects):
            try:
                form = json.loads(
                    interpretation.awkward_form(interpretation.branch.file).tojson(
                        verbose=True
                    )
                )
            except uproot4.interpretation.objects.CannotBeAwkward as err:
                raise ValueError(
                    """cannot produce Awkward Arrays for interpretation {0} because

    {1}

instead, try library="np" instead of library="ak"

in file {2}
in object {3}""".format(
                        repr(interpretation),
                        err.because,
                        interpretation.branch.file.file_path,
                        interpretation.branch.object_path,
                    )
                )

            unlabeled = awkward1.from_iter(
                (_object_to_awkward_json(form, x) for x in array), highlevel=False
            )
            return awkward1.Array(_awkward_json_to_array(awkward1, form, unlabeled))

        elif array.dtype.names is not None:
            length, shape = array.shape[0], array.shape[1:]
            array = array.reshape(-1)
            contents = []
            for name in array.dtype.names:
                contents.append(awkward1.layout.NumpyArray(numpy.array(array[name])))
            out = awkward1.layout.RecordArray(contents, array.dtype.names, length)
            for size in shape[::-1]:
                out = awkward1.layout.RegularArray(out, size)
            return awkward1.Array(out)

        else:
            return awkward1.from_numpy(array)

    def group(self, arrays, expression_context, how):
        awkward1 = self.imported

        if how is tuple:
            return tuple(arrays[name] for name, _ in expression_context)
        elif how is list:
            return [arrays[name] for name, _ in expression_context]
        elif how is dict:
            return dict((name, arrays[name]) for name, _ in expression_context)
        elif how is None:
            return awkward1.zip(
                dict((name, arrays[name]) for name, _ in expression_context),
                depth_limit=1,
            )
        elif how == "zip":
            nonjagged = []
            offsets = []
            jaggeds = []
            for name, context in expression_context:
                array = arrays[name]
                if context["is_jagged"]:
                    if len(offsets) == 0:
                        offsets.append(array.layout.offsets)
                        jaggeds.append([name])
                    else:
                        for o, j in zip(offsets, jaggeds):
                            if numpy.array_equal(array.layout.offsets, o):
                                j.append(name)
                                break
                        else:
                            offsets.append(array.layout.offsets)
                            jaggeds.append([name])
                else:
                    nonjagged.append(name)
            out = None
            if len(nonjagged) != 0:
                out = awkward1.zip(
                    dict((name, arrays[name]) for name in nonjagged), depth_limit=1
                )
            for number, jagged in enumerate(jaggeds):
                cut = len(jagged[0])
                for name in jagged:
                    cut = min(cut, len(name))
                    while cut > 0 and (name[:cut] != jagged[0][:cut] or name[cut - 1] not in ("_", ".", "/")):
                        cut -= 1
                    if cut == 0:
                        break
                if cut == 0:
                    common = "jagged{0}".format(number)
                    subarray = awkward1.zip(
                        dict((name, arrays[name]) for name in jagged)
                    )
                else:
                    common = jagged[0][:cut].strip("_./")
                    subarray = awkward1.zip(
                        dict((name[cut:].strip("_./"), arrays[name]) for name in jagged)
                    )
                if out is None:
                    out = awkward1.zip({common: subarray}, depth_limit=1)
                else:
                    for name in jagged:
                        out = awkward1.with_field(out, subarray, common)
            return out
        else:
            raise TypeError(
                'for library {0}, how must be tuple, list, dict, "zip" for '
                "a record array with jagged arrays zipped, if possible, or "
                "None, for an unzipped record array".format(self.name)
            )


def _pandas_rangeindex():
    import pandas

    return getattr(pandas, "RangeIndex", pandas.Int64Index)


def _strided_to_pandas(path, interpretation, data, arrays, columns):
    for name, member in interpretation.members:
        if not name.startswith("@"):
            p = path + (name,)
            if isinstance(member, uproot4.interpretation.objects.AsStridedObjects):
                _strided_to_pandas(p, member, data, arrays, columns)
            else:
                arrays.append(data["/".join(p)])
                columns.append(p)


class Pandas(Library):
    name = "pd"

    @property
    def imported(self):
        try:
            import pandas
        except ImportError:
            raise ImportError(
                """install the 'pandas' package with:

    pip install pandas

or

    conda install pandas"""
            )
        else:
            return pandas

    def finalize(self, array, branch, interpretation):
        pandas = self.imported

        if isinstance(array, uproot4.interpretation.objects.StridedObjectArray):
            arrays = []
            columns = []
            _strided_to_pandas((), array.interpretation, array.array, arrays, columns)
            maxlen = max(len(x) for x in columns)
            if maxlen == 1:
                columns = [x[0] for x in columns]
            else:
                columns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxlen - len(x)) for x in columns]
                )
            out = pandas.DataFrame(dict(zip(columns, arrays)), columns=columns)
            out.leaflist = maxlen != 1
            return out

        elif isinstance(
            array, uproot4.interpretation.jagged.JaggedArray
        ) and isinstance(
            array.content, uproot4.interpretation.objects.StridedObjectArray
        ):
            index = pandas.MultiIndex.from_arrays(
                array.parents_localindex(), names=["entry", "subentry"]
            )
            arrays = []
            columns = []
            _strided_to_pandas(
                (), array.content.interpretation, array.content.array, arrays, columns
            )
            maxlen = max(len(x) for x in columns)
            if maxlen == 1:
                columns = [x[0] for x in columns]
            else:
                columns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxlen - len(x)) for x in columns]
                )
            out = pandas.DataFrame(
                dict(zip(columns, arrays)), columns=columns, index=index
            )
            out.leaflist = maxlen != 1
            return out

        elif isinstance(array, uproot4.interpretation.jagged.JaggedArray):
            index = pandas.MultiIndex.from_arrays(
                array.parents_localindex(), names=["entry", "subentry"]
            )
            return pandas.Series(array.content, index=index)

        elif isinstance(
            array,
            (
                uproot4.interpretation.strings.StringArray,
                uproot4.interpretation.objects.ObjectArray,
            ),
        ):
            out = numpy.zeros(len(array), dtype=numpy.object)
            for i, x in enumerate(array):
                out[i] = x
            return pandas.Series(out)

        elif array.dtype.names is not None and len(array.shape) != 1:
            names = []
            arrays = {}
            for n in array.dtype.names:
                for tup in itertools.product(*[range(d) for d in array.shape[1:]]):
                    name = (n + "".join("[" + str(x) + "]" for x in tup),)
                    names.append(name)
                    arrays[name] = array[n][(slice(None),) + tup]
            out = pandas.DataFrame(arrays, columns=names)
            out.leaflist = True
            return out

        elif array.dtype.names is not None:
            columns = pandas.MultiIndex.from_tuples([(x,) for x in array.dtype.names])
            arrays = dict((y, array[x]) for x, y in zip(array.dtype.names, columns))
            out = pandas.DataFrame(arrays, columns=columns)
            out.leaflist = True
            return out

        elif len(array.shape) != 1:
            names = []
            arrays = {}
            for tup in itertools.product(*[range(d) for d in array.shape[1:]]):
                name = "".join("[" + str(x) + "]" for x in tup)
                names.append(name)
                arrays[name] = array[(slice(None),) + tup]
            out = pandas.DataFrame(arrays, columns=names)
            out.leaflist = False
            return out

        else:
            return pandas.Series(array)

    def _only_series(self, original_arrays, original_names):
        pandas = self.imported
        arrays = {}
        names = []
        for name in original_names:
            if isinstance(original_arrays[name], pandas.Series):
                arrays[name] = original_arrays[name]
                names.append(name)
            else:
                df = original_arrays[name]
                for subname in df.columns:
                    if df.leaflist:
                        if isinstance(subname, tuple):
                            path = (name,) + subname
                        else:
                            path = (name, subname)
                    else:
                        path = name + subname
                    arrays[path] = df[subname]
                    names.append(path)
        return arrays, names

    def group(self, arrays, expression_context, how):
        pandas = self.imported
        names = [name for name, _ in expression_context]
        if how is tuple:
            return tuple(arrays[name] for name in names)
        elif how is list:
            return [arrays[name] for name in names]
        elif how is dict:
            return dict((name, arrays[name]) for name in names)
        elif uproot4._util.isstr(how) or how is None:
            arrays, names = self._only_series(arrays, names)
            if any(isinstance(x, tuple) for x in names):
                names = pandas.MultiIndex.from_tuples(names)
            if all(isinstance(x.index, _pandas_rangeindex()) for x in arrays.values()):
                return pandas.DataFrame(data=arrays, columns=names)
            indexes = []
            groups = []
            for name in names:
                array = arrays[name]
                if isinstance(array.index, pandas.MultiIndex):
                    for index, group in zip(indexes, groups):
                        if numpy.array_equal(array.index, index):
                            group.append(name)
                            break
                    else:
                        indexes.append(array.index)
                        groups.append([name])
            if how is None:
                flat_index = None
                dfs = [[] for x in indexes]
                group_names = [[] for x in indexes]
                for index, group, df, gn in zip(indexes, groups, dfs, group_names):
                    for name in names:
                        array = arrays[name]
                        if isinstance(array.index, _pandas_rangeindex()):
                            if flat_index is None or len(flat_index) != len(
                                array.index
                            ):
                                flat_index = pandas.MultiIndex.from_arrays(
                                    [array.index]
                                )
                            df.append(
                                pandas.Series(array.values, index=flat_index).reindex(
                                    index
                                )
                            )
                            gn.append(name)
                        elif name in group:
                            df.append(array)
                            gn.append(name)
                out = []
                for index, df, gn in zip(indexes, dfs, group_names):
                    out.append(
                        pandas.DataFrame(
                            data=dict(zip(gn, df)), index=index, columns=gn
                        )
                    )
                if len(out) == 1:
                    return out[0]
                else:
                    return tuple(out)
            else:
                out = None
                for index, group in zip(indexes, groups):
                    only = dict((name, arrays[name]) for name in group)
                    df = pandas.DataFrame(data=only, index=index, columns=group)
                    if out is None:
                        out = df
                    else:
                        out = pandas.merge(
                            out, df, how=how, left_index=True, right_index=True
                        )
                flat_names = [
                    name
                    for name in names
                    if isinstance(arrays[name].index, _pandas_rangeindex())
                ]
                if len(flat_names) > 0:
                    flat_index = pandas.MultiIndex.from_arrays(
                        [arrays[flat_names[0]].index]
                    )
                    only = dict(
                        (name, pandas.Series(arrays[name].values, index=flat_index))
                        for name in flat_names
                    )
                    df = pandas.DataFrame(
                        data=only, index=flat_index, columns=flat_names
                    )
                    out = pandas.merge(
                        df.reindex(out.index),
                        out,
                        how=how,
                        left_index=True,
                        right_index=True,
                    )
                return out

        else:
            raise TypeError(
                "for library {0}, how must be tuple, list, dict, str (for "
                "pandas.merge's 'how' parameter, or None (for one or more"
                "DataFrames without merging)".format(self.name)
            )


class CuPy(Library):
    name = "cp"

    @property
    def imported(self):
        try:
            import cupy
        except ImportError:
            raise ImportError(
                """install the 'cupy' package with:

    pip install cupy

or

    conda install cupy"""
            )
        else:
            return cupy

    def empty(self, shape, dtype):
        cupy = self.imported
        return cupy.empty(shape, dtype)

    def finalize(self, array, branch, interpretation):
        cupy = self.imported

        if isinstance(
            array,
            (
                uproot4.interpretation.jagged.JaggedArray,
                uproot4.interpretation.strings.StringArray,
                uproot4.interpretation.objects.ObjectArray,
            ),
        ):
            raise TypeError("jagged arrays and objects are not supported by CuPy")

        else:
            assert isinstance(array, cupy.ndarray)
            return array


_libraries = {
    NumPy.name: NumPy(),
    Awkward.name: Awkward(),
    Pandas.name: Pandas(),
    CuPy.name: CuPy(),
}

_libraries["numpy"] = _libraries[NumPy.name]
_libraries["Numpy"] = _libraries[NumPy.name]
_libraries["NumPy"] = _libraries[NumPy.name]
_libraries["NUMPY"] = _libraries[NumPy.name]

_libraries["awkward1"] = _libraries[Awkward.name]
_libraries["Awkward1"] = _libraries[Awkward.name]
_libraries["AWKWARD1"] = _libraries[Awkward.name]
_libraries["awkward"] = _libraries[Awkward.name]
_libraries["Awkward"] = _libraries[Awkward.name]
_libraries["AWKWARD"] = _libraries[Awkward.name]

_libraries["pandas"] = _libraries[Pandas.name]
_libraries["Pandas"] = _libraries[Pandas.name]
_libraries["PANDAS"] = _libraries[Pandas.name]

_libraries["cupy"] = _libraries[CuPy.name]
_libraries["Cupy"] = _libraries[CuPy.name]
_libraries["CuPy"] = _libraries[CuPy.name]
_libraries["CUPY"] = _libraries[CuPy.name]


def _regularize_library(library):
    if isinstance(library, Library):
        return _libraries[library.name]

    elif isinstance(library, type) and issubclass(library, Library):
        return _libraries[library().name]

    else:
        try:
            return _libraries[library]
        except KeyError:
            raise ValueError("unrecognized library: {0}".format(repr(library)))
