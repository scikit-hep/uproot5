# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module represents external libraries that define "array-like" types so that users can
choose an output format.

The :doc:`uproot.interpretation.library.NumPy` library always works (NumPy is
Uproot's only strict dependency) and outputs NumPy arrays for single arrays
and dict/tuple/list as groups. Objects and jagged arrays are not efficiently
represented, but it provides a zero-dependency least common denominator.

The :doc:`uproot.interpretation.library.Awkward` library is the default and
depends on Awkward Array (``awkward``). It is usually the best option, as it
was designed for Uproot.

The :doc:`uproot.interpretation.library.Pandas` library outputs
``pandas.Series`` for single arrays and ``pandas.DataFrame`` as groups. Objects
are not efficiently represented, but some jagged arrays are encoded as
``pandas.MultiIndex``.

Lazy arrays (:doc:`uproot.behaviors.TBranch.lazy`) can only use the
:doc:`uproot.interpretation.library.Awkward` library.
"""


import gc
import itertools
import json

import numpy

import uproot


def _rename(name, context):
    if context is None or "rename" not in context:
        return name
    else:
        return context["rename"]


class Library:
    """
    Abstract superclass of array-library handlers, for libraries such as NumPy,
    Awkward Array, and Pandas.

    A library is used in the finalization and grouping stages of producing an
    array, converting it from internal representations like
    :doc:`uproot.interpretation.jagged.JaggedArray`,
    :doc:`uproot.interpretation.strings.StringArray`, and
    :doc:`uproot.interpretation.objects.ObjectArray` into the library's
    equivalents. It can also be required for concatenation and other late-stage
    operations on the output arrays.

    Libraries are usually selected by a string name. These names are held in a
    private registry in the :doc:`uproot.interpretation.library` module.
    """

    @property
    def imported(self):
        """
        Attempts to import the library and returns the imported module.
        """
        raise AssertionError

    def empty(self, shape, dtype):
        """
        Args:
            shape (tuple of int): NumPy array ``shape``. (The first item must
                be zero.)
            dtype (``numpy.dtype`` or its constructor argument): NumPy array
                ``dtype``.

        Returns an empty NumPy-like array.
        """
        return numpy.empty(shape, dtype)

    def zeros(self, shape, dtype):
        """
        Args:
            shape (tuple of int): NumPy array ``shape``. (The first item must
                be zero.)
            dtype (``numpy.dtype`` or its constructor argument): NumPy array
                ``dtype``.

        Returns a NumPy-like array of zeros.
        """
        return numpy.zeros(shape, dtype)

    def finalize(self, array, branch, interpretation, entry_start, entry_stop):
        """
        Args:
            array (array): Internal, temporary, trimmed array. If this is a
                NumPy array, it may be identical to the output array.
            branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch``
                that is represented by this array.
            interpretation (:doc:`uproot.interpretation.Interpretation`): The
                interpretation that produced the ``array``.
            entry_start (int): First entry that is included in the output.
            entry_stop (int): FIrst entry that is excluded (one greater than
                the last entry that is included) in the output.

        Create a library-appropriate output array for this temporary ``array``.

        This array would represent one ``TBranch`` (i.e. not a "group").
        """
        raise AssertionError

    def group(self, arrays, expression_context, how):
        """
        Args:
            arrays (dict of str \u2192 array): Mapping from names to finalized
                array objets to combine into a group.
            expression_context (list of (str, dict) tuples): Expression strings
                and a dict of metadata about each.
            how (None, str, or container type): Library-dependent instructions
                for grouping. The only recognized container types are ``tuple``,
                ``list``, and ``dict``. Note that the container *type itself*
                must be passed as ``how``, not an instance of that type (i.e.
                ``how=tuple``, not ``how=()``).

        Combine the finalized ``arrays`` into a library-appropriate group type.
        """
        if how is tuple:
            return tuple(arrays[name] for name, _ in expression_context)
        elif how is list:
            return [arrays[name] for name, _ in expression_context]
        elif how is dict or how is None:
            return {_rename(name, c): arrays[name] for name, c in expression_context}
        else:
            raise TypeError(
                "for library {}, how must be tuple, list, dict, or None (for "
                "dict)".format(self.name)
            )

    def global_index(self, array, global_offset):
        """
        Args:
            array (array): The library-appropriate array whose global index
                needs adjustment.
            global_offset (int): A number to add to the global index of
                ``array`` to correct it.

        Apply *in-place* corrections to the global index of ``array`` by adding
        ``global_offset``.

        Even though the operation is performed *in-place*, this method returns
        the ``array``.
        """
        return array

    def concatenate(self, all_arrays):
        """
        Args:
            all_arrays (list of arrays): A list of library-appropriate arrays
                that need to be concatenated.

        Returns a concatenated version of ``all_arrays``.
        """
        raise AssertionError

    def __repr__(self):
        return repr(self.name)

    def __eq__(self, other):
        return type(_libraries[self.name]) is type(_libraries[other.name])  # noqa: E721


class NumPy(Library):
    """
    A :doc:`uproot.interpretation.library.Library` that presents ``TBranch``
    data as NumPy arrays. The standard name for this library is ``"np"``.

    The single-``TBranch`` form for this library is a ``numpy.ndarray``. If
    the data are non-numerical, they will be converted into Python objects and
    stored in an array with ``dtype="O"``. This is inefficient, but it is the
    minimal-dependency option for Python.

    The "group" behavior for this library is:

    * ``how=dict`` or ``how=None``: a dict of str \u2192 array, mapping the
      names to arrays.
    * ``how=tuple``: a tuple of arrays, in the order requested. (Names are
      lost.)
    * ``how=list``: a list of arrays, in the order requested. (Names are lost.)

    Since NumPy arrays are not indexed, ``global_index`` has no effect.
    """

    name = "np"

    @property
    def imported(self):
        import numpy

        return numpy

    def finalize(self, array, branch, interpretation, entry_start, entry_stop):
        if isinstance(array, uproot.interpretation.jagged.JaggedArray) and isinstance(
            array.content,
            uproot.interpretation.objects.StridedObjectArray,
        ):
            out = numpy.zeros(len(array), dtype=object)
            for i, x in enumerate(array):
                out[i] = numpy.zeros(x.shape, dtype=object)
                for j, y in x.ndenumerate():
                    out[i][j] = y
            return out

        elif isinstance(
            array,
            uproot.interpretation.objects.StridedObjectArray,
        ):
            out = numpy.zeros(array.shape, dtype=object)
            for i, x in array.ndenumerate():
                out[i] = x
            return out

        elif isinstance(
            array,
            (
                uproot.interpretation.jagged.JaggedArray,
                uproot.interpretation.strings.StringArray,
                uproot.interpretation.objects.ObjectArray,
            ),
        ):
            out = numpy.zeros(len(array), dtype=object)
            for i, x in enumerate(array):
                out[i] = x
            return out

        else:
            return array

    def concatenate(self, all_arrays):
        if len(all_arrays) == 0:
            return all_arrays

        if isinstance(all_arrays[0], (tuple, list)):
            keys = range(len(all_arrays[0]))
        elif isinstance(all_arrays[0], dict):
            keys = list(all_arrays[0])
        else:
            raise AssertionError(repr(all_arrays[0]))

        to_concatenate = {k: [] for k in keys}
        for arrays in all_arrays:
            for k in keys:
                to_concatenate[k].append(arrays[k])

        concatenated = {k: numpy.concatenate(to_concatenate[k]) for k in keys}

        if isinstance(all_arrays[0], tuple):
            return tuple(concatenated[k] for k in keys)
        elif isinstance(all_arrays[0], list):
            return [concatenated[k] for k in keys]
        elif isinstance(all_arrays[0], dict):
            return concatenated


def _strided_to_awkward(awkward, path, interpretation, data):
    contents = []
    names = []
    data = data.flatten()
    for name, member in interpretation.members:
        if not name.startswith("@"):
            p = name
            if len(path) != 0:
                p = path + "/" + name
            if isinstance(member, uproot.interpretation.objects.AsStridedObjects):
                contents.append(_strided_to_awkward(awkward, p, member, data))
            else:
                contents.append(
                    awkward.from_numpy(
                        numpy.array(data[p]), regulararray=True, highlevel=False
                    )
                )
            names.append(name)
    parameters = {
        "__record__": uproot.model.classname_decode(interpretation.model.__name__)[0]
    }
    length = len(data) if len(contents) == 0 else None
    out = awkward.layout.RecordArray(contents, names, length, parameters=parameters)
    for dim in reversed(interpretation.inner_shape):
        out = awkward.layout.RegularArray(out, dim)
    return out


# FIXME: _object_to_awkward_json and _awkward_json_to_array are slow functions
# with the right outputs to be replaced by compiled versions in awkward._io.


def _object_to_awkward_json(form, obj):
    if form["class"] == "NumpyArray":
        return obj

    elif form["class"] == "RecordArray":
        out = {}
        for name, subform in form["contents"].items():
            if not name.startswith("@"):
                if obj.has_member(name):
                    out[name] = _object_to_awkward_json(subform, obj.member(name))
                else:
                    out[name] = _object_to_awkward_json(subform, getattr(obj, name))
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


def _awkward_offsets(awkward, form, array):
    if isinstance(array, awkward.layout.EmptyArray):
        if form["offsets"] == "i32":
            return awkward.layout.Index32(numpy.zeros(1, dtype=numpy.int32))
        elif form["offsets"] == "u32":
            return awkward.layout.IndexU32(numpy.zeros(1, dtype=numpy.uint32))
        elif form["offsets"] == "i64":
            return awkward.layout.Index64(numpy.zeros(1, dtype=numpy.int64))
        else:
            raise AssertionError(form["offsets"])
    else:
        if form["offsets"] == "i32":
            return awkward.layout.Index32(
                numpy.asarray(array.offsets, dtype=numpy.int32)
            )
        elif form["offsets"] == "u32":
            return awkward.layout.IndexU32(
                numpy.asarray(array.offsets, dtype=numpy.uint32)
            )
        elif form["offsets"] == "i64":
            return awkward.layout.Index64(
                numpy.asarray(array.offsets, dtype=numpy.int64)
            )
        else:
            raise AssertionError(form["offsets"])


def _awkward_json_to_array(awkward, form, array):
    if form["class"] == "NumpyArray":
        if isinstance(array, awkward.layout.EmptyArray):
            dtype = awkward.forms.Form.fromjson(json.dumps(form)).to_numpy()
            return awkward.layout.NumpyArray(numpy.empty(0, dtype=dtype))
        else:
            return array

    elif form["class"] == "RecordArray":
        contents = []
        names = []
        for name, subform in form["contents"].items():
            if not name.startswith("@"):
                if isinstance(array, awkward.layout.EmptyArray):
                    contents.append(_awkward_json_to_array(awkward, subform, array))
                else:
                    contents.append(
                        _awkward_json_to_array(awkward, subform, array[name])
                    )
                names.append(name)
        length = len(array) if len(contents) == 0 else None
        return awkward.layout.RecordArray(
            contents, names, length, parameters=_awkward_p(form)
        )

    elif form["class"][:15] == "ListOffsetArray":
        if form["parameters"].get("__array__") == "string":
            if isinstance(array, awkward.layout.EmptyArray):
                content = awkward.layout.NumpyArray(
                    numpy.empty(0, dtype=numpy.uint8),
                    parameters=_awkward_p(form["content"]),
                )
                return awkward.layout.ListOffsetArray64(
                    awkward.layout.Index64(numpy.array([0], dtype=numpy.uint8)),
                    content,
                    parameters=_awkward_p(form),
                )
            else:
                content = _awkward_json_to_array(
                    awkward, form["content"], array.content
                )
                return type(array)(array.offsets, content, parameters=_awkward_p(form))

        elif form["parameters"].get("__array__") == "sorted_map":
            offsets = _awkward_offsets(awkward, form, array)
            key_form = form["content"]["contents"][0]
            value_form = form["content"]["contents"][1]
            if isinstance(array, awkward.layout.EmptyArray):
                keys = _awkward_json_to_array(awkward, key_form, array)
                values = _awkward_json_to_array(awkward, value_form, array)
                content = awkward.layout.RecordArray(
                    (keys, values),
                    None,
                    0,
                    parameters=_awkward_p(form["content"]),
                )
            else:
                keys = _awkward_json_to_array(awkward, key_form, array.content["0"])
                values = _awkward_json_to_array(awkward, value_form, array.content["1"])
                length = len(array.content) if len(keys) == 0 else None
                content = awkward.layout.RecordArray(
                    (keys, values),
                    None,
                    length,
                    parameters=_awkward_p(form["content"]),
                )
            cls = getattr(awkward.layout, form["class"])
            return cls(offsets, content, parameters=_awkward_p(form))

        else:
            offsets = _awkward_offsets(awkward, form, array)
            if isinstance(array, awkward.layout.EmptyArray):
                content = _awkward_json_to_array(awkward, form["content"], array)
            else:
                content = _awkward_json_to_array(
                    awkward, form["content"], array.content
                )
            cls = getattr(awkward.layout, form["class"])
            return cls(offsets, content, parameters=_awkward_p(form))

    elif form["class"] == "RegularArray":
        if isinstance(array, awkward.layout.EmptyArray):
            content = _awkward_json_to_array(awkward, form["content"], array)
        else:
            content = _awkward_json_to_array(awkward, form["content"], array.content)
        return awkward.layout.RegularArray(
            content, form["size"], parameters=_awkward_p(form)
        )

    else:
        raise AssertionError(form["class"])


class Awkward(Library):
    """
    A :doc:`uproot.interpretation.library.Library` that presents ``TBranch``
    data as Awkward Arrays. The standard name for this library is ``"ak"``.

    This is the default for all functions that require a
    :doc:`uproot.interpretation.library.Library`, though Uproot does not
    explicitly depend on Awkward Array. If you are confronted with a message
    that Awkward Array is not installed, either install ``awkward`` or
    select another library (likely :doc:`uproot.interpretation.library.NumPy`).

    Both the single-``TBranch`` and "group" forms for this library are
    ``ak.Array``, though groups are always arrays of records. Awkward Array
    was originally developed for Uproot, so the data structures are usually
    optimial for Uproot data.

    The "group" behavior for this library is:

    * ``how=None``: an array of Awkward records.
    * ``how=dict``: a dict of str \u2192 array, mapping the names to arrays.
    * ``how=tuple``: a tuple of arrays, in the order requested. (Names are
      lost.)
    * ``how=list``: a list of arrays, in the order requested. (Names are lost.)

    Since Awkward arrays are not indexed, ``global_index`` has no effect.
    """

    name = "ak"

    @property
    def imported(self):
        return uproot.extras.awkward()

    def finalize(self, array, branch, interpretation, entry_start, entry_stop):
        awkward = self.imported

        if isinstance(array, awkward.layout.Content):
            return awkward.Array(array)

        elif isinstance(array, uproot.interpretation.objects.StridedObjectArray):
            return awkward.Array(
                _strided_to_awkward(awkward, "", array.interpretation, array.array)
            )

        elif isinstance(array, uproot.interpretation.jagged.JaggedArray) and isinstance(
            array.content, uproot.interpretation.objects.StridedObjectArray
        ):
            content = _strided_to_awkward(
                awkward, "", array.content.interpretation, array.content.array
            )
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward.layout.Index32(array.offsets)
                layout = awkward.layout.ListOffsetArray32(offsets, content)
            else:
                offsets = awkward.layout.Index64(array.offsets)
                layout = awkward.layout.ListOffsetArray64(offsets, content)
            return awkward.Array(layout)

        elif isinstance(array, uproot.interpretation.jagged.JaggedArray):
            content = awkward.from_numpy(
                array.content, regulararray=True, highlevel=False
            )
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward.layout.Index32(array.offsets)
                layout = awkward.layout.ListOffsetArray32(offsets, content)
            else:
                offsets = awkward.layout.Index64(array.offsets)
                layout = awkward.layout.ListOffsetArray64(offsets, content)
            return awkward.Array(layout)

        elif isinstance(array, uproot.interpretation.strings.StringArray):
            content = awkward.layout.NumpyArray(
                numpy.frombuffer(array.content, dtype=numpy.dtype(numpy.uint8)),
                parameters={"__array__": "char"},
            )
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward.layout.Index32(array.offsets)
                layout = awkward.layout.ListOffsetArray32(
                    offsets, content, parameters={"__array__": "string"}
                )
            elif issubclass(array.offsets.dtype.type, numpy.uint32):
                offsets = awkward.layout.IndexU32(array.offsets)
                layout = awkward.layout.ListOffsetArrayU32(
                    offsets, content, parameters={"__array__": "string"}
                )
            elif issubclass(array.offsets.dtype.type, numpy.int64):
                offsets = awkward.layout.Index64(array.offsets)
                layout = awkward.layout.ListOffsetArray64(
                    offsets, content, parameters={"__array__": "string"}
                )
            else:
                raise AssertionError(repr(array.offsets.dtype))
            return awkward.Array(layout)

        elif isinstance(interpretation, uproot.interpretation.objects.AsObjects):
            try:
                form = json.loads(
                    interpretation.awkward_form(interpretation.branch.file).tojson(
                        verbose=True
                    )
                )
            except uproot.interpretation.objects.CannotBeAwkward as err:
                raise ValueError(
                    """cannot produce Awkward Arrays for interpretation {} because

    {}

instead, try library="np" instead of library="ak" or globally set uproot.default_library

in file {}
in object {}""".format(
                        repr(interpretation),
                        err.because,
                        interpretation.branch.file.file_path,
                        interpretation.branch.object_path,
                    )
                ) from err

            unlabeled = awkward.from_iter(
                (_object_to_awkward_json(form, x) for x in array), highlevel=False
            )
            return awkward.Array(_awkward_json_to_array(awkward, form, unlabeled))

        elif array.dtype.names is not None:
            length, shape = array.shape[0], array.shape[1:]
            array = array.reshape(-1)
            contents = []
            for name in array.dtype.names:
                contents.append(
                    awkward.from_numpy(
                        numpy.array(array[name]), regulararray=True, highlevel=False
                    )
                )
            if len(contents) != 0:
                length = None
            out = awkward.layout.RecordArray(contents, array.dtype.names, length)
            for size in shape[::-1]:
                out = awkward.layout.RegularArray(out, size)
            return awkward.Array(out)

        else:
            return awkward.from_numpy(array, regulararray=True)

    def group(self, arrays, expression_context, how):
        awkward = self.imported

        if how is tuple:
            return tuple(arrays[name] for name, _ in expression_context)
        elif how is list:
            return [arrays[name] for name, _ in expression_context]
        elif how is dict:
            return {_rename(name, c): arrays[name] for name, c in expression_context}
        elif how is None:
            if len(expression_context) == 0:
                return awkward.Array(awkward.layout.RecordArray([], keys=[]))
            else:
                return awkward.Array(
                    {_rename(name, c): arrays[name] for name, c in expression_context}
                )
        elif how == "zip":
            nonjagged = []
            offsets = []
            jaggeds = []
            renamed_arrays = {}
            for name, context in expression_context:
                array = renamed_arrays[_rename(name, context)] = arrays[name]
                if context["is_jagged"]:
                    if (
                        isinstance(
                            array.layout,
                            (
                                awkward.layout.ListArray32,
                                awkward.layout.ListArrayU32,
                                awkward.layout.ListArray64,
                            ),
                        )
                        or array.layout.offsets[0] != 0
                    ):
                        array_layout = array.layout.toListOffsetArray64(True)
                    else:
                        array_layout = array.layout
                    if len(offsets) == 0:
                        offsets.append(array_layout.offsets)
                        jaggeds.append([_rename(name, context)])
                    else:
                        for o, j in zip(offsets, jaggeds):
                            if numpy.array_equal(array_layout.offsets, o):
                                j.append(_rename(name, context))
                                break
                        else:
                            offsets.append(array_layout.offsets)
                            jaggeds.append([_rename(name, context)])
                else:
                    nonjagged.append(_rename(name, context))

            out = None
            if len(nonjagged) != 0:
                if len(nonjagged) == 0:
                    out = awkward.Array(awkward.layout.RecordArray([], keys=[]))
                else:
                    out = awkward.Array(
                        {name: renamed_arrays[name] for name in nonjagged},
                    )
            for number, jagged in enumerate(jaggeds):
                cut = len(jagged[0])
                for name in jagged:
                    cut = min(cut, len(name))
                    while cut > 0 and (
                        name[:cut] != jagged[0][:cut]
                        or name[cut - 1] not in ("_", ".", "/")
                    ):
                        cut -= 1
                    if cut == 0:
                        break
                if (
                    out is not None
                    and cut != 0
                    and jagged[0][:cut].strip("_./") in awkward.fields(out)
                ):
                    cut = 0
                if cut == 0:
                    common = f"jagged{number}"
                    if len(jagged) == 0:
                        subarray = awkward.Array(
                            awkward.layout.RecordArray([], keys=[])
                        )
                    else:
                        subarray = awkward.zip(
                            {name: renamed_arrays[name] for name in jagged}
                        )
                else:
                    common = jagged[0][:cut].strip("_./")
                    if len(jagged) == 0:
                        subarray = awkward.Array(
                            awkward.layout.RecordArray([], keys=[])
                        )
                    else:
                        subarray = awkward.zip(
                            {
                                name[cut:].strip("_./"): renamed_arrays[name]
                                for name in jagged
                            }
                        )
                if out is None:
                    out = awkward.Array({common: subarray})
                else:
                    out = awkward.with_field(out, subarray, common)

            return out
        else:
            raise TypeError(
                'for library {}, how must be tuple, list, dict, "zip" for '
                "a record array with jagged arrays zipped, if possible, or "
                "None, for an unzipped record array".format(self.name)
            )

    def concatenate(self, all_arrays):
        awkward = self.imported

        if len(all_arrays) == 0:
            return all_arrays

        if isinstance(all_arrays[0], (tuple, list)):
            keys = range(len(all_arrays[0]))
        elif isinstance(all_arrays[0], dict):
            keys = list(all_arrays[0])
        else:
            return awkward.concatenate(all_arrays)

        to_concatenate = {k: [] for k in keys}
        for arrays in all_arrays:
            for k in keys:
                to_concatenate[k].append(arrays[k])

        concatenated = {k: awkward.concatenate(to_concatenate[k]) for k in keys}

        if isinstance(all_arrays[0], tuple):
            return tuple(concatenated[k] for k in keys)
        elif isinstance(all_arrays[0], list):
            return [concatenated[k] for k in keys]
        elif isinstance(all_arrays[0], dict):
            return concatenated


def _is_pandas_rangeindex(pandas, index):
    if hasattr(pandas, "RangeIndex") and isinstance(index, pandas.RangeIndex):
        return True
    if hasattr(index, "is_integer") and index.is_integer():
        return True
    if uproot._util.parse_version(pandas.__version__) < uproot._util.parse_version(
        "1.4.0"
    ) and isinstance(index, pandas.Int64Index):
        return True

    return False


def _strided_to_pandas(path, interpretation, data, arrays, columns):
    for name, member in interpretation.members:
        if not name.startswith("@"):
            p = path + (name,)
            if isinstance(member, uproot.interpretation.objects.AsStridedObjects):
                _strided_to_pandas(p, member, data, arrays, columns)
            else:
                arrays.append(data["/".join(p)])
                columns.append(p)


def _pandas_basic_index(pandas, entry_start, entry_stop):
    if hasattr(pandas, "RangeIndex"):
        return pandas.RangeIndex(entry_start, entry_stop)
    else:
        return pandas.Int64Index(range(entry_start, entry_stop))


def _pandas_only_series(pandas, original_arrays, expression_context):
    arrays = {}
    names = []
    for name, context in expression_context:
        if isinstance(original_arrays[name], pandas.Series):
            arrays[_rename(name, context)] = original_arrays[name]
            names.append(_rename(name, context))
        else:
            df = original_arrays[name]
            for subname in df.columns:
                if df.leaflist:
                    if isinstance(subname, tuple):
                        path = (_rename(name, context),) + subname
                    else:
                        path = (_rename(name, context), subname)
                else:
                    path = _rename(name, context) + subname
                arrays[path] = df[subname]
                names.append(path)
    return arrays, names


def _pandas_memory_efficient(pandas, series, names):
    # Pandas copies the data, so at least feed columns one by one
    gc.collect()
    out = None
    for name in names:
        if out is None:
            out = series[name].to_frame(name=name)
        else:
            out[name] = series[name]
        del series[name]
    if out is None:
        return pandas.DataFrame(data=series, columns=names)
    else:
        return out


class Pandas(Library):
    """
    A :doc:`uproot.interpretation.library.Library` that presents ``TBranch``
    data as Pandas Series and DataFrames. The standard name for this library is
    ``"pd"``.

    The single-``TBranch`` (with a single ``TLeaf``) form for this library is
    ``pandas.Series``, and the "group" form is ``pandas.DataFrame``.

    The "group" behavior for this library is:

    * ``how=None`` or a string: passed to ``pandas.merge`` as its ``how``
      parameter, which would be relevant if jagged arrays with different
      multiplicity are requested.
    * ``how=dict``: a dict of str \u2192 array, mapping the names to
      ``pandas.Series``.
    * ``how=tuple``: a tuple of ``pandas.Series``, in the order requested.
      (Names are assigned to the ``pandas.Series``.)
    * ``how=list``: a list of ``pandas.Series``, in the order requested.
      (Names are assigned to the ``pandas.Series``.)

    Pandas Series and DataFrames are indexed, so ``global_index`` adjusts them.
    """

    name = "pd"

    @property
    def imported(self):
        return uproot.extras.pandas()

    def finalize(self, array, branch, interpretation, entry_start, entry_stop):
        pandas = self.imported

        if isinstance(array, uproot.interpretation.objects.StridedObjectArray):
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
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            out = pandas.DataFrame(
                dict(zip(columns, arrays)), columns=columns, index=index
            )
            out.leaflist = maxlen != 1
            return out

        elif isinstance(array, uproot.interpretation.jagged.JaggedArray) and isinstance(
            array.content, uproot.interpretation.objects.StridedObjectArray
        ):
            index = pandas.MultiIndex.from_arrays(
                array.parents_localindex(entry_start, entry_stop),
                names=["entry", "subentry"],
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

        elif isinstance(array, uproot.interpretation.jagged.JaggedArray):
            index = pandas.MultiIndex.from_arrays(
                array.parents_localindex(entry_start, entry_stop),
                names=["entry", "subentry"],
            )
            return pandas.Series(array.content, index=index)

        elif isinstance(
            array,
            (
                uproot.interpretation.strings.StringArray,
                uproot.interpretation.objects.ObjectArray,
            ),
        ):
            out = numpy.zeros(len(array), dtype=object)
            for i, x in enumerate(array):
                out[i] = x
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            return pandas.Series(out, index=index)

        elif array.dtype.names is not None and len(array.shape) != 1:
            names = []
            arrays = {}
            for n in array.dtype.names:
                for tup in itertools.product(*[range(d) for d in array.shape[1:]]):
                    name = (n + "".join("[" + str(x) + "]" for x in tup),)
                    names.append(name)
                    arrays[name] = array[n][(slice(None),) + tup]
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            out = pandas.DataFrame(arrays, columns=names, index=index)
            out.leaflist = True
            return out

        elif array.dtype.names is not None:
            columns = pandas.MultiIndex.from_tuples([(x,) for x in array.dtype.names])
            arrays = {y: array[x] for x, y in zip(array.dtype.names, columns)}
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            out = pandas.DataFrame(arrays, columns=columns, index=index)
            out.leaflist = True
            return out

        elif len(array.shape) != 1:
            names = []
            arrays = {}
            for tup in itertools.product(*[range(d) for d in array.shape[1:]]):
                name = "".join("[" + str(x) + "]" for x in tup)
                names.append(name)
                arrays[name] = array[(slice(None),) + tup]
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            out = pandas.DataFrame(arrays, columns=names, index=index)
            out.leaflist = False
            return out

        else:
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            return pandas.Series(array, index=index)

    def group(self, arrays, expression_context, how):
        pandas = self.imported

        if how is tuple:
            return tuple(arrays[name] for name, _ in expression_context)

        elif how is list:
            return [arrays[name] for name, _ in expression_context]

        elif how is dict:
            return {_rename(name, c): arrays[name] for name, c in expression_context}

        elif uproot._util.isstr(how) or how is None:
            arrays, names = _pandas_only_series(pandas, arrays, expression_context)

            if any(isinstance(x, tuple) for x in names):
                longest = max(len(x) for x in names if isinstance(x, tuple))
                newarrays, newnames = {}, []
                for x in names:
                    if not isinstance(x, tuple):
                        y = (x,) + ("",) * (longest - 1)
                    else:
                        y = x + ("",) * (longest - len(x))
                    newarrays[y] = arrays[x]
                    newnames.append(y)
                arrays = newarrays
                names = pandas.MultiIndex.from_tuples(newnames)

            if all(_is_pandas_rangeindex(pandas, x.index) for x in arrays.values()):
                return _pandas_memory_efficient(pandas, arrays, names)

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
                        if _is_pandas_rangeindex(pandas, array.index):
                            if flat_index is None or len(flat_index) != len(
                                array.index
                            ):
                                flat_index = pandas.MultiIndex.from_arrays(
                                    [array.index]
                                )
                            # Old versions of Pandas handle the following line poorly:
                            # should we support them?
                            #
                            # >>> pandas.__version__
                            # '0.22.0'
                            # >>> from_index = pandas.MultiIndex.from_tuples([(0,), (1,)])
                            # >>> to_index = pandas.MultiIndex.from_tuples([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])
                            # >>> pandas.Series([1.1, 4.4], index=from_index).reindex(to_index)
                            # 0  0   NaN
                            #    1   NaN
                            #    2   NaN
                            # 1  0   NaN
                            #    1   NaN
                            # dtype: float64
                            #
                            # >>> pandas.__version__
                            # '1.3.2'
                            # >>> from_index = pandas.MultiIndex.from_tuples([(0,), (1,)])
                            # >>> to_index = pandas.MultiIndex.from_tuples([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])
                            # >>> pandas.Series([1.1, 4.4], index=from_index).reindex(to_index)
                            # 0  0    1.1
                            #    1    1.1
                            #    2    1.1
                            # 1  0    4.4
                            #    1    4.4
                            # dtype: float64
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
                    only = {name: arrays[name] for name in group}
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
                    if _is_pandas_rangeindex(pandas, arrays[name].index)
                ]
                if len(flat_names) > 0:
                    flat_index = pandas.MultiIndex.from_arrays(
                        [arrays[flat_names[0]].index]
                    )
                    only = {
                        name: pandas.Series(arrays[name].values, index=flat_index)
                        for name in flat_names
                    }
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
                "for library {}, how must be tuple, list, dict, str (for "
                "pandas.merge's 'how' parameter, or None (for one or more"
                "DataFrames without merging)".format(self.name)
            )

    def global_index(self, arrays, global_offset):
        if isinstance(arrays, tuple):
            return tuple(self.global_index(x, global_offset) for x in arrays)

        if type(arrays.index).__name__ == "MultiIndex":
            if hasattr(arrays.index.levels[0], "arrays"):
                index = arrays.index.levels[0].arrays  # pandas>=0.24.0
            else:
                index = arrays.index.levels[0].values  # pandas<0.24.0
            numpy.add(index, global_offset, out=index)

        elif type(arrays.index).__name__ == "RangeIndex":
            if hasattr(arrays.index, "start") and hasattr(arrays.index, "stop"):
                index_start = arrays.index.start  # pandas>=0.25.0
                index_stop = arrays.index.stop
            else:
                index_start = arrays.index._start  # pandas<0.25.0
                index_stop = arrays.index._stop
            arrays.index = type(arrays.index)(
                index_start + global_offset, index_stop + global_offset
            )

        else:
            if hasattr(arrays.index, "arrays"):
                index = arrays.index.arrays  # pandas>=0.24.0
            else:
                index = arrays.index.values  # pandas<0.24.0
            numpy.add(index, global_offset, out=index)

        return arrays

    def concatenate(self, all_arrays):
        pandas = self.imported

        if len(all_arrays) == 0:
            return all_arrays

        if isinstance(all_arrays[0], (tuple, list)):
            keys = range(len(all_arrays[0]))
        elif isinstance(all_arrays[0], dict):
            keys = list(all_arrays[0])
        else:
            return pandas.concat(all_arrays)

        to_concatenate = {k: [] for k in keys}
        for arrays in all_arrays:
            for k in keys:
                to_concatenate[k].append(arrays[k])

        concatenated = {k: pandas.concat(to_concatenate[k]) for k in keys}

        if isinstance(all_arrays[0], tuple):
            return tuple(concatenated[k] for k in keys)
        elif isinstance(all_arrays[0], list):
            return [concatenated[k] for k in keys]
        elif isinstance(all_arrays[0], dict):
            return concatenated


_libraries = {
    NumPy.name: NumPy(),
    Awkward.name: Awkward(),
    Pandas.name: Pandas(),
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


def _regularize_library(library):
    if isinstance(library, Library):
        if library.name in _libraries:
            return _libraries[library.name]
        else:
            raise ValueError(
                "library {} ({}) cannot be used in this function".format(
                    type(library).__name__, repr(library.name)
                )
            )

    elif isinstance(library, type) and issubclass(library, Library):
        if library().name in _libraries:
            return _libraries[library().name]
        else:
            raise ValueError(
                "library {} ({}) cannot be used in this function".format(
                    library.__name__, repr(library().name)
                )
            )

    else:
        try:
            return _libraries[library]
        except KeyError as err:
            raise ValueError(
                """library {} not recognized (for this function); """
                """try "np" (NumPy), "ak" (Awkward Array), or "pd" (Pandas) """
                """instead""".format(repr(library))
            ) from err


_libraries_lazy = {Awkward.name: _libraries[Awkward.name]}

_libraries_lazy["awkward1"] = _libraries_lazy[Awkward.name]
_libraries_lazy["Awkward1"] = _libraries_lazy[Awkward.name]
_libraries_lazy["AWKWARD1"] = _libraries_lazy[Awkward.name]
_libraries_lazy["awkward"] = _libraries_lazy[Awkward.name]
_libraries_lazy["Awkward"] = _libraries_lazy[Awkward.name]
_libraries_lazy["AWKWARD"] = _libraries_lazy[Awkward.name]


def _regularize_library_lazy(library):
    if isinstance(library, Library):
        if library.name in _libraries_lazy:
            return _libraries_lazy[library.name]
        else:
            raise ValueError(
                "library {} ({}) cannot be used in this function".format(
                    type(library).__name__, repr(library.name)
                )
            )

    elif isinstance(library, type) and issubclass(library, Library):
        if library().name in _libraries_lazy:
            return _libraries_lazy[library().name]
        else:
            raise ValueError(
                "library {} ({}) cannot be used in this function".format(
                    library.__name__, repr(library().name)
                )
            )

    else:
        try:
            return _libraries_lazy[library]
        except KeyError as err:
            raise ValueError(
                """library {} not recognized (for this function); """
                """try "ak" (Awkward Array) """
                """instead""".format(repr(library))
            ) from err
