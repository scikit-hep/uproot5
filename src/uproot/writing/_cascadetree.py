# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This is an internal module for writing TTrees in the "cascading" file writer. TTrees
are more like TDirectories than they are like histograms in that they can create
objects, TBaskets, which have to be allocated through the FreeSegments.

The implementation in this module does not use the TTree infrastructure in
:doc:`uproot.models.TTree`, :doc:`uproot.models.TBranch`, and :doc:`uproot.models.TBasket`,
since the models intended for reading have to adapt to different class versions, but
a writer can always write the same class version, and because writing involves allocating
and sometimes freeing data.

See :doc:`uproot.writing._cascade` for a general overview of the cascading writer concept.
"""


import datetime
import math
import struct
import warnings
from collections.abc import Mapping

import numpy

import uproot.compression
import uproot.const
import uproot.reading
import uproot.serialization

_dtype_to_char = {
    numpy.dtype("bool"): "O",
    numpy.dtype(">i1"): "B",
    numpy.dtype(">u1"): "b",
    numpy.dtype(">i2"): "S",
    numpy.dtype(">u2"): "s",
    numpy.dtype(">i4"): "I",
    numpy.dtype(">u4"): "i",
    numpy.dtype(">i8"): "L",
    numpy.dtype(">u8"): "l",
    numpy.dtype(">f4"): "F",
    numpy.dtype(">f8"): "D",
    numpy.dtype(">U"): "C",
}


class Tree:
    """
    Writes a TTree, including all TBranches, TLeaves, and (upon ``extend``) TBaskets.

    Rather than treating TBranches as a separate object, this *writable* TTree writes
    the whole metadata block in one function, so that interrelationships are easier
    to preserve.

    Writes the following class instance versions:

    - TTree: version 20
    - TBranch: version 13
    - TLeaf: version 2
    - TLeaf*: version 1
    - TBasket: version 3

    The ``write_anew`` method writes the whole tree, possibly for the first time, possibly
    because it has been moved (exceeded its initial allocation of TBasket pointers).

    The ``write_updates`` method rewrites the parts that change when new TBaskets are
    added.

    The ``extend`` method adds a TBasket to every TBranch.

    The ``write_np_basket`` and ``write_jagged_basket`` methods write one TBasket in one
    TBranch, either a rectilinear one from NumPy or a simple jagged array from Awkward Array.

    See `ROOT TTree specification <https://github.com/root-project/root/blob/master/io/doc/TFile/ttree.md>`__.
    """

    def __init__(
        self,
        directory,
        name,
        title,
        branch_types,
        freesegments,
        counter_name,
        field_name,
        initial_basket_capacity,
        resize_factor,
    ):
        self._directory = directory
        self._name = name
        self._title = title
        self._freesegments = freesegments
        self._counter_name = counter_name
        self._field_name = field_name
        self._basket_capacity = initial_basket_capacity
        self._resize_factor = resize_factor

        if isinstance(branch_types, dict):
            branch_types_items = branch_types.items()
        else:
            branch_types_items = branch_types

        if len(branch_types) == 0:
            raise ValueError("TTree must have at least one branch")

        self._branch_data = []
        self._branch_lookup = {}
        for branch_name, branch_type in branch_types_items:
            branch_dict = None
            branch_dtype = None
            branch_datashape = None

            if isinstance(branch_type, Mapping) and all(
                uproot._util.isstr(x) for x in branch_type
            ):
                branch_dict = branch_type

            else:
                try:
                    if uproot._util.from_module(branch_type, "awkward"):
                        raise TypeError
                    if (
                        uproot._util.isstr(branch_type)
                        and branch_type.strip() == "bytes"
                    ):
                        raise TypeError
                    branch_dtype = numpy.dtype(branch_type)

                except TypeError as err:
                    try:
                        awkward = uproot.extras.awkward()
                    except ModuleNotFoundError as err:
                        raise TypeError(
                            f"not a NumPy dtype and 'awkward' cannot be imported: {branch_type!r}"
                        ) from err
                    if isinstance(
                        branch_type,
                        (awkward.types.Type, awkward.types.ArrayType),
                    ):
                        branch_datashape = branch_type
                    else:
                        try:
                            branch_datashape = awkward.types.from_datashape(
                                branch_type, highlevel=False
                            )
                        except Exception:
                            raise TypeError(
                                f"not a NumPy dtype or an Awkward datashape: {branch_type!r}"
                            ) from err
                    if isinstance(branch_datashape, awkward.types.ArrayType):
                        branch_datashape = branch_datashape.content

                    branch_dtype = self._branch_ak_to_np(branch_datashape)

            if branch_dict is not None:
                if branch_name not in self._branch_lookup:
                    self._branch_lookup[branch_name] = len(self._branch_data)
                    self._branch_data.append(
                        {
                            "kind": "record",
                            "name": branch_name,
                            "keys": list(branch_dict),
                        }
                    )

                    for key, content in branch_dict.items():
                        subname = self._field_name(branch_name, key)
                        try:
                            dtype = numpy.dtype(content)
                        except Exception as err:
                            raise TypeError(
                                "values of a dict must be NumPy types\n\n    key {} has type {}".format(
                                    repr(key), repr(content)
                                )
                            ) from err
                        self._branch_lookup[subname] = len(self._branch_data)
                        self._branch_data.append(
                            self._branch_np(subname, content, dtype)
                        )

            elif branch_dtype is not None:
                if branch_name not in self._branch_lookup:
                    self._branch_lookup[branch_name] = len(self._branch_data)
                    self._branch_data.append(
                        self._branch_np(branch_name, branch_type, branch_dtype)
                    )

            else:
                parameters = branch_datashape.parameters
                if parameters is None:
                    parameters = {}

                if parameters.get("__array__") == "string":
                    if branch_name not in self._branch_lookup:
                        self._branch_lookup[branch_name] = len(self._branch_data)
                        self._branch_data.append(
                            self._branch_np(branch_name, branch_type, numpy.dtype(str))
                        )

                elif parameters.get("__array__") == "bytes":
                    raise NotImplementedError("array of bytes")

                # 'awkward' is not in namespace
                elif type(branch_datashape).__name__ == "ListType":
                    content = branch_datashape.content

                    counter_name = self._counter_name(branch_name)
                    counter_dtype = numpy.dtype(numpy.int32)
                    counter = self._branch_np(
                        counter_name, counter_dtype, counter_dtype, kind="counter"
                    )
                    if counter_name in self._branch_lookup:
                        # counters always replace non-counters
                        del self._branch_data[self._branch_lookup[counter_name]]
                    self._branch_lookup[counter_name] = len(self._branch_data)
                    self._branch_data.append(counter)

                    if type(content).__name__ == "RecordType":
                        if hasattr(content, "contents"):
                            contents = content.contents
                        else:
                            contents = content.fields()
                        keys = content.fields
                        if callable(keys):
                            keys = keys()
                        if keys is None:
                            keys = [str(x) for x in range(len(contents))]

                        if branch_name not in self._branch_lookup:
                            self._branch_lookup[branch_name] = len(self._branch_data)
                            self._branch_data.append(
                                {"kind": "record", "name": branch_name, "keys": keys}
                            )

                            for key, cont in zip(keys, contents):
                                subname = self._field_name(branch_name, key)
                                dtype = self._branch_ak_to_np(cont)
                                if dtype is None:
                                    raise TypeError(
                                        "fields of a record must be NumPy types, though the record itself may be in a jagged array\n\n    field {} has type {}".format(
                                            repr(key), str(cont)
                                        )
                                    )
                                if subname not in self._branch_lookup:
                                    self._branch_lookup[subname] = len(
                                        self._branch_data
                                    )
                                    self._branch_data.append(
                                        self._branch_np(
                                            subname, cont, dtype, counter=counter
                                        )
                                    )

                    else:
                        dt = self._branch_ak_to_np(content)
                        if dt is None:
                            raise TypeError(
                                "cannot write Awkward Array type to ROOT file:\n\n    {}".format(
                                    str(branch_datashape)
                                )
                            )
                        if branch_name not in self._branch_lookup:
                            self._branch_lookup[branch_name] = len(self._branch_data)
                            self._branch_data.append(
                                self._branch_np(branch_name, dt, dt, counter=counter)
                            )

                elif type(branch_datashape).__name__ == "RecordType":
                    if hasattr(branch_datashape, "contents"):
                        contents = branch_datashape.contents
                    else:
                        contents = branch_datashape.fields()
                    keys = branch_datashape.fields
                    if callable(keys):
                        keys = keys()
                    if keys is None:
                        keys = [str(x) for x in range(len(contents))]

                    if branch_name not in self._branch_lookup:
                        self._branch_lookup[branch_name] = len(self._branch_data)
                        self._branch_data.append(
                            {"kind": "record", "name": branch_name, "keys": keys}
                        )

                        for key, content in zip(keys, contents):
                            subname = self._field_name(branch_name, key)
                            dtype = self._branch_ak_to_np(content)
                            if dtype is None:
                                raise TypeError(
                                    "fields of a record must be NumPy types, though the record itself may be in a jagged array\n\n    field {} has type {}".format(
                                        repr(key), str(content)
                                    )
                                )
                            if subname not in self._branch_lookup:
                                self._branch_lookup[subname] = len(self._branch_data)
                                self._branch_data.append(
                                    self._branch_np(subname, content, dtype)
                                )

                else:
                    raise TypeError(
                        "cannot write Awkward Array type to ROOT file:\n\n    {}".format(
                            str(branch_datashape)
                        )
                    )

        self._num_entries = 0
        self._num_baskets = 0

        self._metadata_start = None
        self._metadata = {
            "fTotBytes": 0,
            "fZipBytes": 0,
            "fSavedBytes": 0,
            "fFlushedBytes": 0,
            "fWeight": 1.0,
            "fTimerInterval": 0,
            "fScanField": 25,
            "fUpdate": 0,
            "fDefaultEntryOffsetLen": 1000,
            "fNClusterRange": 0,
            "fMaxEntries": 1000000000000,
            "fMaxEntryLoop": 1000000000000,
            "fMaxVirtualSize": 0,
            "fAutoSave": -300000000,
            "fAutoFlush": -30000000,
            "fEstimate": 1000000,
        }
        self._key = None

    def _branch_ak_to_np(self, branch_datashape):
        if type(branch_datashape).__name__ == "UnknownType":
            return numpy.dtype("float64")
        elif type(branch_datashape).__name__ == "NumpyType":
            return numpy.dtype(branch_datashape.primitive)
        elif type(branch_datashape).__name__ == "PrimitiveType":
            return numpy.dtype(branch_datashape.dtype)
        elif type(branch_datashape).__name__ == "RegularType":
            content = self._branch_ak_to_np(branch_datashape.content)
            if content is None:
                return None
            elif content.subdtype is None:
                dtype, shape = content, ()
            else:
                dtype, shape = content.subdtype
            return numpy.dtype((dtype, (branch_datashape.size, *shape)))
        else:
            return None

    def _branch_np(
        self, branch_name, branch_type, branch_dtype, counter=None, kind="normal"
    ):
        branch_dtype = branch_dtype.newbyteorder(">")

        if branch_dtype.subdtype is None:
            branch_shape = ()
        else:
            branch_dtype, branch_shape = branch_dtype.subdtype

        letter = _dtype_to_char.get(branch_dtype)
        if letter is None:
            raise TypeError(f"cannot write NumPy dtype {branch_dtype} in TTree")

        if branch_shape == ():
            dims = ""
        else:
            dims = "".join("[" + str(x) + "]" for x in branch_shape)

        title = f"{branch_name}{dims}/{letter}"

        return {
            "fName": branch_name,
            "branch_type": branch_type,
            "kind": kind,
            "counter": counter,
            "dtype": branch_dtype,
            "shape": branch_shape,
            "fTitle": title,
            "compression": self._directory.freesegments.fileheader.compression,
            "fBasketSize": 32000,
            "fEntryOffsetLen": 0 if counter is None else 1000,
            "fOffset": 0,
            "fSplitLevel": 0,
            "fFirstEntry": 0,
            "fTotBytes": 0,
            "fZipBytes": 0,
            "fBasketBytes": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype1
            ),
            "fBasketEntry": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype2
            ),
            "fBasketSeek": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype3
            ),
            "arrays_write_start": 0,
            "arrays_write_stop": 0,
            "metadata_start": None,
            "basket_metadata_start": None,
            "tleaf_reference_number": None,
            "tleaf_maximum_value": 0,
            "tleaf_special_struct": None,
        }

    def __repr__(self):
        return "{}({}, {}, {}, {}, {}, {}, {})".format(
            type(self).__name__,
            self._directory,
            self._name,
            self._title,
            [(datum["fName"], datum["branch_type"]) for datum in self._branch_data],
            self._freesegments,
            self._basket_capacity,
            self._resize_factor,
        )

    @property
    def directory(self):
        return self._directory

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._key.name

    @property
    def title(self):
        return self._key.title

    @property
    def branch_types(self):
        return self._branch_types

    @property
    def freesegments(self):
        return self._freesegments

    @property
    def counter_name(self):
        return self._counter_name

    @property
    def field_name(self):
        return self._field_name

    @property
    def basket_capacity(self):
        return self._basket_capacity

    @property
    def resize_factor(self):
        return self._resize_factor

    @property
    def location(self):
        return self._key.location

    @property
    def num_entries(self):
        return self._num_entries

    @property
    def num_baskets(self):
        return self._num_baskets

    def extend(self, file, sink, data):
        # expand capacity if this would REACH (not EXCEED) the existing capacity
        # that's because completely a full fBasketEntry has nowhere to put the
        # number of entries in the last basket (it's a fencepost principle thing),
        # forcing ROOT and Uproot to look it up from the basket header.
        if self._num_baskets >= self._basket_capacity - 1:
            self._basket_capacity = max(
                self._basket_capacity + 1,
                int(math.ceil(self._basket_capacity * self._resize_factor)),
            )

            for datum in self._branch_data:
                if datum["kind"] == "record":
                    continue

                fBasketBytes = datum["fBasketBytes"]
                fBasketEntry = datum["fBasketEntry"]
                fBasketSeek = datum["fBasketSeek"]
                datum["fBasketBytes"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype1
                )
                datum["fBasketEntry"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype2
                )
                datum["fBasketSeek"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype3
                )
                datum["fBasketBytes"][: len(fBasketBytes)] = fBasketBytes
                datum["fBasketEntry"][: len(fBasketEntry)] = fBasketEntry
                datum["fBasketSeek"][: len(fBasketSeek)] = fBasketSeek
                datum["fBasketEntry"][len(fBasketEntry)] = self._num_entries

            oldloc = start = self._key.location
            stop = start + self._key.num_bytes + self._key.compressed_bytes

            self.write_anew(sink)

            newloc = self._key.seek_location
            file._move_tree(oldloc, newloc)

            self._freesegments.release(start, stop)
            sink.set_file_length(self._freesegments.fileheader.end)
            sink.flush()

        provided = None

        if uproot._util.from_module(data, "pandas"):
            import pandas

            if isinstance(
                data, pandas.DataFrame
            ) and uproot._util.pandas_has_attr_is_numeric(pandas)(data.index):
                provided = dataframe_to_dict(data)

        if uproot._util.from_module(data, "awkward"):
            try:
                awkward = uproot.extras.awkward()
            except ModuleNotFoundError as err:
                raise TypeError(
                    f"an Awkward Array was provided, but 'awkward' cannot be imported: {data!r}"
                ) from err

            if isinstance(data, awkward.Array):
                if data.ndim > 1 and not data.layout.purelist_isregular:
                    provided = {
                        self._counter_name(""): numpy.asarray(
                            awkward.num(data, axis=1), dtype=">u4"
                        )
                    }
                else:
                    provided = {}
                for k, v in zip(awkward.fields(data), awkward.unzip(data)):
                    provided[k] = v

        if isinstance(data, numpy.ndarray) and data.dtype.fields is not None:
            provided = recarray_to_dict(data)

        if provided is None:
            if not isinstance(data, Mapping) or not all(
                uproot._util.isstr(x) for x in data
            ):
                raise TypeError(
                    "'extend' requires a mapping from branch name (str) to arrays"
                )

            provided = {}
            for k, v in data.items():
                if not uproot._util.from_module(
                    v, "pandas"
                ) and not uproot._util.from_module(v, "awkward"):
                    if not hasattr(v, "dtype") and not isinstance(v, Mapping):
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter(
                                    "error", category=numpy.VisibleDeprecationWarning
                                )
                                v = numpy.array(v)  # noqa: PLW2901 (overwriting v)
                            if v.dtype == numpy.dtype("O"):
                                raise Exception
                        except (numpy.VisibleDeprecationWarning, Exception):
                            try:
                                awkward = uproot.extras.awkward()
                            except ModuleNotFoundError as err:
                                raise TypeError(
                                    f"NumPy dtype would be dtype('O'), so we won't use NumPy, but 'awkward' cannot be imported: {k}: {type(v)}"
                                ) from err
                            v = awkward.from_iter(v)  # noqa: PLW2901 (overwriting v)

                    if getattr(v, "dtype", None) == numpy.dtype("O"):
                        try:
                            awkward = uproot.extras.awkward()
                        except ModuleNotFoundError as err:
                            raise TypeError(
                                f"NumPy dtype is dtype('O'), so we won't use NumPy, but 'awkward' cannot be imported: {k}: {type(v)}"
                            ) from err
                        v = awkward.from_iter(v)  # noqa: PLW2901 (overwriting v)

                if uproot._util.from_module(v, "awkward"):
                    try:
                        awkward = uproot.extras.awkward()
                    except ModuleNotFoundError as err:
                        raise TypeError(
                            f"an Awkward Array was provided, but 'awkward' cannot be imported: {k}: {type(v)}"
                        ) from err
                    if (
                        isinstance(v, awkward.Array)
                        and v.ndim > 1
                        and not v.layout.purelist_isregular
                    ):
                        kk = self._counter_name(k)
                        vv = numpy.asarray(awkward.num(v, axis=1), dtype=">u4")
                        if kk in provided and not numpy.array_equal(vv, provided[kk]):
                            raise ValueError(
                                f"branch {kk!r} provided both as an explicit array and generated as a counter, and they disagree"
                            )
                        provided[kk] = vv

                if k in provided and not numpy.array_equal(v, provided[k]):
                    raise ValueError(
                        f"branch {kk!r} provided both as an explicit array and generated as a counter, and they disagree"
                    )
                provided[k] = v

        actual_branches = {}
        for datum in self._branch_data:
            if datum["kind"] == "record":
                if datum["name"] in provided:
                    recordarray = provided.pop(datum["name"])

                    if uproot._util.from_module(recordarray, "pandas"):
                        import pandas

                        if isinstance(recordarray, pandas.DataFrame):
                            tmp = {"index": recordarray.index.values}
                            for column in recordarray.columns:
                                tmp[column] = recordarray[column]
                            recordarray = tmp

                    for key in datum["keys"]:
                        provided[self._field_name(datum["name"], key)] = recordarray[
                            key
                        ]

                elif datum["name"] == "":
                    for key in datum["keys"]:
                        provided[self._field_name(datum["name"], key)] = provided.pop(
                            key
                        )

                else:
                    raise ValueError(
                        "'extend' must be given an array for every branch; missing {}".format(
                            repr(datum["name"])
                        )
                    )

            else:
                if datum["fName"] in provided:
                    actual_branches[datum["fName"]] = provided.pop(datum["fName"])
                else:
                    raise ValueError(
                        "'extend' must be given an array for every branch; missing {}".format(
                            repr(datum["fName"])
                        )
                    )

        if len(provided) != 0:
            raise ValueError(
                "'extend' was given data that do not correspond to any branch: {}".format(
                    ", ".join(repr(x) for x in provided)
                )
            )

        tofill = []
        num_entries = None
        for branch_name, branch_array in actual_branches.items():
            if num_entries is None:
                num_entries = len(branch_array)
            elif num_entries != len(branch_array):
                raise ValueError(
                    "'extend' must fill every branch with the same number of entries; {} has {} entries".format(
                        repr(branch_name),
                        len(branch_array),
                    )
                )

            datum = self._branch_data[self._branch_lookup[branch_name]]
            if datum["kind"] == "record":
                continue

            if datum["counter"] is None:
                if datum["dtype"] == ">U0":
                    lengths = numpy.asarray(awkward.num(branch_array.layout))
                    which_big = lengths >= 255

                    lengths_extension_offsets = numpy.empty(
                        len(branch_array.layout) + 1, numpy.int64
                    )
                    lengths_extension_offsets[0] = 0
                    numpy.cumsum(which_big * 4, out=lengths_extension_offsets[1:])

                    lengths_extension = awkward.contents.ListOffsetArray(
                        awkward.index.Index64(lengths_extension_offsets),
                        awkward.contents.NumpyArray(
                            lengths[which_big].astype(">u4").view("u1")
                        ),
                    )

                    lengths[which_big] = 255

                    leafc_data_awkward = awkward.concatenate(
                        [
                            lengths.reshape(-1, 1).astype("u1"),
                            lengths_extension,
                            awkward.without_parameters(branch_array.layout),
                        ],
                        axis=1,
                    )

                    big_endian = numpy.asarray(awkward.flatten(leafc_data_awkward))
                    big_endian_offsets = (
                        lengths_extension_offsets
                        + numpy.asarray(branch_array.layout.offsets)
                        + numpy.arange(len(branch_array.layout.offsets))
                    ).astype(">i4", copy=True)
                    tofill.append(
                        (
                            branch_name,
                            datum["compression"],
                            big_endian,
                            big_endian_offsets,
                        )
                    )
                else:
                    big_endian = uproot._util.ensure_numpy(branch_array).astype(
                        datum["dtype"]
                    )
                    if big_endian.shape != (len(branch_array),) + datum["shape"]:
                        raise ValueError(
                            "'extend' must fill branches with a consistent shape: has {}, trying to fill with {}".format(
                                datum["shape"],
                                big_endian.shape[1:],
                            )
                        )
                    tofill.append((branch_name, datum["compression"], big_endian, None))
                    if datum["kind"] == "counter":
                        datum["tleaf_maximum_value"] = max(
                            big_endian.max(), datum["tleaf_maximum_value"]
                        )

            else:
                try:
                    awkward = uproot.extras.awkward()
                except ModuleNotFoundError as err:
                    raise TypeError(
                        f"a jagged array was provided (possibly as an iterable), but 'awkward' cannot be imported: {branch_name}: {branch_array!r}"
                    ) from err
                layout = branch_array.layout
                while not isinstance(layout, awkward.contents.ListOffsetArray):
                    if isinstance(layout, awkward.contents.IndexedArray):
                        layout = layout.project()

                    elif isinstance(layout, awkward.contents.ListArray):
                        layout = layout.to_ListOffsetArray64(False)

                    else:
                        raise AssertionError(
                            "how did this pass the type check?\n\n" + repr(layout)
                        )

                content = layout.content
                offsets = numpy.asarray(layout.offsets)

                if offsets[0] != 0:
                    content = content[offsets[0] :]
                    offsets = offsets - offsets[0]
                if len(content) > offsets[-1]:
                    content = content[: offsets[-1]]

                shape = [len(content)]
                while not isinstance(content, awkward.contents.NumpyArray):
                    if isinstance(content, awkward.contents.IndexedArray):
                        content = content.project()

                    elif isinstance(content, awkward.contents.EmptyArray):
                        content = content.to_NumpyArray(dtype=numpy.float64)

                    elif isinstance(content, awkward.contents.RegularArray):
                        shape.append(content.size)
                        content = content.content

                    else:
                        raise AssertionError(
                            "how did this pass the type check?\n\n" + repr(content)
                        )

                big_endian = numpy.asarray(content.data, dtype=datum["dtype"])
                shape = tuple(shape) + big_endian.shape[1:]

                if shape[1:] != datum["shape"]:
                    raise ValueError(
                        "'extend' must fill branches with a consistent shape: has {}, trying to fill with {}".format(
                            datum["shape"],
                            shape[1:],
                        )
                    )
                big_endian_offsets = offsets.astype(">i4", copy=True)

                tofill.append(
                    (
                        branch_name,
                        datum["compression"],
                        big_endian.reshape(-1),
                        big_endian_offsets,
                    )
                )

        # actually write baskets into the file
        uncompressed_bytes = 0
        compressed_bytes = 0
        for branch_name, compression, big_endian, big_endian_offsets in tofill:
            datum = self._branch_data[self._branch_lookup[branch_name]]

            if datum["dtype"] == ">U0":
                totbytes, zipbytes, location = self.write_string_basket(
                    sink, branch_name, compression, big_endian, big_endian_offsets
                )
                datum["fEntryOffsetLen"] = 4 * (len(big_endian_offsets) - 1)

            elif big_endian_offsets is None:
                totbytes, zipbytes, location = self.write_np_basket(
                    sink, branch_name, compression, big_endian
                )
            else:
                totbytes, zipbytes, location = self.write_jagged_basket(
                    sink, branch_name, compression, big_endian, big_endian_offsets
                )
                datum["fEntryOffsetLen"] = 4 * (len(big_endian_offsets) - 1)
            uncompressed_bytes += totbytes
            compressed_bytes += zipbytes

            datum["fTotBytes"] += totbytes
            datum["fZipBytes"] += zipbytes

            datum["fBasketBytes"][self._num_baskets] = zipbytes

            if self._num_baskets + 1 < self._basket_capacity:
                fBasketEntry = datum["fBasketEntry"]
                i = self._num_baskets
                fBasketEntry[i + 1] = num_entries + fBasketEntry[i]

            datum["fBasketSeek"][self._num_baskets] = location

            datum["arrays_write_stop"] = self._num_baskets + 1

        # update TTree metadata in file
        self._num_entries += num_entries
        self._num_baskets += 1
        self._metadata["fTotBytes"] += uncompressed_bytes
        self._metadata["fZipBytes"] += compressed_bytes

        self.write_updates(sink)

    def write_anew(self, sink):
        key_num_bytes = uproot.reading._key_format_big.size + 6
        name_asbytes = self._name.encode(errors="surrogateescape")
        title_asbytes = self._title.encode(errors="surrogateescape")
        key_num_bytes += (1 if len(name_asbytes) < 255 else 5) + len(name_asbytes)
        key_num_bytes += (1 if len(title_asbytes) < 255 else 5) + len(title_asbytes)

        out = [None]
        ttree_header_index = 0

        tobject = uproot.models.TObject.Model_TObject.empty()
        tnamed = uproot.models.TNamed.Model_TNamed.empty()
        tnamed._bases.append(tobject)
        tnamed._members["fTitle"] = self._title
        tnamed._serialize(out, True, self._name, uproot.const.kMustCleanup)

        # TAttLine v2, fLineColor: 602 fLineStyle: 1 fLineWidth: 1
        # TAttFill v2, fFillColor: 0, fFillStyle: 1001
        # TAttMarker v2, fMarkerColor: 1, fMarkerStyle: 1, fMarkerSize: 1.0
        out.append(
            b"@\x00\x00\x08\x00\x02\x02Z\x00\x01\x00\x01"
            b"@\x00\x00\x06\x00\x02\x00\x00\x03\xe9"
            b"@\x00\x00\n\x00\x02\x00\x01\x00\x01?\x80\x00\x00"
        )

        metadata_out_index = len(out)
        out.append(
            uproot.models.TTree._ttree20_format1.pack(
                self._num_entries,
                self._metadata["fTotBytes"],
                self._metadata["fZipBytes"],
                self._metadata["fSavedBytes"],
                self._metadata["fFlushedBytes"],
                self._metadata["fWeight"],
                self._metadata["fTimerInterval"],
                self._metadata["fScanField"],
                self._metadata["fUpdate"],
                self._metadata["fDefaultEntryOffsetLen"],
                self._metadata["fNClusterRange"],
                self._metadata["fMaxEntries"],
                self._metadata["fMaxEntryLoop"],
                self._metadata["fMaxVirtualSize"],
                self._metadata["fAutoSave"],
                self._metadata["fAutoFlush"],
                self._metadata["fEstimate"],
            )
        )

        # speedbump (0), fClusterRangeEnd (empty array),
        # speedbump (0), fClusterSize (empty array)
        # fIOFeatures (TIOFeatures)
        out.append(b"\x00\x00@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00")

        tleaf_reference_numbers = []

        tobjarray_of_branches_index = len(out)
        out.append(None)

        num_branches = sum(
            0 if datum["kind"] == "record" else 1 for datum in self._branch_data
        )

        # TObjArray header with fName: ""
        out.append(b"\x00\x01\x00\x00\x00\x00\x03\x00@\x00\x00")
        out.append(
            uproot.models.TObjArray._tobjarray_format1.pack(
                num_branches,  # TObjArray fSize
                0,  # TObjArray fLowerBound
            )
        )

        for datum in self._branch_data:
            if datum["kind"] == "record":
                continue

            any_tbranch_index = len(out)
            out.append(None)
            out.append(b"TBranch\x00")

            tbranch_index = len(out)
            out.append(None)

            tbranch_tobject = uproot.models.TObject.Model_TObject.empty()
            tbranch_tnamed = uproot.models.TNamed.Model_TNamed.empty()
            tbranch_tnamed._bases.append(tbranch_tobject)
            tbranch_tnamed._members["fTitle"] = datum["fTitle"]
            tbranch_tnamed._serialize(
                out, True, datum["fName"], numpy.uint32(0x00400000)
            )

            # TAttFill v2, fFillColor: 0, fFillStyle: 1001
            out.append(b"@\x00\x00\x06\x00\x02\x00\x00\x03\xe9")

            assert sum(1 if x is None else 0 for x in out) == 4
            datum["metadata_start"] = (6 + 6 + 8 + 6) + sum(
                len(x) for x in out if x is not None
            )

            # Lie about the compression level so that ROOT checks and does the right thing.
            # https://github.com/root-project/root/blob/87a998d48803bc207288d90038e60ff148827664/tree/tree/src/TBasket.cxx#L560-L578
            # Without this, when small buffers are left uncompressed, ROOT complains about them not being compressed.
            # (I don't know where the "no, really, this is uncompressed" bit is.)
            fCompress = 0

            out.append(
                uproot.models.TBranch._tbranch13_format1.pack(
                    fCompress,
                    datum["fBasketSize"],
                    datum["fEntryOffsetLen"],
                    self._num_baskets,  # fWriteBasket
                    self._num_entries,  # fEntryNumber
                )
            )

            # fIOFeatures (TIOFeatures)
            out.append(b"@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00")

            out.append(
                uproot.models.TBranch._tbranch13_format2.pack(
                    datum["fOffset"],
                    self._basket_capacity,  # fMaxBaskets
                    datum["fSplitLevel"],
                    self._num_entries,  # fEntries
                    datum["fFirstEntry"],
                    datum["fTotBytes"],
                    datum["fZipBytes"],
                )
            )

            # empty TObjArray of TBranches
            out.append(
                b"@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )

            subtobjarray_of_leaves_index = len(out)
            out.append(None)

            # TObjArray header with fName: "", fSize: 1, fLowerBound: 0
            out.append(
                b"\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00"
            )

            absolute_location = key_num_bytes + sum(
                len(x) for x in out if x is not None
            )
            absolute_location += 8 + 6 * (sum(1 if x is None else 0 for x in out) - 1)
            datum["tleaf_reference_number"] = absolute_location + 2
            tleaf_reference_numbers.append(datum["tleaf_reference_number"])

            subany_tleaf_index = len(out)
            out.append(None)

            letter = _dtype_to_char[datum["dtype"]]
            letter_upper = letter.upper()
            out.append(("TLeaf" + letter_upper).encode() + b"\x00")
            if letter_upper == "O":
                special_struct = uproot.models.TLeaf._tleafO1_format1
            elif letter_upper == "B":
                special_struct = uproot.models.TLeaf._tleafb1_format1
            elif letter_upper == "S":
                special_struct = uproot.models.TLeaf._tleafs1_format1
            elif letter_upper == "I":
                special_struct = uproot.models.TLeaf._tleafi1_format1
            elif letter_upper == "G":
                special_struct = uproot.models.TLeaf._tleafl1_format0
            elif letter_upper == "L":
                special_struct = uproot.models.TLeaf._tleafl1_format0
            elif letter_upper == "F":
                special_struct = uproot.models.TLeaf._tleaff1_format1
            elif letter_upper == "D":
                special_struct = uproot.models.TLeaf._tleafd1_format1
            elif letter_upper == "C":
                special_struct = uproot.models.TLeaf._tleafc1_format1

            fLenType = datum["dtype"].itemsize
            fIsUnsigned = letter != letter_upper

            if datum["shape"] == ():
                dims = ""
            else:
                dims = "".join("[" + str(x) + "]" for x in datum["shape"])

            if datum["counter"] is not None:
                dims = "[" + datum["counter"]["fName"] + "]" + dims

            # single TLeaf
            leaf_name = datum["fName"].encode(errors="surrogateescape")
            leaf_title = (datum["fName"] + dims).encode(errors="surrogateescape")
            leaf_name_length = (1 if len(leaf_name) < 255 else 5) + len(leaf_name)
            leaf_title_length = (1 if len(leaf_title) < 255 else 5) + len(leaf_title)

            leaf_header = numpy.array(
                [
                    64,
                    0,
                    0,
                    76,
                    0,
                    1,
                    64,
                    0,
                    0,
                    54,
                    0,
                    2,
                    64,
                    0,
                    0,
                    30,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    3,
                    0,
                    0,
                    0,
                ],
                numpy.uint8,
            )
            tmp = leaf_header[0:4].view(">u4")
            tmp[:] = (
                numpy.uint32(
                    42 + leaf_name_length + leaf_title_length + special_struct.size
                )
                | uproot.const.kByteCountMask
            )
            tmp = leaf_header[6:10].view(">u4")
            tmp[:] = (
                numpy.uint32(36 + leaf_name_length + leaf_title_length)
                | uproot.const.kByteCountMask
            )
            tmp = leaf_header[12:16].view(">u4")
            tmp[:] = (
                numpy.uint32(12 + leaf_name_length + leaf_title_length)
                | uproot.const.kByteCountMask
            )

            out.append(uproot._util.tobytes(leaf_header))
            if len(leaf_name) < 255:
                out.append(
                    struct.pack(">B%ds" % len(leaf_name), len(leaf_name), leaf_name)
                )
            else:
                out.append(
                    struct.pack(
                        ">BI%ds" % len(leaf_name), 255, len(leaf_name), leaf_name
                    )
                )
            if len(leaf_title) < 255:
                out.append(
                    struct.pack(">B%ds" % len(leaf_title), len(leaf_title), leaf_title)
                )
            else:
                out.append(
                    struct.pack(
                        ">BI%ds" % len(leaf_title), 255, len(leaf_title), leaf_title
                    )
                )

            fLen = 1
            for item in datum["shape"]:
                fLen *= item

            # generic TLeaf members
            out.append(
                uproot.models.TLeaf._tleaf2_format0.pack(
                    fLen,
                    fLenType,
                    0,  # fOffset
                    datum["kind"] == "counter",  # fIsRange
                    fIsUnsigned,
                )
            )

            if datum["counter"] is None:
                # null fLeafCount
                out.append(b"\x00\x00\x00\x00")
            else:
                # reference to fLeafCount
                out.append(
                    uproot.deserialization._read_object_any_format1.pack(
                        datum["counter"]["tleaf_reference_number"]
                    )
                )

            # specialized TLeaf* members (fMinimum, fMaximum)
            out.append(special_struct.pack(0, 0))
            datum["tleaf_special_struct"] = special_struct

            out[
                subany_tleaf_index
            ] = uproot.serialization._serialize_object_any_format1.pack(
                numpy.uint32(sum(len(x) for x in out[subany_tleaf_index + 1 :]) + 4)
                | uproot.const.kByteCountMask,
                uproot.const.kNewClassTag,
            )

            out[subtobjarray_of_leaves_index] = uproot.serialization.numbytes_version(
                sum(len(x) for x in out[subtobjarray_of_leaves_index + 1 :]),
                3,  # TObjArray
            )

            # empty TObjArray of fBaskets (embedded)
            out.append(
                b"@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )

            assert sum(1 if x is None else 0 for x in out) == 4
            datum["basket_metadata_start"] = (6 + 6 + 8 + 6) + sum(
                len(x) for x in out if x is not None
            )

            # speedbump and fBasketBytes
            out.append(b"\x01")
            out.append(uproot._util.tobytes(datum["fBasketBytes"]))

            # speedbump and fBasketEntry
            out.append(b"\x01")
            out.append(uproot._util.tobytes(datum["fBasketEntry"]))

            # speedbump and fBasketSeek
            out.append(b"\x01")
            out.append(uproot._util.tobytes(datum["fBasketSeek"]))

            # empty fFileName
            out.append(b"\x00")

            out[tbranch_index] = uproot.serialization.numbytes_version(
                sum(len(x) for x in out[tbranch_index + 1 :]), 13  # TBranch
            )

            out[
                any_tbranch_index
            ] = uproot.serialization._serialize_object_any_format1.pack(
                numpy.uint32(sum(len(x) for x in out[any_tbranch_index + 1 :]) + 4)
                | uproot.const.kByteCountMask,
                uproot.const.kNewClassTag,
            )

        out[tobjarray_of_branches_index] = uproot.serialization.numbytes_version(
            sum(len(x) for x in out[tobjarray_of_branches_index + 1 :]), 3  # TObjArray
        )

        # TObjArray of TLeaf references
        tleaf_reference_bytes = uproot._util.tobytes(
            numpy.array(tleaf_reference_numbers, ">u4")
        )
        out.append(
            struct.pack(
                ">I13sI4s",
                (21 + len(tleaf_reference_bytes)) | uproot.const.kByteCountMask,
                b"\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00",
                len(tleaf_reference_numbers),
                b"\x00\x00\x00\x00",
            )
        )

        out.append(tleaf_reference_bytes)

        # null fAliases (b"\x00\x00\x00\x00")
        # empty fIndexValues array (4-byte length is zero)
        # empty fIndex array (4-byte length is zero)
        # null fTreeIndex (b"\x00\x00\x00\x00")
        # null fFriends (b"\x00\x00\x00\x00")
        # null fUserInfo (b"\x00\x00\x00\x00")
        # null fBranchRef (b"\x00\x00\x00\x00")
        out.append(b"\x00" * 28)

        out[ttree_header_index] = uproot.serialization.numbytes_version(
            sum(len(x) for x in out[ttree_header_index + 1 :]), 20  # TTree
        )

        self._metadata_start = sum(len(x) for x in out[:metadata_out_index])

        raw_data = b"".join(out)
        self._key = self._directory.add_object(
            sink,
            "TTree",
            self._name,
            self._title,
            raw_data,
            len(raw_data),
            replaces=self._key,
            big=True,
        )

    def write_updates(self, sink):
        base = self._key.seek_location + self._key.num_bytes

        sink.write(
            base + self._metadata_start,
            uproot.models.TTree._ttree20_format1.pack(
                self._num_entries,
                self._metadata["fTotBytes"],
                self._metadata["fZipBytes"],
                self._metadata["fSavedBytes"],
                self._metadata["fFlushedBytes"],
                self._metadata["fWeight"],
                self._metadata["fTimerInterval"],
                self._metadata["fScanField"],
                self._metadata["fUpdate"],
                self._metadata["fDefaultEntryOffsetLen"],
                self._metadata["fNClusterRange"],
                self._metadata["fMaxEntries"],
                self._metadata["fMaxEntryLoop"],
                self._metadata["fMaxVirtualSize"],
                self._metadata["fAutoSave"],
                self._metadata["fAutoFlush"],
                self._metadata["fEstimate"],
            ),
        )

        for datum in self._branch_data:
            if datum["kind"] == "record":
                continue

            position = base + datum["metadata_start"]

            # Lie about the compression level so that ROOT checks and does the right thing.
            # https://github.com/root-project/root/blob/87a998d48803bc207288d90038e60ff148827664/tree/tree/src/TBasket.cxx#L560-L578
            # Without this, when small buffers are left uncompressed, ROOT complains about them not being compressed.
            # (I don't know where the "no, really, this is uncompressed" bit is.)
            fCompress = 0

            sink.write(
                position,
                uproot.models.TBranch._tbranch13_format1.pack(
                    fCompress,
                    datum["fBasketSize"],
                    datum["fEntryOffsetLen"],
                    self._num_baskets,  # fWriteBasket
                    self._num_entries,  # fEntryNumber
                ),
            )

            position += uproot.models.TBranch._tbranch13_format1.size + 11
            sink.write(
                position,
                uproot.models.TBranch._tbranch13_format2.pack(
                    datum["fOffset"],
                    self._basket_capacity,  # fMaxBaskets
                    datum["fSplitLevel"],
                    self._num_entries,  # fEntries
                    datum["fFirstEntry"],
                    datum["fTotBytes"],
                    datum["fZipBytes"],
                ),
            )

            start, stop = datum["arrays_write_start"], datum["arrays_write_stop"]

            fBasketBytes_part = uproot._util.tobytes(datum["fBasketBytes"][start:stop])
            fBasketEntry_part = uproot._util.tobytes(
                datum["fBasketEntry"][start : stop + 1]
            )
            fBasketSeek_part = uproot._util.tobytes(datum["fBasketSeek"][start:stop])

            position = base + datum["basket_metadata_start"] + 1
            position += datum["fBasketBytes"][:start].nbytes
            sink.write(position, fBasketBytes_part)
            position += len(fBasketBytes_part)
            position += datum["fBasketBytes"][stop:].nbytes

            position += 1
            position += datum["fBasketEntry"][:start].nbytes
            sink.write(position, fBasketEntry_part)
            position += len(fBasketEntry_part)
            position += datum["fBasketEntry"][stop + 1 :].nbytes

            position += 1
            position += datum["fBasketSeek"][:start].nbytes
            sink.write(position, fBasketSeek_part)
            position += len(fBasketSeek_part)
            position += datum["fBasketSeek"][stop:].nbytes

            datum["arrays_write_start"] = datum["arrays_write_stop"]

            if datum["dtype"] == ">U0":
                position = (
                    base
                    + datum["basket_metadata_start"]
                    - 25  # empty TObjArray of fBaskets (embedded)
                    - 8  # specialized TLeaf* members (fMinimum, fMaximum)
                    - 4  # null fLeafCount
                    - 14  # generic TLeaf members
                )
                sink.write(
                    position,
                    uproot.models.TLeaf._tleaf2_format0.pack(
                        self._metadata["fLen"],
                        datum["dtype"].itemsize,
                        0,
                        datum["kind"] == "counter",
                        _dtype_to_char[datum["dtype"]]
                        != _dtype_to_char[datum["dtype"]].upper(),
                    ),
                )

            if datum["kind"] == "counter":
                position = (
                    base
                    + datum["basket_metadata_start"]
                    - 25  # empty TObjArray of fBaskets (embedded)
                    - datum["tleaf_special_struct"].size
                )
                sink.write(
                    position,
                    datum["tleaf_special_struct"].pack(
                        0,
                        datum["tleaf_maximum_value"],
                    ),
                )

        sink.flush()

    def write_np_basket(self, sink, branch_name, compression, array):
        fClassName = uproot.serialization.string("TBasket")
        fName = uproot.serialization.string(branch_name)
        fTitle = uproot.serialization.string(self._name)

        fKeylen = (
            uproot.reading._key_format_big.size
            + len(fClassName)
            + len(fName)
            + len(fTitle)
            + uproot.models.TBasket._tbasket_format2.size
            + 1
        )

        itemsize = array.dtype.itemsize
        for item in array.shape[1:]:
            itemsize *= item

        uncompressed_data = uproot._util.tobytes(array)
        compressed_data = uproot.compression.compress(uncompressed_data, compression)

        fObjlen = len(uncompressed_data)
        fNbytes = fKeylen + len(compressed_data)

        parent_location = self._directory.key.location  # FIXME: is this correct?

        location = self._freesegments.allocate(fNbytes, dry_run=False)

        out = []
        out.append(
            uproot.reading._key_format_big.pack(
                fNbytes,
                1004,  # fVersion
                fObjlen,
                uproot._util.datetime_to_code(datetime.datetime.now()),  # fDatime
                fKeylen,
                0,  # fCycle
                location,  # fSeekKey
                parent_location,  # fSeekPdir
            )
        )
        out.append(fClassName)
        out.append(fName)
        out.append(fTitle)
        out.append(
            uproot.models.TBasket._tbasket_format2.pack(
                3,  # fVersion
                32000,  # fBufferSize
                itemsize,  # fNevBufSize
                len(array),  # fNevBuf
                fKeylen + len(uncompressed_data),  # fLast
            )
        )
        out.append(b"\x00")  # part of the Key (included in fKeylen, at least)

        out.append(compressed_data)

        sink.write(location, b"".join(out))
        self._freesegments.write(sink)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return fKeylen + fObjlen, fNbytes, location

    def write_jagged_basket(self, sink, branch_name, compression, array, offsets):
        fClassName = uproot.serialization.string("TBasket")
        fName = uproot.serialization.string(branch_name)
        fTitle = uproot.serialization.string(self._name)

        fKeylen = (
            uproot.reading._key_format_big.size
            + len(fClassName)
            + len(fName)
            + len(fTitle)
            + uproot.models.TBasket._tbasket_format2.size
            + 1
        )

        # offsets became a *copy* of the Awkward Array's offsets
        # when it was converted to big-endian (astype with copy=True)
        itemsize = array.dtype.itemsize
        for item in array.shape[1:]:
            itemsize *= item
        offsets *= itemsize
        offsets += fKeylen

        raw_array = uproot._util.tobytes(array)
        raw_offsets = uproot._util.tobytes(offsets)
        uncompressed_data = (
            raw_array + _tbasket_offsets_length.pack(len(offsets)) + raw_offsets
        )
        compressed_data = uproot.compression.compress(uncompressed_data, compression)

        fLast = offsets[-1]
        offsets[-1] = 0

        fObjlen = len(uncompressed_data)
        fNbytes = fKeylen + len(compressed_data)

        parent_location = self._directory.key.location  # FIXME: is this correct?

        location = self._freesegments.allocate(fNbytes, dry_run=False)

        out = []
        out.append(
            uproot.reading._key_format_big.pack(
                fNbytes,
                1004,  # fVersion
                fObjlen,
                uproot._util.datetime_to_code(datetime.datetime.now()),  # fDatime
                fKeylen,
                0,  # fCycle
                location,  # fSeekKey
                parent_location,  # fSeekPdir
            )
        )
        out.append(fClassName)
        out.append(fName)
        out.append(fTitle)
        out.append(
            uproot.models.TBasket._tbasket_format2.pack(
                3,  # fVersion
                32000,  # fBufferSize
                len(offsets) + 1,  # fNevBufSize
                len(offsets) - 1,  # fNevBuf
                fLast,
            )
        )
        out.append(b"\x00")  # part of the Key (included in fKeylen, at least)

        out.append(compressed_data)

        sink.write(location, b"".join(out))
        self._freesegments.write(sink)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return fKeylen + fObjlen, fNbytes, location

    def write_string_basket(self, sink, branch_name, compression, array, offsets):
        fClassName = uproot.serialization.string("TBasket")
        fName = uproot.serialization.string(branch_name)
        fTitle = uproot.serialization.string(self._name)
        fKeylen = (
            uproot.reading._key_format_big.size
            + len(fClassName)
            + len(fName)
            + len(fTitle)
            + uproot.models.TBasket._tbasket_format2.size
            + 1
        )

        itemsize = array.dtype.itemsize
        for item in array.shape[1:]:
            itemsize *= item
        try:
            uproot.extras.awkward()
        except ModuleNotFoundError as err:
            raise TypeError(
                f"'awkward' cannot be imported: {self._branch_type!r}"
            ) from err

        offsets *= itemsize
        offsets += fKeylen

        raw_array = uproot._util.tobytes(array)
        raw_offsets = uproot._util.tobytes(offsets)
        uncompressed_data = (
            raw_array
            + _tbasket_offsets_length.pack(len(offsets))
            + raw_offsets[:-4]
            + b"\x00\x00\x00\x00"
        )
        compressed_data = uproot.compression.compress(uncompressed_data, compression)

        # get size of biggest string
        self._metadata["fLen"] = (
            0
            if len(offsets) == 1
            else max([offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)])
        )

        fLast = offsets[-1]
        offsets[-1] = 0

        fObjlen = len(uncompressed_data)
        fNbytes = fKeylen + len(compressed_data)

        parent_location = self._directory.key.location  # FIXME: is this correct?

        location = self._freesegments.allocate(fNbytes, dry_run=False)

        out = []
        out.append(
            uproot.reading._key_format_big.pack(
                fNbytes,
                1004,  # fVersion
                fObjlen,
                uproot._util.datetime_to_code(datetime.datetime.now()),  # fDatime
                fKeylen,
                0,  # fCycle
                location,  # fSeekKey
                parent_location,  # fSeekPdir
            )
        )
        out.append(fClassName)
        out.append(fName)
        out.append(fTitle)
        out.append(
            uproot.models.TBasket._tbasket_format2.pack(
                3,  # fVersion
                32000,  # fBufferSize
                1000,  # fNevBufSize
                len(offsets) - 1,  # fNevBuf
                fLast,
            )
        )

        out.append(b"\x00")  # part of the Key (included in fKeylen, at least)

        out.append(compressed_data)

        sink.write(location, b"".join(out))
        self._freesegments.write(sink)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return fKeylen + fObjlen, fNbytes, location


_tbasket_offsets_length = struct.Struct(">I")


def dataframe_to_dict(df):
    """
    Converts a Pandas DataFrame into a dict of NumPy arrays for writing.
    """
    out = {"index": df.index.values}
    for column_name in df.columns:
        out[str(column_name)] = df[column_name].values
    return out


def recarray_to_dict(array):
    """
    Converts a NumPy structured array into a dict of non-structured arrays for writing.
    """
    out = {}
    for field_name in array.dtype.fields:
        field = array[field_name]
        if field.dtype.fields is not None:
            for subfield_name, subfield in recarray_to_dict(field):
                out[field_name + "." + subfield_name] = subfield
        else:
            out[field_name] = field
    return out
