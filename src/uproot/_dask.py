import numpy

import uproot
from uproot._util import no_filter
from uproot.behaviors.TBranch import (
    HasBranches,
    TBranch,
    _regularize_entries_start_stop,
)
from collections.abc import Callable, Iterable, Mapping

def dask(
    files,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    step_size="100 MB",
    library="ak",
    custom_classes=None,
    allow_missing=False,
    open_files=True,
    **options,
):
    """
    Args:
        files: See below.
        filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by name.
        filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by type.
        filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
            filter to select ``TBranches`` using the full
            :doc:`uproot.behaviors.TBranch.TBranch` object. If the function
            returns False or None, the ``TBranch`` is excluded; if the function
            returns True, it is included with its standard
            :ref:`uproot.behaviors.TBranch.TBranch.interpretation`; if an
            :doc:`uproot.interpretation.Interpretation`, this interpretation
            overrules the standard one.
        recursive (bool): If True, include all subbranches of branches as
            separate fields; otherwise, only search one level deep.
        full_paths (bool): If True, include the full path to each subbranch
            with slashes (``/``); otherwise, use the descendant's name as
            the field name.
        step_size (int or str): If an integer, the maximum number of entries to
            include in each chunk; if a string, the maximum memory_size to include
            in each chunk. The string must be a number followed by a memory unit,
            such as "100 MB".
        library (str or :doc:`uproot.interpretation.library.Library`): The library
            that is used to represent arrays.
        custom_classes (None or dict): If a dict, override the classes from
            the :doc:`uproot.reading.ReadOnlyFile` or ``uproot.classes``.
        allow_missing (bool): If True, skip over any files that do not contain
            the specified ``TTree``.
        options: See below.

    Returns dask equivalents of the backends supported by uproot.

    Allowed types for the ``files`` parameter:

    * str/bytes: relative or absolute filesystem path or URL, without any colons
      other than Windows drive letter or URL schema.
      Examples: ``"rel/file.root"``, ``"C:\\abs\\file.root"``, ``"http://where/what.root"``
    * str/bytes: same with an object-within-ROOT path, separated by a colon.
      Example: ``"rel/file.root:tdirectory/ttree"``
    * pathlib.Path: always interpreted as a filesystem path or URL only (no
      object-within-ROOT path), regardless of whether there are any colons.
      Examples: ``Path("rel:/file.root")``, ``Path("/abs/path:stuff.root")``
    * glob syntax in str/bytes and pathlib.Path.
      Examples: ``Path("rel/*.root")``, ``"/abs/*.root:tdirectory/ttree"``
    * dict: keys are filesystem paths, values are objects-within-ROOT paths.
      Example: ``{{"/data_v1/*.root": "ttree_v1", "/data_v2/*.root": "ttree_v2"}}``
    * already-open TTree objects.
    * iterables of the above.

    Options (type; default):

    * file_handler (:doc:`uproot.source.chunk.Source` class; :doc:`uproot.source.file.MemmapSource`)
    * xrootd_handler (:doc:`uproot.source.chunk.Source` class; :doc:`uproot.source.xrootd.XRootDSource`)
    * http_handler (:doc:`uproot.source.chunk.Source` class; :doc:`uproot.source.http.HTTPSource`)
    * object_handler (:doc:`uproot.source.chunk.Source` class; :doc:`uproot.source.object.ObjectSource`)
    * timeout (float for HTTP, int for XRootD; 30)
    * max_num_elements (None or int; None)
    * num_workers (int; 1)
    * num_fallback_workers (int; 10)
    * begin_chunk_size (memory_size; 512)
    * minimal_ttree_metadata (bool; True)

    Other file entry points:

    * :doc:`uproot.reading.open`: opens one file to read any of its objects.
    * :doc:`uproot.behaviors.TBranch.iterate`: iterates through chunks of
      contiguous entries in ``TTrees``.
    * :doc:`uproot.behaviors.TBranch.concatenate`: returns a single
      concatenated array from ``TTrees``.
    * :doc:`uproot.behaviors.TBranch.lazy` (this function): returns a lazily
      read array from ``TTrees``.
    """

    files = uproot._util.regularize_files(files)
    library = uproot.interpretation.library._regularize_library(library)

    if library.name == "pd":
        raise NotImplementedError()

    real_options = dict(options)
    if "num_workers" not in real_options:
        real_options["num_workers"] = 1
    if "num_fallback_workers" not in real_options:
        real_options["num_fallback_workers"] = 1

    filter_branch = uproot._util.regularize_filter(filter_branch)

    if library.name == "np":
        if open_files:
            return _get_dask_array(
                files,
                filter_name,
                filter_typename,
                filter_branch,
                recursive,
                full_paths,
                step_size,
                custom_classes,
                allow_missing,
                real_options,
            )
        else:
            return _get_dask_array_delay_open(
                files,
                filter_name,
                filter_typename,
                filter_branch,
                recursive,
                full_paths,
                custom_classes,
                allow_missing,
                real_options,
            )
    elif library.name == "ak":
        if open_files:
            return _get_dak_array(
                files,
                filter_name,
                filter_typename,
                filter_branch,
                recursive,
                full_paths,
                step_size,
                custom_classes,
                allow_missing,
                real_options,
            )
        else:
            return _get_dak_array_delay_open(
                files,
                filter_name,
                filter_typename,
                filter_branch,
                recursive,
                full_paths,
                custom_classes,
                allow_missing,
                real_options,
            )
    else:
        raise NotImplementedError()

class _PackedArgCallable:
    """Wrap a callable such that packed arguments can be unrolled.
    Inspired by dask.dataframe.io.io._PackedArgCallable.
    """

    def __init__(
        self,
        func: Callable,
        args= None,
        kwargs= None,
        packed: bool = False,
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.packed = packed

    def __call__(self, packed_arg):
        if not self.packed:
            packed_arg = (packed_arg,)
        return self.func(
            *packed_arg,
            *(self.args or []),
            **(self.kwargs or {}),
        )

class LazyInputsDict(Mapping):
    """Dictionary with lazy key value pairs
    Parameters
    ----------
    inputs : list[Any]
        The list of dicionary values.
    """

    def __init__(self, inputs, **kwargs) -> None:
        self.inputs = inputs
        self.kwargs = kwargs

    def __len__(self):
        return len(self.inputs)

    def __iter__(self):
        return (self[k] for k in self.keys())

    def __getitem__(self, i):
        return self.inputs[i[0]]

    def __contains__(self, k):
        if isinstance(k, tuple):
            if isinstance(k[0], int):
                return k[0] >= 0 and k[0] < len(self)
        return False

    def keys(self):
        return ((i,) for i in range(len(self.inputs)))

def _dask_array_from_map(
    func,
    *iterables,
    chunks,
    dtype,
    args= None,
    label= None,
    token= None,
    **kwargs,
):
    dask, da = uproot.extras.dask()
    if not callable(func):
        raise ValueError("`func` argument must be `callable`")
    lengths = set()
    iters= list(iterables)
    for i, iterable in enumerate(iters):
        if not isinstance(iterable, Iterable):
            raise ValueError(
                f"All elements of `iterables` must be Iterable, got {type(iterable)}"
            )
        try:
            lengths.add(len(iterable))  # type: ignore
        except (AttributeError, TypeError):
            iters[i] = list(iterable)
            lengths.add(len(iters[i]))  # type: ignore
    if len(lengths) == 0:
        raise ValueError("`from_map` requires at least one Iterable input")
    elif len(lengths) > 1:
        raise ValueError("All `iterables` must have the same length")
    if lengths == {0}:
        raise ValueError("All `iterables` must have a non-zero length")

    # Check for `produces_tasks` and `creation_info`
    produces_tasks = kwargs.pop("produces_tasks", False)
    # creation_info = kwargs.pop("creation_info", None)

    if produces_tasks or len(iters) == 1:
        if len(iters) > 1:
            # Tasks are not detected correctly when they are "packed"
            # within an outer list/tuple
            raise ValueError(
                "Multiple iterables not supported when produces_tasks=True"
            )
        inputs = list(iters[0])
        packed = False
    else:
        # Structure inputs such that the tuple of arguments pair each 0th,
        # 1st, 2nd, ... elements together; for example:
        # from_map(f, [1, 2, 3], [4, 5, 6]) --> [f(1, 4), f(2, 5), f(3, 6)]
        inputs = list(zip(*iters))
        packed = True

    # Define collection name
    label = label or dask.utils.funcname(func)
    token = token or dask.base.tokenize(func, iters, **kwargs)
    name = f"{label}-{token}"

    # Define io_func
    if packed or args or kwargs:
        io_func= _PackedArgCallable(
            func,
            args=args,
            kwargs=kwargs,
            packed=packed,
        )
    else:
        io_func = func


    io_arg_map = dask.blockwise.BlockwiseDepDict(
        mapping=LazyInputsDict(inputs),  # type: ignore
        produces_tasks=produces_tasks,
    )

    dsk = dask.blockwise.Blockwise(
        output=name,
        output_indices="i",
        dsk={name: (io_func, dask.blockwise.blockwise_token(0))},
        indices=[(io_arg_map, "i")],
        numblocks={},
        annotations=None,
    )

    hlg = dask.highlevelgraph.HighLevelGraph.from_collections(name, dsk)
    return dask.array.core.Array(hlg,name,chunks,dtype=dtype)

class _UprootReadNumpy:
    def __init__(self, hasbranches, key) -> None:
        self.hasbranches = hasbranches
        self.key = key

    def __call__(self, i_start_stop):
        i, start, stop = i_start_stop
        return self.hasbranches[i][self.key].array(
            entry_start=start, entry_stop=stop,library='np'
        )

def _get_dask_array(
    files,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    step_size="100 MB",
    custom_classes=None,
    allow_missing=False,
    real_options=None,
):
    dask, da = uproot.extras.dask()
    hasbranches = []
    common_keys = None
    is_self = []

    count = 0
    for file_path, object_path in files:
        obj = uproot._util.regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, real_options
        )

        if obj is not None:
            count += 1

            if isinstance(obj, TBranch) and len(obj.keys(recursive=True)) == 0:
                original = obj
                obj = obj.parent
                is_self.append(True)

                def real_filter_branch(branch):
                    return branch is original and filter_branch(branch)

            else:
                is_self.append(False)
                real_filter_branch = filter_branch

            hasbranches.append(obj)

            new_keys = obj.keys(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=real_filter_branch,
                full_paths=full_paths,
            )

            if common_keys is None:
                common_keys = new_keys
            else:
                new_keys = set(new_keys)
                common_keys = [key for key in common_keys if key in new_keys]

    if count == 0:
        raise ValueError(
            "allow_missing=True and no TTrees found in\n\n    {}".format(
                "\n    ".join(
                    "{"
                    + "{}: {}".format(
                        repr(f.file_path if isinstance(f, HasBranches) else f),
                        repr(f.object_path if isinstance(f, HasBranches) else o),
                    )
                    + "}"
                    for f, o in files
                )
            )
        )

    if len(common_keys) == 0 or not (all(is_self) or not any(is_self)):
        raise ValueError(
            "TTrees in\n\n    {}\n\nhave no TBranches in common".format(
                "\n    ".join(
                    "{"
                    + "{}: {}".format(
                        repr(f.file_path if isinstance(f, HasBranches) else f),
                        repr(f.object_path if isinstance(f, HasBranches) else o),
                    )
                    + "}"
                    for f, o in files
                )
            )
        )

    dask_dict = {}

    for key in common_keys:
        dt = hasbranches[0][key].interpretation.numpy_dtype
        if dt.subdtype is None:
            inner_shape = ()
        else:
            dt, inner_shape = dt.subdtype

        chunks = []
        chunk_args = []
        for i,ttree in enumerate(hasbranches):
            entry_start, entry_stop = _regularize_entries_start_stop(
                ttree.tree.num_entries, None, None
            )
            entry_step = 0
            if uproot._util.isint(step_size):
                entry_step = step_size
            else:
                entry_step = ttree.num_entries_for(step_size, expressions=f"{key}")
            
            def foreach(start):
                stop = min(start + entry_step, entry_stop)
                length = stop - start
                chunks.append(length)
                chunk_args.append((i,start,stop))

            for start in range(entry_start, entry_stop, entry_step):
                foreach(start)

        dask_dict[key] = _dask_array_from_map(
            _UprootReadNumpy(hasbranches,key),
            chunk_args,
            chunks=(tuple(chunks),),
            dtype=dt,
        )
    
    return dask_dict

def _get_dask_array_delay_open(
    files,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    custom_classes=None,
    allow_missing=False,
    real_options=None,
):
    dask, da = uproot.extras.dask()
    ffile_path, fobject_path = files[0]
    obj = uproot._util.regularize_object_path(
        ffile_path, fobject_path, custom_classes, allow_missing, real_options
    )
    common_keys = obj.keys(
        recursive=recursive,
        filter_name=filter_name,
        filter_typename=filter_typename,
        filter_branch=filter_branch,
        full_paths=full_paths,
    )

    dask_dict = {}
    delayed_open_fn = dask.delayed(uproot._util.regularize_object_path)

    @dask.delayed
    def delayed_get_array(ttree, key):
        return ttree[key].array(library="np")

    for key in common_keys:
        dask_arrays = []
        for file_path, object_path in files:
            delayed_tree = delayed_open_fn(
                file_path, object_path, custom_classes, allow_missing, real_options
            )
            delayed_array = delayed_get_array(delayed_tree, key)
            dt = obj[key].interpretation.numpy_dtype
            if dt.subdtype is not None:
                dt, inner_shape = dt.subdtype

            dask_arrays.append(
                da.from_delayed(delayed_array, shape=(numpy.nan,), dtype=dt)
            )
        dask_dict[key] = da.concatenate(dask_arrays, allow_unknown_chunksizes=True)
    return dask_dict


class _UprootRead:
    def __init__(self, hasbranches, branches) -> None:
        self.hasbranches = hasbranches
        self.branches = branches

    def __call__(self, i_start_stop):
        i, start, stop = i_start_stop
        return self.hasbranches[i].arrays(
            self.branches, entry_start=start, entry_stop=stop
        )


class _UprootOpenAndRead:
    def __init__(
        self, custom_classes, allow_missing, real_options, common_keys
    ) -> None:
        self.custom_classes = custom_classes
        self.allow_missing = allow_missing
        self.real_options = real_options
        self.common_keys = common_keys

    def __call__(self, file_path_object_path):
        file_path, object_path = file_path_object_path
        ttree = uproot._util.regularize_object_path(
            file_path,
            object_path,
            self.custom_classes,
            self.allow_missing,
            self.real_options,
        )
        return ttree.arrays(self.common_keys)


def _get_dak_array(
    files,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    step_size="100 MB",
    custom_classes=None,
    allow_missing=False,
    real_options=None,
):
    dask_awkward = uproot.extras.dask_awkward()

    hasbranches = []
    common_keys = None
    is_self = []

    count = 0
    for file_path, object_path in files:
        obj = uproot._util.regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, real_options
        )

        if obj is not None:
            count += 1

            if isinstance(obj, TBranch) and len(obj.keys(recursive=True)) == 0:
                original = obj
                obj = obj.parent
                is_self.append(True)

                def real_filter_branch(branch):
                    return branch is original and filter_branch(branch)

            else:
                is_self.append(False)
                real_filter_branch = filter_branch

            hasbranches.append(obj)

            new_keys = obj.keys(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=real_filter_branch,
                full_paths=full_paths,
            )

            if common_keys is None:
                common_keys = new_keys
            else:
                new_keys = set(new_keys)
                common_keys = [key for key in common_keys if key in new_keys]

    if count == 0:
        raise ValueError(
            "allow_missing=True and no TTrees found in\n\n    {}".format(
                "\n    ".join(
                    "{"
                    + "{}: {}".format(
                        repr(f.file_path if isinstance(f, HasBranches) else f),
                        repr(f.object_path if isinstance(f, HasBranches) else o),
                    )
                    + "}"
                    for f, o in files
                )
            )
        )

    if len(common_keys) == 0 or not (all(is_self) or not any(is_self)):
        raise ValueError(
            "TTrees in\n\n    {}\n\nhave no TBranches in common".format(
                "\n    ".join(
                    "{"
                    + "{}: {}".format(
                        repr(f.file_path if isinstance(f, HasBranches) else f),
                        repr(f.object_path if isinstance(f, HasBranches) else o),
                    )
                    + "}"
                    for f, o in files
                )
            )
        )

    partition_args = []
    for i, ttree in enumerate(hasbranches):
        entry_start, entry_stop = _regularize_entries_start_stop(
            ttree.num_entries, None, None
        )
        entry_step = 0
        if uproot._util.isint(step_size):
            entry_step = step_size
        else:
            entry_step = ttree.num_entries_for(step_size, expressions=common_keys)

        def foreach(start):
            stop = min(start + entry_step, entry_stop)
            partition_args.append((i, start, stop))

        for start in range(entry_start, entry_stop, entry_step):
            foreach(start)

    first_basket_start, first_basket_stop = hasbranches[0][
        common_keys[0]
    ].basket_entry_start_stop(0)
    first_basket = hasbranches[0].arrays(
        common_keys, entry_start=first_basket_start, entry_stop=first_basket_stop
    )
    meta = dask_awkward.core.typetracer_array(first_basket)

    return dask_awkward.from_map(
        _UprootRead(hasbranches, common_keys),
        partition_args,
        label="from-uproot",
        meta=meta,
    )


def _get_dak_array_delay_open(
    files,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    custom_classes=None,
    allow_missing=False,
    real_options=None,
):
    dask_awkward = uproot.extras.dask_awkward()

    ffile_path, fobject_path = files[0]
    obj = uproot._util.regularize_object_path(
        ffile_path, fobject_path, custom_classes, allow_missing, real_options
    )
    common_keys = obj.keys(
        recursive=recursive,
        filter_name=filter_name,
        filter_typename=filter_typename,
        filter_branch=filter_branch,
        full_paths=full_paths,
    )

    first_basket_start, first_basket_stop = obj[common_keys[0]].basket_entry_start_stop(
        0
    )
    first_basket = obj.arrays(
        common_keys, entry_start=first_basket_start, entry_stop=first_basket_stop
    )
    meta = dask_awkward.core.typetracer_array(first_basket)

    return dask_awkward.from_map(
        _UprootOpenAndRead(custom_classes, allow_missing, real_options, common_keys),
        files,
        label="from-uproot",
        meta=meta,
    )
