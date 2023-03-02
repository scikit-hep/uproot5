from collections.abc import Callable, Iterable, Mapping

import numpy

import uproot
from uproot._util import no_filter
from uproot.behaviors.TBranch import HasBranches, TBranch, _regularize_step_size


def dask(
    files,
    *,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    step_size="100 MB",
    library="ak",
    ak_add_doc=False,
    custom_classes=None,
    allow_missing=False,
    open_files=True,
    form_mapping=None,
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
            that is used to represent arrays. If ``library='np'`` it returns a dict
            of dask arrays and if ``library='ak'`` it returns a single dask-awkward
            array. ``library='pd'`` has not been implemented yet and will raise a
            ``NotImplementedError``.
        ak_add_doc (bool): If True and ``library="ak"``, add the TBranch ``title``
            to the Awkward ``__doc__`` parameter of the array.
        custom_classes (None or dict): If a dict, override the classes from
            the :doc:`uproot.reading.ReadOnlyFile` or ``uproot.classes``.
        allow_missing (bool): If True, skip over any files that do not contain
            the specified ``TTree``.
        open_files (bool): If True (default), the function will open the files to read file
            metadata, i.e. only the main data read is delayed till the compute call on
            the dask collections. If False, the opening of the files and reading the
            metadata is also delayed till the compute call. In this case, branch-names
            are inferred by opening only the first file.
        form_mapping (Callable[awkward.forms.Form] -> awkward.forms.Form | None): If not none
            and library="ak" then apply this remapping function to the awkward form of the input
            data. The form keys of the desired form should be available data in the input form.
        options: See below.

    Returns dask equivalents of the backends supported by uproot. If ``library='np'``,
    the function returns a Python dict of dask arrays. If ``library='ak'``, the
    function returns a single dask-awkward array.

    For example:

    .. code-block:: python

        >>> uproot.dask(root_file)
        dask.awkward<from-uproot, npartitions=1>
        >>> uproot.dask(root_file,library='np')
        {'Type': dask.array<Type-from-uproot, shape=(2304,), dtype=object, chunksize=(2304,), chunktype=numpy.ndarray>, ...}


    This function (naturally) depends on Dask. To use it with ``library="np"``:

    .. code-block:: bash

        # with pip
        pip install "dask[complete]"
        # or with conda
        conda install dask

    For using ``library='ak'``

    .. code-block:: bash

        pip install dask-awkward   # not on conda-forge yet

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
    * :doc:`uproot._dask.dask` (this function): returns an unevaluated Dask
      array from ``TTrees``.
    """

    files = uproot._util.regularize_files(files)
    library = uproot.interpretation.library._regularize_library(library)

    if library.name == "pd":
        raise NotImplementedError()

    if library.name != "ak" and form_mapping is not None:
        raise NotImplementedError()

    real_options = options.copy()
    if "num_workers" not in real_options:
        real_options["num_workers"] = 1
    if "num_fallback_workers" not in real_options:
        real_options["num_fallback_workers"] = 1

    filter_branch = uproot._util.regularize_filter(filter_branch)

    interp_options = {"ak_add_doc": ak_add_doc}

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
                interp_options,
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
                interp_options,
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
                interp_options,
                form_mapping,
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
                interp_options,
                form_mapping,
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
        args=None,
        kwargs=None,
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


class _LazyInputsDict(Mapping):
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
        if isinstance(k, tuple) and isinstance(k[0], int):
            return k[0] >= 0 and k[0] < len(self)
        return False

    def keys(self):
        return ((i,) for i in range(len(self.inputs)))


def _dask_array_from_map(
    func,
    *iterables,
    chunks,
    dtype,
    args=None,
    label=None,
    token=None,
    **kwargs,
):
    dask = uproot.extras.dask()
    da = uproot.extras.dask_array()
    if not callable(func):
        raise ValueError("`func` argument must be `callable`")
    lengths = set()
    iters = list(iterables)
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
        io_func = _PackedArgCallable(
            func,
            args=args,
            kwargs=kwargs,
            packed=packed,
        )
    else:
        io_func = func

    io_arg_map = dask.blockwise.BlockwiseDepDict(
        mapping=_LazyInputsDict(inputs),  # type: ignore
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
    return da.core.Array(hlg, name, chunks, dtype=dtype)


class _UprootReadNumpy:
    def __init__(self, ttrees, key, interp_options) -> None:
        self.ttrees = ttrees
        self.key = key
        self.interp_options = interp_options

    def __call__(self, i_start_stop):
        i, start, stop = i_start_stop
        return self.ttrees[i][self.key].array(
            entry_start=start,
            entry_stop=stop,
            library="np",
            ak_add_doc=self.interp_options["ak_add_doc"],
        )


class _UprootOpenAndReadNumpy:
    def __init__(
        self, custom_classes, allow_missing, real_options, key, interp_options
    ):
        self.custom_classes = custom_classes
        self.allow_missing = allow_missing
        self.real_options = real_options
        self.key = key
        self.interp_options = interp_options

    def __call__(self, file_path_object_path):
        file_path, object_path = file_path_object_path
        ttree = uproot._util.regularize_object_path(
            file_path,
            object_path,
            self.custom_classes,
            self.allow_missing,
            self.real_options,
        )
        return ttree[self.key].array(
            library="np", ak_add_doc=self.interp_options["ak_add_doc"]
        )


def _get_dask_array(
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
    interp_options,
):
    ttrees = []
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
                    return branch is original and filter_branch(branch)  # noqa: B023

            else:
                is_self.append(False)
                real_filter_branch = filter_branch

            ttrees.append(obj)

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

    step_sum = 0
    for ttree in ttrees:
        entry_start = 0
        entry_stop = ttree.num_entries

        branchid_interpretation = {}
        for key in common_keys:
            branch = ttree[key]
            branchid_interpretation[branch.cache_key] = branch.interpretation
        ttree_step = _regularize_step_size(
            ttree, step_size, entry_start, entry_stop, branchid_interpretation
        )
        step_sum += int(ttree_step)

    entry_step = int(round(step_sum / len(ttrees)))
    assert entry_step >= 1

    for key in common_keys:
        dt = ttrees[0][key].interpretation.numpy_dtype
        if dt.subdtype is None:
            inner_shape = ()
        else:
            dt, inner_shape = dt.subdtype

        chunks = []
        chunk_args = []
        for i, ttree in enumerate(ttrees):
            entry_start = 0
            entry_stop = ttree.num_entries

            def foreach(start):
                stop = min(start + entry_step, entry_stop)  # noqa: B023
                length = stop - start
                chunks.append(length)  # noqa: B023
                chunk_args.append((i, start, stop))  # noqa: B023

            for start in range(entry_start, entry_stop, entry_step):
                foreach(start)

        if len(chunk_args) == 0:
            chunks.append(0)
            chunk_args.append((0, 0, 0))

        dask_dict[key] = _dask_array_from_map(
            _UprootReadNumpy(ttrees, key, interp_options),
            chunk_args,
            chunks=(tuple(chunks),),
            dtype=dt,
            label=f"{key}-from-uproot",
        )

    return dask_dict


def _get_dask_array_delay_open(
    files,
    filter_name,
    filter_typename,
    filter_branch,
    recursive,
    full_paths,
    custom_classes,
    allow_missing,
    real_options,
    interp_options,
):
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

    for key in common_keys:
        dt = obj[key].interpretation.numpy_dtype
        if dt.subdtype is None:
            inner_shape = ()
        else:
            dt, inner_shape = dt.subdtype

        dask_dict[key] = _dask_array_from_map(
            _UprootOpenAndReadNumpy(
                custom_classes, allow_missing, real_options, key, interp_options
            ),
            files,
            chunks=((numpy.nan,) * len(files),),
            dtype=dt,
            label=f"{key}-from-uproot",
        )
    return dask_dict


class _UprootRead:
    def __init__(
        self,
        ttrees,
        common_keys,
        common_base_keys,
        interp_options,
        form_mapping,
        rendered_form,
    ) -> None:
        self.ttrees = ttrees
        self.common_keys = common_keys
        self.common_base_keys = common_base_keys
        self.interp_options = interp_options
        self.form_mapping = form_mapping
        self.rendered_form = rendered_form

    def __call__(self, i_start_stop):
        i, start, stop = i_start_stop

        if self.form_mapping is not None:
            awkward = uproot.extras.awkward()

            actual_form = self.rendered_form.select_columns(self.common_keys)

            mapping, buffer_key = self.form_mapping.create_column_mapping_and_key(
                self.ttrees[i], start, stop, self.interp_options
            )

            return awkward.from_buffers(
                actual_form,
                stop - start,
                mapping,
                buffer_key=buffer_key,
                behavior=self.form_mapping.behavior,
            )

        return self.ttrees[i].arrays(
            self.common_keys,
            entry_start=start,
            entry_stop=stop,
            ak_add_doc=self.interp_options["ak_add_doc"],
        )

    def project_columns(self, common_keys):
        common_base_keys = self.common_base_keys
        if self.form_mapping is not None:
            awkward = uproot.extras.awkward()

            (
                new_meta_labelled,
                report,
            ) = awkward._nplikes.typetracer.typetracer_with_report(self.rendered_form)
            tt = awkward.Array(new_meta_labelled)

            if common_keys is not None:
                for key in common_keys:
                    tt[tuple(key.split("."))].layout._touch_data(recursive=True)

                common_base_keys = [
                    x
                    for x in self.form_mapping.extract_form_keys_base_columns(
                        set(report.data_touched)
                    )
                    if x in self.common_base_keys
                ]
        elif common_keys is not None:
            common_keys = [x for x in common_keys if x in self.common_keys]

        return _UprootRead(
            self.ttrees,
            common_keys,
            common_keys if self.form_mapping is None else common_base_keys,
            self.interp_options,
            self.form_mapping,
            self.rendered_form,
        )


class _UprootOpenAndRead:
    def __init__(
        self,
        custom_classes,
        allow_missing,
        real_options,
        common_keys,
        common_base_keys,
        interp_options,
        form_mapping,
        rendered_form,
    ) -> None:
        self.custom_classes = custom_classes
        self.allow_missing = allow_missing
        self.real_options = real_options
        self.common_keys = common_keys
        self.common_base_keys = common_base_keys
        self.interp_options = interp_options
        self.form_mapping = form_mapping
        self.rendered_form = rendered_form

    def __call__(self, file_path_object_path):
        file_path, object_path = file_path_object_path
        ttree = uproot._util.regularize_object_path(
            file_path,
            object_path,
            self.custom_classes,
            self.allow_missing,
            self.real_options,
        )

        if self.form_mapping is not None:
            awkward = uproot.extras.awkward()

            actual_form = self.rendered_form.select_columns(self.common_keys)

            mapping, buffer_key = self.form_mapping.create_column_mapping_and_key(
                ttree, 0, ttree.num_entries, self.interp_options
            )

            return awkward.from_buffers(
                actual_form,
                ttree.num_entries,
                mapping,
                buffer_key=buffer_key,
                behavior=self.form_mapping.behavior,
            )

        return ttree.arrays(
            self.common_keys, ak_add_doc=self.interp_options["ak_add_doc"]
        )

    def project_columns(self, common_keys):
        common_base_keys = self.common_base_keys
        if self.form_mapping is not None:
            awkward = uproot.extras.awkward()

            (
                new_meta_labelled,
                report,
            ) = awkward._nplikes.typetracer.typetracer_with_report(self.rendered_form)
            tt = awkward.Array(new_meta_labelled)

            if common_keys is not None:
                for key in common_keys:
                    tt[tuple(key.split("."))].layout._touch_data(recursive=True)

                common_base_keys = [
                    x
                    for x in self.form_mapping.extract_form_keys_base_columns(
                        set(report.data_touched)
                    )
                    if x in self.common_base_keys
                ]

        elif common_keys is not None:
            common_keys = [x for x in common_keys if x in self.common_keys]

        return _UprootOpenAndRead(
            self.custom_classes,
            self.allow_missing,
            self.real_options,
            common_keys,
            common_keys if self.form_mapping is None else common_base_keys,
            self.interp_options,
            self.form_mapping,
            self.rendered_form,
        )


def _get_meta_array(
    awkward,
    dask_awkward,
    ttree,
    common_keys,
    form_mapping,
    ak_add_doc,
):
    contents = []
    for key in common_keys:
        branch = ttree[key]
        content_form = branch.interpretation.awkward_form(ttree.file)
        if ak_add_doc:
            content_form = content_form.copy(parameters={"__doc__": branch.title})
        contents.append(content_form)

    parameters = {"__doc__": ttree.title} if ak_add_doc else None

    form = awkward.forms.RecordForm(contents, common_keys, parameters=parameters)

    if form_mapping is not None:
        form = form_mapping(form)

    empty_arr = form.length_zero_array(
        behavior=None if form_mapping is None else form_mapping.behavior
    )

    return dask_awkward.core.typetracer_array(empty_arr), form


def _get_dak_array(
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
    interp_options,
    form_mapping,
):
    dask_awkward = uproot.extras.dask_awkward()
    awkward = uproot.extras.awkward()

    ttrees = []
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
                    return branch is original and filter_branch(branch)  # noqa: B023

            else:
                is_self.append(False)
                real_filter_branch = filter_branch

            ttrees.append(obj)

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

    step_sum = 0
    for ttree in ttrees:
        entry_start = 0
        entry_stop = ttree.num_entries

        branchid_interpretation = {}
        for key in common_keys:
            branch = ttree[key]
            branchid_interpretation[branch.cache_key] = branch.interpretation
        ttree_step = _regularize_step_size(
            ttree, step_size, entry_start, entry_stop, branchid_interpretation
        )
        step_sum += int(ttree_step)

    entry_step = int(round(step_sum / len(ttrees)))

    partition_args = []
    for i, ttree in enumerate(ttrees):
        entry_start = 0
        entry_stop = ttree.num_entries

        def foreach(start):
            stop = min(start + entry_step, entry_stop)  # noqa: B023
            partition_args.append((i, start, stop))  # noqa: B023

        for start in range(entry_start, entry_stop, entry_step):
            foreach(start)

    meta, form = _get_meta_array(
        awkward,
        dask_awkward,
        ttrees[0],
        common_keys,
        form_mapping,
        interp_options.get("ak_add_doc"),
    )

    if len(partition_args) == 0:
        partition_args.append((0, 0, 0))
    return dask_awkward.from_map(
        _UprootRead(
            ttrees,
            common_keys,
            common_keys,
            interp_options,
            form_mapping=form_mapping,
            rendered_form=None if form_mapping is None else form,
        ),
        partition_args,
        label="from-uproot",
        behavior=None if form_mapping is None else form_mapping.behavior,
        meta=meta,
    )


def _get_dak_array_delay_open(
    files,
    filter_name,
    filter_typename,
    filter_branch,
    recursive,
    full_paths,
    custom_classes,
    allow_missing,
    real_options,
    interp_options,
    form_mapping,
):
    dask_awkward = uproot.extras.dask_awkward()
    awkward = uproot.extras.awkward()

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

    meta, form = _get_meta_array(
        awkward,
        dask_awkward,
        obj,
        common_keys,
        form_mapping,
        interp_options.get("ak_add_doc"),
    )

    return dask_awkward.from_map(
        _UprootOpenAndRead(
            custom_classes,
            allow_missing,
            real_options,
            common_keys,
            common_keys,
            interp_options,
            form_mapping=form_mapping,
            rendered_form=None if form_mapping is None else form,
        ),
        files,
        label="from-uproot",
        behavior=None if form_mapping is None else form_mapping.behavior,
        meta=meta,
    )
