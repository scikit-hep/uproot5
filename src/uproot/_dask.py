from __future__ import annotations

import functools
import math
import socket
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Executor

from uproot.source.chunk import SourcePerformanceCounters

try:
    from typing import TYPE_CHECKING, Final

    from typing_extensions import Any, Protocol, TypeVar
except ImportError:
    from typing import TYPE_CHECKING, Any, Final, Protocol, TypeVar

import numpy

import uproot
from uproot._util import no_filter, unset
from uproot.behaviors.TBranch import HasBranches, TBranch, _regularize_step_size

if TYPE_CHECKING:
    from awkward._nplikes.typetracer import TypeTracerReport
    from awkward.forms import Form
    from awkward.highlevel import Array as AwkArray


def dask(
    files,
    *,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    step_size=unset,
    steps_per_file=unset,
    library="ak",
    ak_add_doc=False,
    custom_classes=None,
    allow_missing=False,
    open_files=True,
    form_mapping=None,
    allow_read_errors_with_report=False,
    known_base_form=None,
    decompression_executor=None,
    interpretation_executor=None,
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
            include in each chunk/partition; if a string, the maximum memory_size to include
            in each chunk/partition. The string must be a number followed by a memory unit,
            such as "100 MB". Mutually incompatible with steps_per_file: only set
            step_size or steps_per_file, not both. Cannot be used with
            ``open_files=False``.
        steps_per_file (int, default 1):
            Subdivide files into the specified number of chunks/partitions. Mutually incompatible
            with step_size: only set step_size or steps_per_file, not both.
            If both ``step_size`` and ``steps_per_file`` are unset,
            ``steps_per_file``'s default value of 1 (whole file per chunk/partition) is used,
            regardless of ``open_files``.
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
        allow_read_errors_with_report (bool or tuple of exceptions): If True, catch OSError
            exceptions and return an empty array for these nodes in the task graph. If a tuple,
            catch any of those exceptions and return empty arrays for those nodes. In either of
            those cases, The return of this function becomes a two element tuple, where the
            first return is the dask-awkward collection of interest and the second return is a
            report dask-awkward collection.
        known_base_form (awkward.forms.Form | None): If not none use this form instead of opening
            one file to determine the dataset's form. Only available with open_files=False.
        decompression_executor (None or Executor with a ``submit`` method): The
            executor that is used to decompress ``TBaskets``; if None, a
            :doc:`uproot.source.futures.TrivialExecutor` is created.
            Executors attached to a file are ``shutdown`` when the file is closed.
        interpretation_executor (None or Executor with a ``submit`` method): The
            executor that is used to interpret uncompressed ``TBasket`` data as
            arrays; if None, a :doc:`uproot.source.futures.TrivialExecutor`
            is created.
            Executors attached to a file are ``shutdown`` when the file is closed.
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
      Example: ``{"/data_v1/*.root": "ttree_v1", "/data_v2/*.root": "ttree_v2"}``
    * dict: keys are filesystem paths, values are dicts containing objects-within-ROOT and
      steps (chunks/partitions) as a list of starts and stops or steps as a list of offsets
      Example:

          {{"/data_v1/tree1.root": {"object_path": "ttree_v1", "steps": [[0, 10000], [15000, 20000], ...]},
            "/data_v1/tree2.root": {"object_path": "ttree_v1", "steps": [0, 10000, 20000, ...]}}}

      (This ``files`` pattern is incompatible with ``step_size`` and ``steps_per_file``.)
    * already-open TTree objects.
    * iterables of the above.

    Options (type; default):

    * handler (:doc:`uproot.source.chunk.Source` class; None)
    * timeout (float for HTTP, int for XRootD; 30)
    * max_num_elements (None or int; None)
    * num_workers (int; 1)
    * use_threads (bool; False on the emscripten platform (i.e. in a web browser), else True)
    * num_fallback_workers (int; 10)
    * begin_chunk_size (memory_size; 403, the smallest a ROOT file can be)
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

    files = uproot._util.regularize_files(files, steps_allowed=True, **options)

    is_3arg = [len(x) == 3 for x in files]
    if any(is_3arg):
        if not all(is_3arg):
            raise TypeError(
                "partition sizes for some but not all 'files' have been assigned"
            )
        if step_size is not unset:
            raise TypeError(
                "partition sizes for 'files' is incompatible with 'step_size'"
            )
        if steps_per_file is not unset:
            raise TypeError(
                "partition sizes for 'files' is incompatible with 'steps_per_file'"
            )

    library = uproot.interpretation.library._regularize_library(library)

    if step_size is not unset and steps_per_file is not unset:
        raise TypeError(
            f"only 'step_size' or 'steps_per_file' should be set, not both; got step_size={step_size!r} and steps_per_file={steps_per_file!r}"
        )
    elif step_size is not unset:
        if not open_files:
            # the not open_files case FAILS if only step_size is supplied
            raise TypeError(
                f"'step_size' should not be set when 'open_files' is False; got {step_size!r}"
            )
        else:
            # the open_files case uses step_size (only)
            pass
    elif steps_per_file is not unset:
        # the not open_files case uses steps_per_file (only)
        # the open_files case converts steps_per_file into step_size
        pass
    else:
        steps_per_file = 1

    if known_base_form is not None and open_files:
        raise TypeError("known_base_form must be None if open_files is True")

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
                steps_per_file,
                decompression_executor,
                interpretation_executor,
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
                steps_per_file,
                decompression_executor,
                interpretation_executor,
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
                steps_per_file,
                allow_read_errors_with_report,
                decompression_executor,
                interpretation_executor,
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
                steps_per_file,
                allow_read_errors_with_report,
                known_base_form,
                decompression_executor,
                interpretation_executor,
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
    def __init__(
        self,
        ttrees,
        key,
        interp_options,
        decompression_executor=None,
        interpretation_executor=None,
    ) -> None:
        self.ttrees = ttrees
        self.key = key
        self.interp_options = interp_options
        self.decompression_executor = decompression_executor
        self.interpretation_executor = interpretation_executor

    def __call__(self, i_start_stop):
        i, start, stop = i_start_stop
        return self.ttrees[i][self.key].array(
            entry_start=start,
            entry_stop=stop,
            library="np",
            ak_add_doc=self.interp_options["ak_add_doc"],
            decompression_executor=self.decompression_executor,
            interpretation_executor=self.interpretation_executor,
        )


class _UprootOpenAndReadNumpy:
    def __init__(
        self,
        custom_classes,
        allow_missing,
        real_options,
        key,
        interp_options,
        decompression_executor=None,
        interpretation_executor=None,
    ):
        self.custom_classes = custom_classes
        self.allow_missing = allow_missing
        self.real_options = real_options
        self.key = key
        self.interp_options = interp_options
        self.decompression_executor = decompression_executor
        self.interpretation_executor = interpretation_executor

    def __call__(self, file_path_object_path_istep_nsteps_ischunk):
        (
            file_path,
            object_path,
            istep_or_start,
            nsteps_or_stop,
            ischunk,
        ) = file_path_object_path_istep_nsteps_ischunk
        ttree = uproot._util.regularize_object_path(
            file_path,
            object_path,
            self.custom_classes,
            self.allow_missing,
            self.real_options,
        )
        num_entries = ttree.num_entries
        start, stop = istep_or_start, nsteps_or_stop
        if not ischunk:
            events_per_steps = math.ceil(num_entries / nsteps_or_stop)
            start, stop = (istep_or_start * events_per_steps), min(
                (istep_or_start + 1) * events_per_steps, num_entries
            )
        elif (not 0 <= start < num_entries) or (not 0 <= stop <= num_entries):
            raise ValueError(
                f"""explicit entry start ({start}) or stop ({stop}) from uproot.dask 'files' argument is out of bounds for file

    {ttree.file.file_path}

TTree in path

    {ttree.object_path}

which has {num_entries} entries"""
            )

        return ttree[self.key].array(
            library="np",
            entry_start=start,
            entry_stop=stop,
            ak_add_doc=self.interp_options["ak_add_doc"],
            decompression_executor=self.decompression_executor,
            interpretation_executor=self.interpretation_executor,
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
    steps_per_file,
    decompression_executor,
    interpretation_executor,
):
    ttrees = []
    explicit_chunks = []
    common_keys = None
    is_self = []

    count = 0
    for file_object_maybechunks in files:
        file_path, object_path = file_object_maybechunks[0:2]

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
            if len(file_object_maybechunks) == 3:
                explicit_chunks.append(file_object_maybechunks[2])
            else:
                explicit_chunks = None  # they all have it or none of them have it

            new_keys = obj.keys(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=real_filter_branch,
                full_paths=full_paths,
                ignore_duplicates=True,
            )

            if common_keys is None:
                common_keys = new_keys
            else:
                new_keys = set(new_keys)
                common_keys = [key for key in common_keys if key in new_keys]

    # this is the earliest time we can deal with an unset step_size
    if step_size is unset:
        assert steps_per_file is not unset  # either assigned or assumed to be 1
        total_files = len(ttrees)
        total_entries = sum(ttree.num_entries for ttree in ttrees)
        step_size = max(
            1, int(math.ceil(total_entries / (total_files * steps_per_file)))
        )

    if count == 0:
        raise ValueError(
            "allow_missing=True and no TTrees found in\n\n    {}".format(
                "\n    ".join(
                    f"{{{f.file_path if isinstance(f, HasBranches) else f!r}: {f.object_path if isinstance(f, HasBranches) else o!r}}}"
                    for f, o in files
                )
            )
        )

    if len(common_keys) == 0 or not (all(is_self) or not any(is_self)):
        raise ValueError(
            "TTrees in\n\n    {}\n\nhave no TBranches in common".format(
                "\n    ".join(
                    f"{{{f.file_path if isinstance(f, HasBranches) else f!r}: {f.object_path if isinstance(f, HasBranches) else o!r}}}"
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

            if explicit_chunks is None:
                for start in range(entry_start, entry_stop, entry_step):
                    stop = min(start + entry_step, entry_stop)
                    length = stop - start
                    if length > 0:
                        chunks.append(length)
                        chunk_args.append((i, start, stop))
            else:
                for start, stop in explicit_chunks[i]:
                    if (not 0 <= start < entry_stop) or (not 0 <= stop <= entry_stop):
                        raise ValueError(
                            f"""explicit entry start ({start}) or stop ({stop}) from uproot.dask 'files' argument is out of bounds for file

    {ttree.file.file_path}

TTree in path

    {ttree.object_path}

which has {entry_stop} entries"""
                        )
                    length = stop - start
                    if length > 0:
                        chunks.append(length)
                        chunk_args.append((i, start, stop))

        if len(chunk_args) == 0:
            chunks.append(0)
            chunk_args.append((0, 0, 0))

        dask_dict[key] = _dask_array_from_map(
            _UprootReadNumpy(
                ttrees,
                key,
                interp_options,
                decompression_executor,
                interpretation_executor,
            ),
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
    steps_per_file,
    decompression_executor,
    interpretation_executor,
):
    ffile_path, fobject_path = files[0][0:2]
    obj = uproot._util.regularize_object_path(
        ffile_path, fobject_path, custom_classes, allow_missing, real_options
    )
    common_keys = obj.keys(
        recursive=recursive,
        filter_name=filter_name,
        filter_typename=filter_typename,
        filter_branch=filter_branch,
        full_paths=full_paths,
        ignore_duplicates=True,
    )

    dask_dict = {}

    for key in common_keys:
        dt = obj[key].interpretation.numpy_dtype
        if dt.subdtype is None:
            inner_shape = ()
        else:
            dt, inner_shape = dt.subdtype

        partitions = []
        partition_args = []
        for ifile_iobject_maybeichunks in files:
            ifile_path, iobject_path = ifile_iobject_maybeichunks[0:2]

            chunks = None
            if len(ifile_iobject_maybeichunks) == 3:
                chunks = ifile_iobject_maybeichunks[2]

            if chunks is not None:
                partitions.extend([stop - start for start, stop in chunks])
                for start, stop in chunks:
                    partition_args.append(
                        (
                            ifile_path,
                            iobject_path,
                            start,
                            stop,
                            True,
                        )
                    )
            else:
                partitions.extend([numpy.nan] * steps_per_file)
                for istep in range(steps_per_file):
                    partition_args.append(
                        (
                            ifile_path,
                            iobject_path,
                            istep,
                            steps_per_file,
                            False,
                        )
                    )

        dask_dict[key] = _dask_array_from_map(
            _UprootOpenAndReadNumpy(
                custom_classes,
                allow_missing,
                real_options,
                key,
                interp_options,
                decompression_executor,
                interpretation_executor,
            ),
            partition_args,
            chunks=(tuple(partitions),),
            dtype=dt,
            label=f"{key}-from-uproot",
        )
    return dask_dict


class ImplementsFormMappingInfo(Protocol):
    @property
    def behavior(self) -> dict | None: ...

    buffer_key: str | Callable

    def parse_buffer_key(self, buffer_key: str) -> tuple[str, str]: ...

    def keys_for_buffer_keys(self, buffer_keys: frozenset[str]) -> frozenset[str]: ...

    def load_buffers(
        self,
        tree: HasBranches,
        keys: frozenset[str],
        start: int,
        stop: int,
        decompression_executor: Executor,
        interpretation_executor: Executor,
        options: Any,
    ) -> Mapping[str, AwkArray]: ...


class ImplementsFormMapping(Protocol):
    def __call__(self, form: Form) -> tuple[Form, ImplementsFormMappingInfo]: ...


class TrivialFormMappingInfo(ImplementsFormMappingInfo):
    def __init__(self, form):
        awkward = uproot.extras.awkward()
        assert isinstance(form, awkward.forms.RecordForm)

        self._form = form
        self._form_key_to_key = self.build_form_key_to_key(form)

    @property
    def behavior(self) -> None:
        return None

    @staticmethod
    def build_form_key_to_key(form: Form) -> dict[str, str | None]:
        form_key_to_path: dict[str, str | None] = {}

        def impl(form, column_path):
            # Store columnar path
            form_key_to_path[form.form_key] = column_path[0] if column_path else None

            if form.is_union:
                for _i, entry in enumerate(form.contents):
                    impl(entry, column_path)
            elif form.is_indexed:
                impl(form.content, column_path)
            elif form.is_list:
                impl(form.content, column_path)
            elif form.is_option:
                impl(form.content, column_path)
            elif form.is_record:
                for field in form.fields:
                    impl(form.content(field), (*column_path, field))
            elif form.is_unknown or form.is_numpy:
                pass
            else:
                raise AssertionError(form)

        impl(form, ())

        return form_key_to_path

    buffer_key: Final[str] = "{form_key}-{attribute}"

    def parse_buffer_key(self, buffer_key: str) -> tuple[str, str]:
        form_key, attribute = buffer_key.rsplit("-", maxsplit=1)
        return form_key, attribute

    def keys_for_buffer_keys(self, buffer_keys: frozenset[str]) -> frozenset[str]:
        keys: set[str] = set()
        for buffer_key in buffer_keys:
            # Identify form key
            form_key, attribute = buffer_key.rsplit("-", maxsplit=1)
            # Identify key from form_key
            keys.add(self._form_key_to_key[form_key])
        return frozenset(keys)

    def load_buffers(
        self,
        tree: HasBranches,
        keys: frozenset[str],
        start: int,
        stop: int,
        decompression_executor,
        interpretation_executor,
        options: Any,
    ) -> Mapping[str, AwkArray]:
        # First, let's read the arrays as a tuple (to associate with each key)
        arrays = tree.arrays(
            keys,
            entry_start=start,
            entry_stop=stop,
            ak_add_doc=options["ak_add_doc"],
            decompression_executor=decompression_executor,
            interpretation_executor=interpretation_executor,
            how=tuple,
        )

        awkward = uproot.extras.awkward()

        # The subform generated by awkward.to_buffers() has different form keys
        # from those used to perform buffer projection. However, the subform
        # structure should be identical to the projection optimisation
        # subform, as they're derived from `branch.interpretation.awkward_form`
        # Therefore, we can correlate the subform keys using `expected_from_buffers`
        container = {}
        for key, array in zip(keys, arrays):
            # First, convert the sub-array into buffers
            ttree_subform, length, ttree_container = awkward.to_buffers(array)

            # Load the associated projection subform
            projection_subform = self._form.content(key)

            # Correlate each TTree form key with the projection form key
            for (src, src_dtype), (dst, dst_dtype) in zip(
                ttree_subform.expected_from_buffers().items(),
                projection_subform.expected_from_buffers(self.buffer_key).items(),
            ):
                assert src_dtype == dst_dtype  # Sanity check!
                container[dst] = ttree_container[src]

        return container


class TrivialFormMapping(ImplementsFormMapping):
    def __call__(self, form: Form) -> tuple[Form, TrivialFormMappingInfo]:
        dask_awkward = uproot.extras.dask_awkward()
        new_form = dask_awkward.lib.utils.form_with_unique_keys(form, "<root>")
        return new_form, TrivialFormMappingInfo(new_form)


T = TypeVar("T")


class UprootReadMixin:
    base_form: Form
    expected_form: Form
    form_mapping_info: ImplementsFormMappingInfo
    common_keys: frozenset[str]
    interp_options: dict[str, Any]
    allow_read_errors_with_report: bool | tuple[type[BaseException], ...]

    @property
    def allowed_exceptions(self):
        if isinstance(self.allow_read_errors_with_report, tuple):
            return self.allow_read_errors_with_report
        return (OSError,)

    @property
    def return_report(self) -> bool:
        return bool(self.allow_read_errors_with_report)

    def read_tree(
        self, tree: HasBranches, start: int, stop: int
    ) -> tuple[AwkArray, SourcePerformanceCounters]:
        assert start <= stop

        from awkward._nplikes.numpy import Numpy

        awkward = uproot.extras.awkward()
        nplike = Numpy.instance()

        # The remap implementation should correctly populate the generated
        # buffer mapping in __call__, such that the high-level form can be
        # used in `from_buffers`
        mapping = self.form_mapping_info.load_buffers(
            tree,
            self.common_keys,
            start,
            stop,
            self.decompression_executor,
            self.interpretation_executor,
            self.interp_options,
        )

        # Populate container with placeholders if keys aren't required
        # Otherwise, read from disk
        container = {}
        for buffer_key, dtype in self.expected_form.expected_from_buffers(
            buffer_key=self.form_mapping_info.buffer_key
        ).items():
            # Which key(s) does this buffer require. This code permits the caller
            # to require multiple keys to compute a single buffer.
            keys_for_buffer = self.form_mapping_info.keys_for_buffer_keys(
                frozenset({buffer_key})
            )
            # If reading this buffer loads a permitted key, read from the tree
            # We might not have _all_ keys if e.g. buffer A requires one
            # but not two of the keys required for buffer B
            if all(k in self.common_keys for k in keys_for_buffer):
                container[buffer_key] = mapping[buffer_key]
            # Otherwise, introduce a placeholder
            else:
                container[buffer_key] = awkward.typetracer.PlaceholderArray(
                    nplike=nplike,
                    shape=(awkward.typetracer.unknown_length,),
                    dtype=dtype,
                )

        out = awkward.from_buffers(
            self.expected_form,
            stop - start,
            container,
            behavior=self.form_mapping_info.behavior,
            buffer_key=self.form_mapping_info.buffer_key,
        )
        assert tree.source  # we must be reading something here
        return out, tree.source.performance_counters

    def mock(self) -> AwkArray:
        awkward = uproot.extras.awkward()
        return awkward.typetracer.typetracer_from_form(
            self.expected_form,
            highlevel=True,
            behavior=self.form_mapping_info.behavior,
        )

    def mock_empty(self, backend) -> AwkArray:
        awkward = uproot.extras.awkward()
        return awkward.to_backend(
            self.expected_form.length_zero_array(highlevel=False),
            backend=backend,
            highlevel=True,
            behavior=self.form_mapping_info.behavior,
        )

    def prepare_for_projection(self) -> tuple[AwkArray, TypeTracerReport, dict]:
        awkward = uproot.extras.awkward()
        dask_awkward = uproot.extras.dask_awkward()

        # A form mapping will (may) remap the base form into a new form
        # The remapped form can be queried for structural information

        # Build typetracer and associated report object
        meta, report = awkward.typetracer.typetracer_with_report(
            self.expected_form,
            highlevel=True,
            behavior=self.form_mapping_info.behavior,
            buffer_key=self.form_mapping_info.buffer_key,
        )

        return (
            meta,
            report,
            {
                "trace": dask_awkward.lib.utils.trace_form_structure(
                    self.expected_form,
                    buffer_key=self.form_mapping_info.buffer_key,
                ),
                "form_info": self.form_mapping_info,
            },
        )

    def project(self: T, *, report: TypeTracerReport, state: dict) -> T:
        keys = self.necessary_columns(report=report, state=state)
        return self.project_keys(keys)

    def necessary_columns(
        self, *, report: TypeTracerReport, state: dict
    ) -> frozenset[str]:
        ## Read from stash
        # Form hierarchy information
        form_key_to_parent_form_key: dict = state["trace"][
            "form_key_to_parent_form_key"
        ]
        # Buffer hierarchy information
        form_key_to_buffer_keys: dict = state["trace"]["form_key_to_buffer_keys"]
        # Restructured form information
        form_info = state["form_info"]

        # Require the data of metadata buffers above shape-only requests
        dask_awkward = uproot.extras.dask_awkward()
        data_buffers = {
            *report.data_touched,
            *dask_awkward.lib.utils.buffer_keys_required_to_compute_shapes(
                form_info.parse_buffer_key,
                report.shape_touched,
                form_key_to_parent_form_key,
                form_key_to_buffer_keys,
            ),
        }

        # Determine which TTree keys need to be read
        return form_info.keys_for_buffer_keys(data_buffers) & frozenset(
            self.common_keys
        )

    def project_keys(self: T, keys: frozenset[str]) -> T:
        raise NotImplementedError


def _report_failure(exception, call_time, *args, **kwargs):
    awkward = uproot.extras.awkward()
    return awkward.Array(
        [
            {
                "call_time": call_time,
                "duration": None,
                "performance_counters": None,
                "args": [repr(a) for a in args],
                "kwargs": [[k, repr(v)] for k, v in kwargs.items()],
                "exception": type(exception).__name__,
                "message": str(exception),
                "fqdn": socket.getfqdn(),
                "hostname": socket.gethostname(),
            }
        ]
    )


def _report_success(duration, *args, **kwargs):
    awkward = uproot.extras.awkward()
    counters = kwargs.pop("counters")
    return awkward.Array(
        [
            {
                "call_time": None,
                "duration": duration,
                "performance_counters": counters.asdict(),
                "args": [repr(a) for a in args],
                "kwargs": [[k, repr(v)] for k, v in kwargs.items()],
                "exception": None,
                "message": None,
                "fqdn": None,
                "hostname": None,
            }
        ]
    )


def with_duration(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = f(*args, **kwargs)
        stop = time.monotonic()
        return result, (stop - start)

    return wrapper


class _UprootRead(UprootReadMixin):
    def __init__(
        self,
        ttrees,
        common_keys: frozenset[str],
        interp_options: dict[str, Any],
        base_form: Form,
        expected_form: Form,
        form_mapping_info: ImplementsFormMappingInfo,
        allow_read_errors_with_report: bool | tuple[type[BaseException], ...],
        decompression_executor,
        interpretation_executor,
    ) -> None:
        self.ttrees = ttrees
        self.common_keys = frozenset(common_keys)
        self.interp_options = interp_options
        self.base_form = base_form
        self.expected_form = expected_form
        self.form_mapping_info = form_mapping_info
        self.allow_read_errors_with_report = allow_read_errors_with_report
        self.decompression_executor = decompression_executor
        self.interpretation_executor = interpretation_executor

    def project_keys(self: T, keys: frozenset[str]) -> T:
        return _UprootRead(
            self.ttrees,
            keys,
            self.interp_options,
            self.base_form,
            self.expected_form,
            self.form_mapping_info,
            self.allow_read_errors_with_report,
            self.decompression_executor,
            self.interpretation_executor,
        )

    def __call__(self, i_start_stop):
        i, start, stop = i_start_stop
        if self.return_report:
            call_time = time.time_ns()
            try:
                (result, counters), duration = with_duration(self._call_impl)(
                    i, start, stop
                )
                return (
                    result,
                    _report_success(
                        duration,
                        self.ttrees[i],
                        start,
                        stop,
                        counters=counters,
                    ),
                )
            except self.allowed_exceptions as err:
                return (
                    self.mock_empty(backend="cpu"),
                    _report_failure(
                        err,
                        call_time,
                        self.ttrees[i],
                        start,
                        stop,
                    ),
                )

        result, _ = self._call_impl(i, start, stop)
        return result

    def _call_impl(self, i, start, stop):
        return self.read_tree(
            self.ttrees[i],
            start,
            stop,
        )


class _UprootOpenAndRead(UprootReadMixin):
    def __init__(
        self,
        custom_classes,
        allow_missing,
        real_options,
        common_keys,
        interp_options,
        base_form: Form,
        expected_form: Form,
        form_mapping_info: ImplementsFormMappingInfo,
        allow_read_errors_with_report: bool | tuple[type[BaseException], ...],
        decompression_executor,
        interpretation_executor,
    ) -> None:
        self.custom_classes = custom_classes
        self.allow_missing = allow_missing
        self.real_options = real_options
        self.common_keys = frozenset(common_keys)
        self.interp_options = interp_options
        self.base_form = base_form
        self.expected_form = expected_form
        self.form_mapping_info = form_mapping_info
        self.allow_read_errors_with_report = allow_read_errors_with_report
        self.decompression_executor = decompression_executor
        self.interpretation_executor = interpretation_executor

    def _call_impl(
        self, file_path, object_path, i_step_or_start, n_steps_or_stop, is_chunk
    ):
        ttree = uproot._util.regularize_object_path(
            file_path,
            object_path,
            self.custom_classes,
            self.allow_missing,
            self.real_options,
        )
        num_entries = ttree.num_entries
        if is_chunk:
            start, stop = i_step_or_start, n_steps_or_stop
            if (not 0 <= start < num_entries) or (not 0 <= stop <= num_entries):
                raise ValueError(
                    f"""explicit entry start ({start}) or stop ({stop}) from uproot.dask 'files' argument is out of bounds for file

    {ttree.file.file_path}

TTree in path

    {ttree.object_path}

which has {num_entries} entries"""
                )
        else:
            events_per_step = math.ceil(num_entries / n_steps_or_stop)
            start, stop = min((i_step_or_start * events_per_step), num_entries), min(
                (i_step_or_start + 1) * events_per_step, num_entries
            )

        assert start <= stop

        return self.read_tree(
            ttree,
            start,
            stop,
        )

    def __call__(self, blockwise_args):
        (
            file_path,
            object_path,
            i_step_or_start,
            n_steps_or_stop,
            is_chunk,
        ) = blockwise_args

        if self.return_report:
            call_time = time.time_ns()
            try:
                (result, counters), duration = with_duration(self._call_impl)(
                    file_path, object_path, i_step_or_start, n_steps_or_stop, is_chunk
                )
                return (
                    result,
                    _report_success(
                        duration,
                        file_path,
                        object_path,
                        i_step_or_start,
                        n_steps_or_stop,
                        is_chunk,
                        counters=counters,
                    ),
                )
            except self.allowed_exceptions as err:
                return (
                    self.mock_empty(backend="cpu"),
                    _report_failure(
                        err,
                        call_time,
                        file_path,
                        object_path,
                        i_step_or_start,
                        n_steps_or_stop,
                        is_chunk,
                    ),
                )

        result, _ = self._call_impl(
            file_path, object_path, i_step_or_start, n_steps_or_stop, is_chunk
        )
        return result

    def project_keys(self: T, keys: frozenset[str]) -> T:
        return _UprootOpenAndRead(
            self.custom_classes,
            self.allow_missing,
            self.real_options,
            keys,
            self.interp_options,
            self.base_form,
            self.expected_form,
            self.form_mapping_info,
            self.allow_read_errors_with_report,
            self.decompression_executor,
            self.interpretation_executor,
        )


def _get_ttree_form(
    awkward,
    ttree,
    common_keys,
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

    return awkward.forms.RecordForm(contents, common_keys, parameters=parameters)


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
    steps_per_file,
    allow_read_errors_with_report,
    decompression_executor,
    interpretation_executor,
):
    dask_awkward = uproot.extras.dask_awkward()
    awkward = uproot.extras.awkward()

    ttrees = []
    explicit_chunks = []
    common_keys = None
    is_self = []

    count = 0
    for file_object_maybechunks in files:
        file_path, object_path = file_object_maybechunks[0:2]

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
            if len(file_object_maybechunks) == 3:
                explicit_chunks.append(file_object_maybechunks[2])
            else:
                explicit_chunks = None  # they all have it or none of them have it

            new_keys = obj.keys(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=real_filter_branch,
                full_paths=full_paths,
                ignore_duplicates=True,
            )

            if common_keys is None:
                common_keys = new_keys
            else:
                new_keys = set(new_keys)
                common_keys = [key for key in common_keys if key in new_keys]

    # this is the earliest time we can deal with an unset step_size
    if step_size is unset:
        assert steps_per_file is not unset  # either assigned or assumed to be 1
        total_files = len(ttrees)
        total_entries = sum(ttree.num_entries for ttree in ttrees)
        step_size = max(
            1, int(math.ceil(total_entries / (total_files * steps_per_file)))
        )

    if count == 0:
        raise ValueError(
            "allow_missing=True and no TTrees found in\n\n    {}".format(
                "\n    ".join(
                    f"{{{f.file_path if isinstance(f, HasBranches) else f!r}: {f.object_path if isinstance(f, HasBranches) else o!r}}}"
                    for f, o in files
                )
            )
        )

    if len(common_keys) == 0 or not (all(is_self) or not any(is_self)):
        raise ValueError(
            "TTrees in\n\n    {}\n\nhave no TBranches in common".format(
                "\n    ".join(
                    f"{{{f.file_path if isinstance(f, HasBranches) else f!r}: {f.object_path if isinstance(f, HasBranches) else o!r}}}"
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

    divisions = [0]
    partition_args = []
    for i, ttree in enumerate(ttrees):
        entry_start = 0
        entry_stop = ttree.num_entries

        if explicit_chunks is None:
            for start in range(entry_start, entry_stop, entry_step):
                stop = min(start + entry_step, entry_stop)
                length = stop - start
                if length > 0:
                    divisions.append(divisions[-1] + length)
                    partition_args.append((i, start, stop))
        else:
            for start, stop in explicit_chunks[i]:
                if (not 0 <= start < entry_stop) or (not 0 <= stop <= entry_stop):
                    raise ValueError(
                        f"""explicit entry start ({start}) or stop ({stop}) from uproot.dask 'files' argument is out of bounds for file

    {ttree.file.file_path}

TTree in path

    {ttree.object_path}

which has {entry_stop} entries"""
                    )
                length = stop - start
                if length > 0:
                    divisions.append(divisions[-1] + length)
                    partition_args.append((i, start, stop))

    base_form = _get_ttree_form(
        awkward, ttrees[0], common_keys, interp_options.get("ak_add_doc")
    )

    if len(partition_args) == 0:
        divisions.append(0)
        partition_args.append((0, 0, 0))

    if form_mapping is None:
        expected_form = dask_awkward.lib.utils.form_with_unique_keys(
            base_form, "<root>"
        )
        form_mapping_info = TrivialFormMappingInfo(expected_form)
    else:
        expected_form, form_mapping_info = form_mapping(base_form)

    fn = _UprootRead(
        ttrees,
        common_keys,
        interp_options,
        base_form=base_form,
        expected_form=expected_form,
        form_mapping_info=form_mapping_info,
        allow_read_errors_with_report=allow_read_errors_with_report,
        decompression_executor=decompression_executor,
        interpretation_executor=interpretation_executor,
    )

    return dask_awkward.from_map(
        fn,
        partition_args,
        divisions=tuple(divisions),
        label="from-uproot",
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
    steps_per_file,
    allow_read_errors_with_report,
    known_base_form,
    decompression_executor,
    interpretation_executor,
):
    dask_awkward = uproot.extras.dask_awkward()
    awkward = uproot.extras.awkward()

    ffile_path, fobject_path = files[0][0:2]

    if known_base_form is not None:
        common_keys = list(dict.fromkeys(known_base_form.fields))
        base_form = known_base_form
    else:
        obj = uproot._util.regularize_object_path(
            ffile_path, fobject_path, custom_classes, allow_missing, real_options
        )
        common_keys = obj.keys(
            recursive=recursive,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            full_paths=full_paths,
            ignore_duplicates=True,
        )
        base_form = _get_ttree_form(
            awkward, obj, common_keys, interp_options.get("ak_add_doc")
        )

    divisions = [0]
    partition_args = []
    for ifile_iobject_maybeichunks in files:
        chunks = None
        ifile_path, iobject_path = ifile_iobject_maybeichunks[0:2]
        if len(ifile_iobject_maybeichunks) == 3:
            chunks = ifile_iobject_maybeichunks[2]

        if chunks is not None:
            for start, stop in chunks:
                divisions.append(divisions[-1] + (stop - start))
                partition_args.append(
                    (
                        ifile_path,
                        iobject_path,
                        start,
                        stop,
                        True,
                    )
                )
        else:
            divisions = None  # they all have it or none of them have it
            for istep in range(steps_per_file):
                partition_args.append(
                    (
                        ifile_path,
                        iobject_path,
                        istep,
                        steps_per_file,
                        False,
                    )
                )

    if form_mapping is None:
        expected_form = dask_awkward.lib.utils.form_with_unique_keys(
            base_form, "<root>"
        )
        form_mapping_info = TrivialFormMappingInfo(expected_form)
    else:
        expected_form, form_mapping_info = form_mapping(base_form)

    fn = _UprootOpenAndRead(
        custom_classes,
        allow_missing,
        real_options,
        common_keys,
        interp_options,
        base_form=base_form,
        expected_form=expected_form,
        form_mapping_info=form_mapping_info,
        allow_read_errors_with_report=allow_read_errors_with_report,
        decompression_executor=decompression_executor,
        interpretation_executor=interpretation_executor,
    )

    return dask_awkward.from_map(
        fn,
        partition_args,
        divisions=None if divisions is None else tuple(divisions),
        label="from-uproot",
    )
