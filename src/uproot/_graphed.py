# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""
This module defines :doc:`uproot._graphed.graphed`, which reads ``TTrees`` into a deferred
`graphed <https://github.com/graphed-org/graphed-project-mvp>`__ array — the analogue of
:doc:`uproot._dask.dask` for the ``graphed`` task-graph system (MVP).

``uproot.graphed(files, library="ak")`` returns a deferred ``graphed`` ``Array`` (recorded on the
``graphed-awkward`` backend); construction reads only metadata (the ``TTree`` form). ``graphed`` does
not impersonate a deferred-array ``.compute()`` — instead, a recorded analysis is executed the way one
actually runs a task graph: per partition through the ``graphed-exec-local`` executors
(``ProcessExecutor`` / ``ThreadExecutor``) and tree-reduced. :doc:`uproot._graphed.graphed_partitions`
builds the ``graphed_core.Task`` chunks; :doc:`uproot._graphed.necessary_columns` reports the column
projection (so each chunk reads only the ``TBranches`` the analysis touches — the dask-awkward
necessary-columns optimization, expressed through ``graphed``).

``graphed`` (and its backends) are imported lazily, so importing ``uproot`` does not require them.
"""
from __future__ import annotations

import uproot
import uproot._util
import uproot.interpretation.library
from uproot._dask import _get_ttree_form
from uproot._util import no_filter, unset
from uproot.behaviors.RNTuple import HasFields


class _GraphedTTreeSource:
    """A lazy ``graphed`` source: reads the selected ``TBranches`` from the resolved ``TTree``(s) via
    ``uproot`` when the graph is computed. ``columns`` (set by projection) restricts the read to the
    branches the analysis touches; ``None`` reads every selected branch."""

    def __init__(self, file_tree, common_keys, custom_classes, allow_missing, options):
        self._file_tree = file_tree  # list of (file_path, object_path)
        self._common_keys = list(common_keys)
        self._custom_classes = custom_classes
        self._allow_missing = allow_missing
        self._options = options
        self.columns = None  # None -> all common_keys; otherwise the projected subset
        self.last_columns_read = None  # set on each read, for inspection/tests

    def __call__(self):
        import awkward

        cols = list(self.columns) if self.columns is not None else list(self._common_keys)
        self.last_columns_read = list(cols)
        parts = []
        for file_path, object_path in self._file_tree:
            ttree = uproot._util.regularize_object_path(
                file_path, object_path, self._custom_classes, self._allow_missing, self._options
            )
            if ttree is not None:
                parts.append(ttree.arrays(cols, library="ak"))
        if not parts:
            return awkward.Array([])
        return parts[0] if len(parts) == 1 else awkward.concatenate(parts)


def graphed(
    files,
    *,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    library="ak",
    ak_add_doc=False,
    custom_classes=None,
    allow_missing=False,
    **options,
):
    """
    Args:
        files: The ``TTree``(s) to read, in any form accepted by :doc:`uproot._dask.dask` /
            :doc:`uproot.behaviors.TBranch.iterate` (a path with ``"file.root:tree"``, a list of
            such, a dict, etc.).
        filter_name, filter_typename, filter_branch, recursive, full_paths: ``TBranch`` selection,
            as in :doc:`uproot._dask.dask`.
        library (str): Only ``"ak"`` is supported (a single deferred ``graphed`` array). ``"np"`` and
            ``"pd"`` raise ``NotImplementedError``.
        ak_add_doc, custom_classes, allow_missing: As in :doc:`uproot._dask.dask`.
        options: Passed through to file opening.

    Returns a deferred ``graphed`` ``Array`` for the selected ``TTree``(s). Construction reads only
    metadata; computing the expression (e.g. via :doc:`uproot._graphed.compute`) triggers the read,
    fetching only the ``TBranches`` the recorded graph touches.

    This is the ``graphed`` analogue of :doc:`uproot._dask.dask`.
    """
    library = uproot.interpretation.library._regularize_library(library)
    if library.name != "ak":
        raise NotImplementedError(
            f"uproot.graphed currently supports only library='ak', not {library.name!r}"
        )

    import awkward
    from graphed import Session
    from graphed_awkward import AwkwardBackend, AwkwardForm

    real_options = options.copy()
    real_options.setdefault("num_workers", 1)
    filter_branch = uproot._util.regularize_filter(filter_branch)

    resolved = uproot._util.regularize_files(files, steps_allowed=False, **options)

    file_tree = []
    common_keys = None
    first_ttree = None
    for ftuple in resolved:
        file_path, object_path = ftuple[0], ftuple[1]
        obj = uproot._util.regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, real_options
        )
        if obj is None:
            continue
        # TTrees take filter_branch; RNTuples (HasFields) take filter_field (cf. uproot._dask)
        filter_kw = "filter_field" if isinstance(obj, HasFields) else "filter_branch"
        keys = obj.keys(
            recursive=recursive,
            filter_name=filter_name,
            filter_typename=filter_typename,
            full_paths=full_paths,
            ignore_duplicates=True,
            **{filter_kw: filter_branch},
        )
        if common_keys is None:
            common_keys = list(keys)
        else:
            keyset = set(keys)
            common_keys = [k for k in common_keys if k in keyset]
        if first_ttree is None:
            first_ttree = obj
        file_tree.append((file_path, object_path))

    if first_ttree is None:
        raise ValueError("uproot.graphed: no TTrees found in the given files")
    if not common_keys:
        raise ValueError("uproot.graphed: the TTrees have no TBranches in common")

    # RNTuples (HasFields) expose the awkward form directly; TTrees build it from branch interpretations
    if isinstance(first_ttree, HasFields):
        record_form = first_ttree.to_akform(filter_name=common_keys)[0]
    else:
        record_form = _get_ttree_form(awkward, first_ttree, common_keys, ak_add_doc)
    typetracer = awkward.Array(
        record_form.length_zero_array(highlevel=False).to_typetracer(forget_length=True)
    )

    session = Session(AwkwardBackend())
    source = _GraphedTTreeSource(file_tree, common_keys, custom_classes, allow_missing, real_options)
    name = getattr(first_ttree, "name", None) or "events"
    return session.source(name, form=AwkwardForm(typetracer), data=source)


def necessary_columns(array, *, on_fail="raise"):
    """The ``TBranches`` each source must read for ``array`` — ``graphed``'s necessary-buffer
    projection (metadata-only). Returns ``{source_name: frozenset(branch_names)}``."""
    from graphed_awkward.projection import project

    return dict(project(array, on_fail=on_fail).read_columns)


def graphed_partitions(
    files,
    *,
    step_size=unset,
    steps_per_file=unset,
    open_files=True,
    custom_classes=None,
    allow_missing=False,
    **options,
):
    """Partition a uproot dataset into ``graphed_core.Task`` chunks for the ``graphed-exec-local``
    executors. Every chunk is a ``Task(key, Partition(file_path, tree, entry_start, entry_stop))``.

    Mirrors :doc:`uproot._dask.dask`'s chunking knobs:

    - ``steps_per_file`` (default 1): split each ``TTree`` into that many contiguous chunks.
    - ``step_size`` (int entries, or a memory string like ``"100 MB"``): cap each chunk's size.
      Mutually exclusive with ``steps_per_file``, and **incompatible with** ``open_files=False``.
    - ``open_files`` (default ``True``): open every file to read its entry count and emit exact entry
      ranges. ``open_files=False`` is the **blind-steps** mode — files are *not* opened here; each
      chunk records ``(step_index, n_steps)`` instead (encoded as ``entry_start=step_index``,
      ``entry_stop=-n_steps``) and the real entry range is resolved against the file's own count when
      the chunk is read (:doc:`uproot._graphed.read_graphed_partition`).

    The chunks feed a ``graphed_core.Plan`` run by ``graphed_exec_local.ProcessExecutor`` /
    ``ThreadExecutor`` — the per-partition, tree-reduced execution that a deferred-array ``.compute()``
    hides."""
    from graphed_core import Partition, Task

    have_step_size = not isinstance(step_size, uproot._util._Unset)
    have_steps_per_file = not isinstance(steps_per_file, uproot._util._Unset)
    if have_step_size and not open_files:
        raise TypeError("step_size cannot be used with open_files=False; use steps_per_file")
    if have_step_size and have_steps_per_file:
        raise TypeError("step_size and steps_per_file are mutually exclusive; set only one")
    n_steps = int(steps_per_file) if have_steps_per_file else 1

    real_options = options.copy()
    real_options.setdefault("num_workers", 1)
    resolved = uproot._util.regularize_files(files, steps_allowed=False, **options)

    tasks = []
    key = 0
    for ftuple in resolved:
        file_path, object_path = ftuple[0], ftuple[1]
        if not open_files:
            # BLIND: do not open the file; emit (step_index, n_steps) chunks resolved at read time
            for step in range(n_steps):
                tasks.append(Task(key, Partition(file_path, object_path, step, -n_steps)))
                key += 1
            continue
        obj = uproot._util.regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, real_options
        )
        if obj is None:
            continue
        n_entries = obj.num_entries
        if have_step_size:
            per = step_size if isinstance(step_size, int) else obj.num_entries_for(step_size)
            per = max(1, int(per))
            ranges = [(s, min(s + per, n_entries)) for s in range(0, n_entries, per)]
        else:
            ranges = [((i * n_entries) // n_steps, ((i + 1) * n_entries) // n_steps) for i in range(n_steps)]
        for start, stop in ranges:
            if stop > start:
                tasks.append(Task(key, Partition(file_path, object_path, int(start), int(stop))))
                key += 1
    return tasks


def read_graphed_partition(partition, columns, *, tree=None, library="ak", **open_options):
    """Read a ``graphed_core.Partition``'s chunk of ``columns`` from its ROOT file.

    Resolves **blind** partitions (``entry_stop < 0`` encodes ``step_index`` / ``n_steps`` from
    ``graphed_partitions(..., open_files=False)``) against the file's *actual* entry count here, so a
    blindly-stepped dataset still reads every entry exactly once. Pass an already-open ``tree`` to reuse
    a per-worker ``open_once`` handle."""
    if tree is None:
        tree = uproot.open(partition.uri, **open_options)[partition.tree]
    start, stop = partition.entry_start, partition.entry_stop
    if stop < 0:  # blind: step `start` of `-stop` steps, resolved against this file's num_entries
        n_steps, step, n_entries = -stop, start, tree.num_entries
        start = (step * n_entries) // n_steps
        stop = ((step + 1) * n_entries) // n_steps
    return tree.arrays(list(columns), entry_start=start, entry_stop=stop, library=library)
