# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""
This module defines :doc:`uproot._graphed.graphed`, which reads ``TTrees`` into a deferred
`graphed <https://github.com/graphed-org/graphed>`__ array — the analogue of
:doc:`uproot._dask.dask` for the ``graphed`` task-graph system.

**Experimental.** ``uproot.graphed``, :doc:`uproot._graphed.graphed_head` and
:doc:`uproot.writing._graphed_write.graphed_write` track a pre-1.0 ``graphed`` and may change
without a deprecation cycle.

``uproot.graphed(files)`` returns a deferred ``graphed`` ``Array`` recorded on the
``graphed.awkward`` backend; construction reads only metadata (the ``TTree`` form). A recorded
analysis runs the way a task graph runs: per partition through a ``graphed-executors`` executor
and tree-reduced (``graphed.aggregate_plan``, ``graphed.awkward.to_parquet``,
``uproot.graphed_write``). The array's source is a ``graphed`` partitioned source; every read goes
through a form mapping exactly as in :doc:`uproot._dask.dask` — ``TrivialFormMapping`` unless the
caller gives one — so a chunk holds every field of the mapped form, with the ``TBranches`` the
recorded graph does not touch standing in as placeholders. The source DECLARES that read list in
``TBranch`` names (``graphed``'s buffer projection, walked back through the mapping);
:doc:`uproot._graphed.necessary_columns` reports it without running anything.

``graphed`` is imported lazily, so importing ``uproot`` does not require it.
"""

from __future__ import annotations

import uproot
import uproot._util
import uproot.interpretation.library
import uproot.reading
from uproot._dask import (
    _base_form_of,
    _mapping_of,
    _normalize_grouped_keys,
    _read_tree,
    _resolve_trees_and_keys,
)
from uproot._util import no_filter, unset
from uproot.behaviors.TBranch import HasBranches

#: ``uproot.dask`` parameters ``uproot.graphed`` has no counterpart for, and where that concern
#: lives instead; ``**options`` would otherwise swallow them silently.
_DASK_ONLY = {
    "step_size": "partition with uproot._graphed.graphed_partitions(step_size=...) and pass the "
    "result to graphed.aggregate_plan(partitions=...)",
    "steps_per_file": "pass steps_per_file= where the plan is built: graphed.aggregate_plan, "
    "graphed.awkward.to_parquet or uproot.graphed_write",
    "open_files": "uproot.graphed opens files only to read the form; pass known_base_form= to "
    "open none",
    "allow_read_errors_with_report": "a failed partition is dead-lettered and retried by "
    "graphed.checkpoint.run_resumable",
}


class _GraphedTTreeSource:
    """A lazy ``graphed`` source over resolved ``(file_path, object_path)`` pairs, read through a
    form mapping. ``graphed.write.PartitionedSource``: ``partitions`` are blind (planning opens
    no file) unless the caller gave explicit ``steps``; ``read_partition`` opens each file once per
    worker; ``projected_columns`` declares the ``TBranches`` a set of outputs needs."""

    def __init__(
        self,
        file_tree,
        common_keys,
        *,
        name,
        expected_form,
        form_mapping_info,
        interp_options,
        custom_classes,
        allow_missing,
        options,
        decompression_executor,
        interpretation_executor,
        explicit_chunks=None,
    ):
        self._file_tree = list(file_tree)  # [(file_path, object_path or None)]
        self._common_keys = list(common_keys)
        self._name = name  # the graphed source name, which the projection answers under
        self._expected_form = expected_form
        self._form_mapping_info = form_mapping_info
        self._interp_options = interp_options
        self._custom_classes = custom_classes
        self._allow_missing = allow_missing
        self._options = options
        self._decompression_executor = decompression_executor
        self._interpretation_executor = interpretation_executor
        self._explicit_chunks = explicit_chunks  # per file: [(start, stop), ...]

    # ---- opening -------------------------------------------------------------------------
    def _open_directory(self, file_path):
        return uproot.reading.ReadOnlyFile(
            file_path,
            object_cache=None,
            array_cache=None,
            custom_classes=self._custom_classes,
            **self._options,
        ).root_directory

    def _tree_in(self, directory, object_path):
        return uproot._util.object_in_directory(
            directory, object_path or None, self._allow_missing
        )

    def open_tree(self, partition, resources):
        """This partition's ``TTree``, from a file opened once per worker with the options
        ``uproot.graphed`` was given; ``None`` when ``allow_missing`` and the file lacks it.
        """
        return self._tree_in(
            resources.open_once(partition.uri, self._open_directory), partition.tree
        )

    def _trees(self):
        """Every tree, opened here and now (the whole-dataset loader and ``graphed_head``)."""
        for file_path, object_path in self._file_tree:
            tree = self._tree_in(self._open_directory(file_path), object_path)
            if tree is not None:
                yield tree

    # ---- reading -------------------------------------------------------------------------
    def read_range(self, tree, keys, start, stop):
        """Entries ``[start, stop)`` of ``keys`` through the mapping (unread buffers are
        placeholders), as :doc:`uproot._dask.dask` reads a partition."""
        return _read_tree(
            tree,
            keys,
            start,
            stop,
            expected_form=self._expected_form,
            form_mapping_info=self._form_mapping_info,
            interp_options=self._interp_options,
            decompression_executor=self._decompression_executor,
            interpretation_executor=self._interpretation_executor,
        )

    def _empty(self):
        import awkward

        return awkward.Array(
            self._expected_form.length_zero_array(highlevel=False),
            behavior=self._form_mapping_info.behavior,
        )

    def __call__(self):
        import awkward

        parts = [
            self.read_range(tree, self._common_keys, 0, tree.num_entries)
            for tree in self._trees()
        ]
        if not parts:
            return self._empty()
        return parts[0] if len(parts) == 1 else awkward.concatenate(parts)

    # ---- graphed.write.PartitionedSource ----------------------------------------------
    def partitions(self, steps_per_file=1):
        """One partition per (file x step) — BLIND, so planning opens no file — or, when the
        ``files`` carried explicit ``steps``, exactly those entry ranges."""
        graphed = uproot.extras.graphed()

        if self._explicit_chunks is not None:
            if steps_per_file != 1:
                raise TypeError(
                    "explicit 'steps' in 'files' are incompatible with steps_per_file"
                )
            return tuple(
                graphed.core.Partition(
                    file_path, object_path or "", int(start), int(stop)
                )
                for (file_path, object_path), chunks in zip(
                    self._file_tree, self._explicit_chunks, strict=True
                )
                for start, stop in chunks
            )
        return tuple(
            graphed.core.Partition.blind(
                file_path, object_path or "", step, steps_per_file
            )
            for file_path, object_path in self._file_tree
            for step in range(steps_per_file)
        )

    def read_partition(self, partition, columns, resources):
        """One partition's chunk; ``columns`` is the declared read list (``None`` only from a
        hand-built plan: every selected branch)."""
        tree = self.open_tree(partition, resources)
        if tree is None:
            return self._empty()
        keys = list(columns) if columns is not None else list(self._common_keys)
        return self.read_range(
            tree, keys, *_partition_range(partition, tree.num_entries)
        )

    def projected_columns(self, outputs, *, on_fail="pass"):
        """The ``TBranches`` this source declares for ``outputs`` (asked driver-side through
        ``graphed.write.declared_columns``).

        ``graphed``'s buffer projection answers in the mapped form's dotted paths, so each is
        walked back to the buffer keys it needs — a list's structure is a buffer of its own — and
        those go to the mapping's ``keys_for_buffer_keys``. ``on_fail="pass"`` keeps an opaque node
        (a user callable ``graphed`` cannot see through) from raising: the node is treated as
        needing everything its inputs carry, so the read list stays a superset of what is needed.
        """
        graphed = uproot.extras.graphed()

        info = self._form_mapping_info
        buffer_keys = set()
        for output in outputs:
            needs = graphed.awkward.projection.project_buffers(
                output, on_fail=on_fail
            ).read_buffers.get(self._name, {})
            for path, need in needs.items():
                buffer_keys.update(
                    _mapped_buffer_keys(
                        self._expected_form,
                        path,
                        need is graphed.BufferNeed.DATA,
                        info.buffer_key,
                    )
                )
        return tuple(sorted(info.keys_for_buffer_keys(frozenset(buffer_keys))))


def _mapped_buffer_keys(form, path, data, buffer_key):
    """The mapped form's buffer keys for one projected ``path``: every structure crossed on the
    way down to it, plus the endpoint's own — recursively when its ``data`` is read, and the list
    offsets alone when only its shape is."""
    keys = set()
    fields = path.split(".") if path != "<root>" else []
    while fields:
        if form.is_record:
            form = form.content(fields.pop(0))
        else:  # a list/option wrapper between records: its structure shapes everything below
            keys.update(
                form.expected_from_buffers(buffer_key=buffer_key, recursive=False)
            )
            form = form.content
    keys.update(form.expected_from_buffers(buffer_key=buffer_key, recursive=data))
    return keys


def _name_from_object_path(file_tree):
    """The ``TTree``'s name when no file was opened to ask it (``known_base_form=``): the object
    path the caller gave, without its ``TDirectory`` prefix or ``;cycle`` suffix — the same name
    the opened ``TTree`` reports, so a source keeps its identity either way."""
    return (file_tree[0][1] or "").rpartition("/")[2].partition(";")[0] or "events"


def _file_and_path(file_path, object_path):
    """An already-open ``TTree`` given as a file is re-opened by path in the workers."""
    if isinstance(file_path, HasBranches):
        return file_path.file.file_path, file_path.object_path
    return file_path, object_path


def _uproot_source(array, what):
    """The one ``uproot.graphed`` source behind ``array``, as ``(node_id, source)``."""
    sources = [
        (nid, s)
        for nid, s in array.session.sources().items()
        if isinstance(s, _GraphedTTreeSource)
    ]
    if len(sources) != 1:
        raise TypeError(
            f"{what} supports exactly one uproot.graphed source per array; "
            f"this array is backed by {len(sources)}"
        )
    return sources[0]


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
    behavior=None,
    form_mapping=None,
    known_base_form=None,
    backend=None,
    decompression_executor=None,
    interpretation_executor=None,
    **options,
):
    """
    Args:
        files: The ``TTree``(s) to read, in any form accepted by :doc:`uproot._dask.dask` (a
            path with ``"file.root:tree"``, a list, a dict, an open ``TTree``, a dict entry with
            explicit ``"steps"``; a single-``TTree`` file needs no object path).
        filter_name, filter_typename, filter_branch, recursive, full_paths: ``TBranch`` selection,
            as in :doc:`uproot._dask.dask`.
        library (str): Only ``"ak"`` is supported (a single deferred ``graphed`` array). ``"np"`` and
            ``"pd"`` raise ``NotImplementedError``.
        ak_add_doc, custom_classes, allow_missing: As in :doc:`uproot._dask.dask`.
        behavior (dict or None): An awkward behavior dict (e.g. ``vector``'s) registered on the
            recording backend, for the default backend and the unmapped form: with
            ``gak.with_name``, behavior PROPERTIES (``.pt``, ``.mass``) work through plain attribute
            access — typetracer forms at record time, projectable down to exactly the branches a
            property reads.
        form_mapping (ImplementsFormMapping or None): The mapping :doc:`uproot._dask.dask` takes —
            it restructures the flat ``TTree`` form into the one the user sees (coffea's
            ``NanoEventsFactory`` is the archetype) and fills that form's buffers from the file.
            The recorded array then has the MAPPED form and the mapping info's ``behavior`` is
            registered on the backend. Without one, ``TrivialFormMapping`` reads the file's form.
        known_base_form (awkward.forms.Form | None): If not none use this form instead of opening
            one file to determine the dataset's form, as in :doc:`uproot._dask.dask`. No file is
            opened at all; the ``TTree``'s name then comes from the object path in ``files``.
        backend (graphed Backend or None): The ``graphed`` backend **instance** the session
            records on, in place of the default ``graphed.awkward.AwkwardBackend`` — the route for
            a caller's own ``Array`` subclass. Its behavior (and, with ``form_mapping``, the
            mapping info's) is then the caller's business, so ``behavior=`` is refused beside it.
        decompression_executor, interpretation_executor: As in :doc:`uproot._dask.dask`; every
            partition read uses them.
        options: Passed through to file opening.

    Returns a deferred ``graphed`` ``Array`` for the selected ``TTree``(s). Construction reads only
    metadata; running the recorded analysis (``graphed.aggregate_plan``,
    ``graphed.awkward.to_parquet``, :doc:`uproot.writing._graphed_write.graphed_write`) triggers
    the read, partition by partition, fetching only the ``TBranches`` the recorded graph touches.

    Partitioning is decided where a plan is built (``steps_per_file=`` there), not here, so
    ``step_size``, ``steps_per_file`` and ``open_files`` are refused by name.

    **Experimental**: tracks a pre-1.0 ``graphed``; the API may change without a deprecation cycle.

    This is the ``graphed`` analogue of :doc:`uproot._dask.dask`.
    """
    for name, instead in _DASK_ONLY.items():
        if name in options:
            raise TypeError(
                f"uproot.graphed() got {name}=, which belongs to uproot.dask; {instead}"
            )
    library = uproot.interpretation.library._regularize_library(library)
    if library.name != "ak":
        raise NotImplementedError(
            f"uproot.graphed currently supports only library='ak', not {library.name!r}"
        )
    if behavior is not None and (backend is not None or form_mapping is not None):
        raise TypeError(
            "uproot.graphed: behavior= is for the default backend and the unmapped form; with "
            "backend= the behavior belongs to that backend, with form_mapping= to its mapping info"
        )

    import awkward

    graphed = uproot.extras.graphed()

    real_options = options.copy()
    real_options.setdefault("num_workers", 1)
    real_options.setdefault("num_fallback_workers", 1)
    filter_branch = uproot._util.regularize_filter(filter_branch)
    interp_options = {"ak_add_doc": ak_add_doc}

    files = uproot._util.regularize_files(files, steps_allowed=True, **options)
    is_3arg = [len(x) == 3 for x in files]
    if any(is_3arg) and not all(is_3arg):
        raise TypeError(
            "partition sizes for some but not all 'files' have been assigned"
        )

    if known_base_form is not None:  # the form is given: nothing is opened
        common_keys = list(dict.fromkeys(known_base_form.fields))
        base_form = known_base_form
        file_tree = [_file_and_path(f, o) for f, o, *_ in files]
        explicit_chunks = [f[2] for f in files] if all(is_3arg) else None
        name = _name_from_object_path(file_tree)
    else:
        ttrees, common_keys, _is_self, explicit_chunks = _resolve_trees_and_keys(
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
        common_keys = _normalize_grouped_keys(ttrees[0], common_keys, full_paths)
        base_form = _base_form_of(ttrees[0], common_keys, ak_add_doc, form_mapping)
        file_tree = [(t.file.file_path, t.object_path) for t in ttrees]
        name = getattr(ttrees[0], "name", None) or _name_from_object_path(file_tree)

    expected_form, info = _mapping_of(form_mapping, base_form)
    typetracer = awkward.typetracer.typetracer_from_form(
        expected_form, highlevel=True, behavior=info.behavior
    )
    source = _GraphedTTreeSource(
        file_tree,
        common_keys,
        name=name,
        expected_form=expected_form,
        form_mapping_info=info,
        interp_options=interp_options,
        custom_classes=custom_classes,
        allow_missing=allow_missing,
        options=real_options,
        decompression_executor=decompression_executor,
        interpretation_executor=interpretation_executor,
        explicit_chunks=explicit_chunks,
    )
    if backend is None:
        backend = graphed.awkward.AwkwardBackend(
            behavior=behavior if form_mapping is None else info.behavior
        )
    return graphed.Session(backend).source(
        name, form=graphed.awkward.AwkwardForm(typetracer), data=source
    )


def necessary_columns(array, *, on_fail="raise"):
    """The ``TBranches`` each ``uproot.graphed`` source must read for ``array`` — ``graphed``'s
    buffer projection walked back through the form mapping (metadata-only). Returns
    ``{source_name: frozenset(branch_names)}``."""
    return {
        array.session.source_name(nid): frozenset(
            source.projected_columns((array,), on_fail=on_fail)
        )
        for nid, source in array.session.sources().items()
        if isinstance(source, _GraphedTTreeSource)
    }


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
    """Partition a uproot dataset into ``graphed.core.Partition`` chunks, for
    ``graphed.aggregate_plan(..., partitions=...)`` or a hand-built ``graphed.core.Plan``.

    Mirrors :doc:`uproot._dask.dask`'s chunking knobs:

    - ``steps_per_file`` (default 1): split each ``TTree`` into that many contiguous chunks.
    - ``step_size`` (int entries, or a memory string like ``"100 MB"``): cap each chunk's size.
      Mutually exclusive with ``steps_per_file``, and **incompatible with** ``open_files=False``.
    - ``open_files`` (default ``True``): open every file to read its entry count and emit exact
      entry ranges. ``open_files=False`` emits **blind** partitions (``Partition.blind``): files
      are not opened here and the entry range is resolved against the file's own count when the
      partition is read (:doc:`uproot._graphed.read_graphed_partition`)."""
    graphed = uproot.extras.graphed()

    have_step_size = not isinstance(step_size, uproot._util._Unset)
    have_steps_per_file = not isinstance(steps_per_file, uproot._util._Unset)
    if have_step_size and not open_files:
        raise TypeError(
            "step_size cannot be used with open_files=False; use steps_per_file"
        )
    if have_step_size and have_steps_per_file:
        raise TypeError(
            "step_size and steps_per_file are mutually exclusive; set only one"
        )
    n_steps = int(steps_per_file) if have_steps_per_file else 1

    real_options = options.copy()
    real_options.setdefault("num_workers", 1)
    resolved = uproot._util.regularize_files(files, steps_allowed=False, **options)

    partitions = []
    for file_path, object_path, *_ in resolved:
        if not open_files:
            partitions.extend(
                graphed.core.Partition.blind(
                    file_path, object_path or "", step, n_steps
                )
                for step in range(n_steps)
            )
            continue
        obj = uproot._util.regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, real_options
        )
        if obj is None:
            continue
        n_entries = obj.num_entries
        if have_step_size:
            per = (
                step_size
                if isinstance(step_size, int)
                else obj.num_entries_for(step_size)
            )
            per = max(1, int(per))
            ranges = [(s, min(s + per, n_entries)) for s in range(0, n_entries, per)]
        else:
            ranges = [
                ((i * n_entries) // n_steps, ((i + 1) * n_entries) // n_steps)
                for i in range(n_steps)
            ]
        partitions.extend(
            graphed.core.Partition(
                obj.file.file_path, obj.object_path, int(start), int(stop)
            )
            for start, stop in ranges
            if stop > start
        )
    return partitions


def read_graphed_partition(
    partition, columns, *, tree=None, library="ak", **open_options
):
    """Read a ``graphed.core.Partition``'s chunk of ``columns`` from its ROOT file, resolving a
    **blind** partition against the file's actual entry count. Pass an already-open ``tree`` to
    reuse a per-worker ``open_once`` handle."""
    if tree is None:
        tree = uproot.open(partition.uri, **open_options)[partition.tree]
    start, stop = _partition_range(partition, tree.num_entries)
    return tree.arrays(
        list(columns), entry_start=start, entry_stop=stop, library=library
    )


def _partition_range(partition, n_entries):
    """A partition's concrete ``(entry_start, entry_stop)`` against a tree's entry count."""
    partition = partition.resolve(n_entries)
    return partition.entry_start, partition.entry_stop


def graphed_head(array, n=5):
    """EAGER peek at the first ``n`` rows of a recorded analysis's OUTPUT, reading only the first
    file's leading entries (and only the branches the graph needs) and evaluating the compiled IR
    over them — never the whole dataset. The prefix read grows until ``n`` output rows exist or the
    first file is exhausted, so a selection or an axis-0 slice peeks at its own first rows.

    The ``graphed`` analogue of a dask collection's ``head``.
    """
    graphed = uproot.extras.graphed()

    session = array.session
    nid, source = _uproot_source(array, "graphed_head")
    columns = list(source.projected_columns((array,)))
    compiled = graphed.compile_ir(session, array)
    externals = graphed.aggregate.external_evaluators(session, compiled)
    source_name = session.source_name(nid)

    tree = next(source._trees(), None)
    rows = 0 if tree is None else tree.num_entries
    stop = min(int(n), rows)
    while True:
        chunk = (
            source._empty()
            if tree is None
            else source.read_range(tree, columns, 0, stop)
        )
        (out,) = graphed.evaluate_ir(
            compiled, session.backend, {source_name: chunk}, externals=externals
        )
        try:
            n_out = len(out)
        except TypeError:  # a scalar peek (a reduction over the prefix)
            return out
        if n_out >= n or stop >= rows:
            return out[:n]
        stop = min(rows, max(2 * stop, int(n)))
