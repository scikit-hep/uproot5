# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""
This module defines :doc:`uproot._graphed.graphed`, which reads ``TTrees`` into a deferred
`graphed <https://github.com/graphed-org/graphed>`__ array — the analogue of
:doc:`uproot._dask.dask` for the ``graphed`` task-graph system.

``uproot.graphed(files, library="ak")`` returns a deferred ``graphed`` ``Array`` (recorded on the
``graphed.awkward`` backend); construction reads only metadata (the ``TTree`` form). ``graphed`` does
not impersonate a deferred-array ``.compute()`` — instead, a recorded analysis is executed the way one
actually runs a task graph: per partition through the ``graphed-executors`` executors
(``ProcessPoolExecutor`` / ``ThreadExecutor``) and tree-reduced. :doc:`uproot._graphed.graphed_partitions`
builds the ``graphed.core.Task`` chunks; :doc:`uproot._graphed.necessary_columns` reports the column
projection (so each chunk reads only the ``TBranches`` the analysis touches — the dask-awkward
necessary-columns optimization, expressed through ``graphed``).

With a ``form_mapping`` (coffea's ``NanoEventsFactory`` is the archetype) the source reads through
the mapping instead: each chunk is assembled by the mapping's ``load_buffers`` and
``awkward.from_buffers``, and because the graph's field names are then the MAPPED form's, the
source DECLARES its read list in ``TBranch`` names rather than letting a driver guess them.

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

    def _resolve(self, file_path, object_path):
        """Re-resolve one ``(file_path, object_path)``: ``_file_tree`` holds only what resolved
        when ``graphed`` built it; as in uproot._dask, the read expects it to resolve again.
        """
        return uproot._util.regularize_object_path(
            file_path,
            object_path,
            self._custom_classes,
            self._allow_missing,
            self._options,
        )

    def read_range(self, tree, columns, start, stop):
        """Read ``columns`` for entries ``[start, stop)`` from an already-open ``TTree``."""
        return tree.arrays(
            list(columns), entry_start=start, entry_stop=stop, library="ak"
        )

    def __call__(self):
        import awkward

        cols = (
            list(self.columns) if self.columns is not None else list(self._common_keys)
        )
        self.last_columns_read = list(cols)
        trees = [self._resolve(*file_tree) for file_tree in self._file_tree]
        parts = [self.read_range(tree, cols, 0, tree.num_entries) for tree in trees]
        return parts[0] if len(parts) == 1 else awkward.concatenate(parts)

    # ---- graphed.write.PartitionedSource: partition-wise reading -----------------------------
    def partitions(self, steps_per_file=1):
        """BLIND partitions, one per (file x step): planning opens no files."""
        graphed = uproot.extras.graphed()

        return tuple(
            graphed.core.Partition.blind(file_path, object_path, s, steps_per_file)
            for (file_path, object_path) in self._file_tree
            for s in range(steps_per_file)
        )

    def read_partition(self, partition, columns, resources):
        """Read one partition's branches. The file is opened once per worker (via ``resources``)
        with the options ``uproot.graphed`` was given, so ``decompression_executor`` /
        ``interpretation_executor`` reach the read the way they do on any other uproot read.
        """
        tree = resources.open_once(
            partition.uri, lambda uri: uproot.open(uri, **self._options)
        )[partition.tree]
        return self.read_range(
            tree,
            self._read_columns(columns),
            *_partition_range(partition, tree.num_entries),
        )

    def _read_columns(self, columns):
        """The branches one partition reads. This source declares no read list, so an empty one
        means the driver's syntactic walk named no branch (a whole-record read), not a request to
        read nothing: fall back to every selected branch."""
        return list(columns) if columns else list(self._common_keys)


class _MappedGraphedTTreeSource(_GraphedTTreeSource):
    """A ``_GraphedTTreeSource`` read through a ``form_mapping``: every chunk is assembled by the
    mapping's own ``load_buffers`` and ``awkward.from_buffers`` (a branch outside the read list
    becomes a placeholder, as in ``uproot._dask``'s ``UprootReadMixin.read_tree``), and the source
    DECLARES its read list, because the graph's field names are the MAPPED form's, not the file's
    ``TBranch`` names."""

    def __init__(self, *args, name, expected_form, form_mapping_info, interp_options):
        super().__init__(*args)
        self._name = name  # the graphed source name, which the projection answers under
        self._expected_form = expected_form
        self._form_mapping_info = form_mapping_info
        self._interp_options = interp_options

    def read_range(self, tree, columns, start, stop):
        import awkward
        from awkward._nplikes.numpy import Numpy

        info = self._form_mapping_info
        keys = frozenset(columns)
        buffers = info.load_buffers(
            tree, keys, start, stop, None, None, self._interp_options
        )
        container = {}
        for buffer_key, dtype in self._expected_form.expected_from_buffers(
            buffer_key=info.buffer_key
        ).items():
            if info.keys_for_buffer_keys(frozenset({buffer_key})) <= keys:
                container[buffer_key] = buffers[buffer_key]
            else:
                container[buffer_key] = awkward.typetracer.PlaceholderArray(
                    nplike=Numpy.instance(),
                    shape=(awkward.typetracer.unknown_length,),
                    dtype=dtype,
                )
        return awkward.from_buffers(
            self._expected_form,
            stop - start,
            container,
            behavior=info.behavior,
            buffer_key=info.buffer_key,
        )

    def _read_columns(self, columns):
        """This source DECLARES its read list, so the declaration is honoured verbatim — an empty
        one included. Every buffer outside it is a placeholder and the chunk's length comes from
        the partition's entry range, so an output that needs only a length
        (``gak.num(events, axis=0)``) reads no ``TBranch`` at all. Only ``None`` (nothing declared)
        falls back to every selected branch."""
        return list(columns) if columns is not None else list(self._common_keys)

    def projected_columns(self, outputs, *, on_fail="pass"):
        """The ``TBranches`` this source declares for ``outputs``, asked driver-side through
        ``graphed.write.declared_columns``.

        ``graphed``'s buffer projection answers in the mapped form's dotted paths, so each is
        walked back to the buffer keys it needs — a list's structure is a buffer of its own — and
        those go to the mapping's ``keys_for_buffer_keys``. ``on_fail="pass"`` keeps an opaque node
        (a user callable ``graphed`` cannot see through) from raising: the node is treated as
        needing everything its inputs carry, so the read list stays a superset of what is needed.
        Refusing instead would leave this source with no read list at all."""
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
    object_path = file_tree[0][1] if file_tree else None
    return (object_path or "").rpartition("/")[2].partition(";")[0] or "events"


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
        behavior (dict or None): An awkward behavior dict (e.g. ``vector``'s) registered on the
            recording backend: with ``gak.with_name``, behavior PROPERTIES
            (``.pt``, ``.mass``) work through plain attribute access — typetracer forms at record
            time, projectable down to exactly the branches a property reads.
        form_mapping (ImplementsFormMapping or None): The mapping :doc:`uproot._dask.dask` takes —
            it restructures the flat ``TTree`` form into the one the user sees (coffea's
            ``NanoEventsFactory`` is the archetype) and fills that form's buffers from the file.
            The recorded array then has the MAPPED form, the mapping info's ``behavior`` is
            registered on the backend, and the source declares its read list in ``TBranch`` names
            (see :doc:`uproot._graphed.necessary_columns`).
        known_base_form (awkward.forms.Form | None): If not none use this form instead of opening
            one file to determine the dataset's form, as in :doc:`uproot._dask.dask`. No file is
            opened at all; the ``TTree``'s name then comes from the object path in ``files``.
        backend (graphed Backend or None): The ``graphed`` backend **instance** the session
            records on, in place of the default ``graphed.awkward.AwkwardBackend`` — the route for
            a caller's own ``Array`` subclass. Its behavior (and, with ``form_mapping``, the
            mapping info's) is then the caller's business, so ``behavior=`` is refused beside it.
        options: Passed through to file opening — including ``decompression_executor`` and
            ``interpretation_executor``, which every read here takes from the file it opened
            (``TTree.arrays`` falls back to the file's), so they need no parameter of their own.

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
    if behavior is not None and (backend is not None or form_mapping is not None):
        raise TypeError(
            "uproot.graphed: behavior= is for the default backend and the unmapped form; with "
            "backend= the behavior belongs to that backend, with form_mapping= to its mapping info"
        )

    import awkward

    graphed = uproot.extras.graphed()

    real_options = options.copy()
    real_options.setdefault("num_workers", 1)
    filter_branch = uproot._util.regularize_filter(filter_branch)

    resolved = uproot._util.regularize_files(files, steps_allowed=False, **options)

    file_tree = []
    common_keys = None
    first_ttree = None
    for ftuple in resolved:
        file_path, object_path = ftuple[0], ftuple[1]
        if known_base_form is not None:  # the form is given: nothing is opened
            file_tree.append((file_path, object_path))
            continue
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

    if known_base_form is not None:
        record_form = known_base_form
        common_keys = list(dict.fromkeys(known_base_form.fields))
    else:
        if first_ttree is None:
            raise ValueError("uproot.graphed: no TTrees found in the given files")
        if not common_keys:
            raise ValueError("uproot.graphed: the TTrees have no TBranches in common")
        # RNTuples (HasFields) expose the awkward form directly; TTrees build it from branch interpretations
        if isinstance(first_ttree, HasFields):
            record_form = first_ttree.to_akform(filter_name=common_keys)[0]
        else:
            record_form = _get_ttree_form(awkward, first_ttree, common_keys, ak_add_doc)
        # as in uproot._dask, a mapping may read the branch typenames off the base form
        if form_mapping is not None:
            record_form.parameters["typenames"] = first_ttree.typenames()

    name = getattr(first_ttree, "name", None) or _name_from_object_path(file_tree)
    args = (file_tree, common_keys, custom_classes, allow_missing, real_options)
    if form_mapping is None:
        typetracer = awkward.Array(
            record_form.length_zero_array(highlevel=False).to_typetracer(
                forget_length=True
            )
        )
        source = _GraphedTTreeSource(*args)
    else:
        expected_form, info = form_mapping(record_form)
        typetracer = awkward.typetracer.typetracer_from_form(
            expected_form, highlevel=True, behavior=info.behavior
        )
        source = _MappedGraphedTTreeSource(
            *args,
            name=name,
            expected_form=expected_form,
            form_mapping_info=info,
            interp_options={"ak_add_doc": ak_add_doc},
        )
        behavior = info.behavior

    if backend is None:
        backend = graphed.awkward.AwkwardBackend(behavior=behavior)
    return graphed.Session(backend).source(
        name, form=graphed.awkward.AwkwardForm(typetracer), data=source
    )


def _mapped_branches(array, *, on_fail):
    """``{source_name: (branch_name, ...)}`` for the form-mapped sources backing ``array``: the
    names the FILE has, which the projection's mapped field paths are not."""
    return {
        array.session.source_name(nid): source.projected_columns(
            (array,), on_fail=on_fail
        )
        for nid, source in array.session.sources().items()
        if isinstance(source, _MappedGraphedTTreeSource)
    }


def necessary_columns(array, *, on_fail="raise"):
    """The ``TBranches`` each source must read for ``array`` — ``graphed``'s necessary-buffer
    projection (metadata-only). Returns ``{source_name: frozenset(branch_names)}``."""
    graphed = uproot.extras.graphed()

    columns = dict(
        graphed.awkward.projection.project(array, on_fail=on_fail).read_columns
    )
    columns.update(
        {
            name: frozenset(branches)
            for name, branches in _mapped_branches(array, on_fail=on_fail).items()
        }
    )
    return columns


def necessary_buffers(array, *, on_fail="raise"):
    """Buffer-granular projection: per source, each needed column with its
    :class:`graphed.BufferNeed` (``DATA`` — the leaf values are read; ``OFFSETS`` — only the list
    STRUCTURE is needed, e.g. a multiplicity). Strictly finer than :doc:`necessary_columns`: a
    count-only analysis truthfully reports ``{collection: OFFSETS}`` where the column view reports
    the empty set — feed it to :doc:`resolve_read_branches` to serve the count from the jagged
    branch's COUNTER branch without reading the payload baskets.

    Through a ``form_mapping`` the answer is in ``TBranch`` names and every need is ``DATA``: the
    mapping reads whole branches, and there a list's structure IS a branch (its counter).
    """
    graphed = uproot.extras.graphed()

    buffers = {
        name: dict(needs)
        for name, needs in graphed.awkward.projection.project_buffers(
            array, on_fail=on_fail
        ).read_buffers.items()
    }
    buffers.update(
        {
            name: dict.fromkeys(branches, graphed.BufferNeed.DATA)
            for name, branches in _mapped_branches(array, on_fail=on_fail).items()
        }
    )
    return buffers


def resolve_read_branches(obj, needs):
    """Translate buffer needs from :doc:`necessary_buffers` into the concrete branches to read
    from ``obj`` (an open ``TTree`` or ``RNTuple``).

    ``DATA`` needs read the branch itself. An ``OFFSETS``-only need reads the jagged ``TBranch``'s
    **counter branch** when the file provides one (``TBranch.count_branch``) — the list lengths
    without the payload baskets; where no counter exists (or for ``RNTuple``, whose index column is
    not independently addressable through the public API) it falls back to the branch itself.
    Returns ``{branch_to_read: requested_path}``."""
    graphed = uproot.extras.graphed()

    out = {}
    for path, need in needs.items():
        if need is graphed.BufferNeed.DATA or str(need) == "data":
            out[path] = path
            continue
        counter = None
        if not isinstance(
            obj, HasFields
        ):  # TTree: jagged branches carry a counter branch
            try:
                counter = obj[path].count_branch
            except (KeyError, AttributeError):
                counter = None
        out[counter.name if counter is not None else path] = path
    return out


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
    """Partition a uproot dataset into ``graphed.core.Task`` chunks for the ``graphed-executors``
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

    The chunks feed a ``graphed.core.Plan`` run by ``graphed_executors.local.ProcessPoolExecutor`` /
    ``ThreadExecutor`` — the per-partition, tree-reduced execution that a deferred-array ``.compute()``
    hides."""
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

    tasks = []
    key = 0
    for ftuple in resolved:
        file_path, object_path = ftuple[0], ftuple[1]
        if not open_files:
            # BLIND: do not open the file; emit first-class blind chunks
            # (``graphed.core.Partition.blind`` records (step, n_steps) explicitly),
            # resolved against the file's actual entry count at read time
            for step in range(n_steps):
                tasks.append(
                    graphed.core.Task(
                        key,
                        graphed.core.Partition.blind(
                            file_path, object_path, step, n_steps
                        ),
                    )
                )
                key += 1
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
        for start, stop in ranges:
            if stop > start:
                tasks.append(
                    graphed.core.Task(
                        key,
                        graphed.core.Partition(
                            file_path, object_path, int(start), int(stop)
                        ),
                    )
                )
                key += 1
    return tasks


def read_graphed_partition(
    partition, columns, *, tree=None, library="ak", **open_options
):
    """Read a ``graphed.core.Partition``'s chunk of ``columns`` from its ROOT file.

    Resolves **blind** partitions (``entry_stop < 0`` encodes ``step_index`` / ``n_steps`` from
    ``graphed_partitions(..., open_files=False)``) against the file's *actual* entry count here, so a
    blindly-stepped dataset still reads every entry exactly once. Pass an already-open ``tree`` to reuse
    a per-worker ``open_once`` handle."""
    if tree is None:
        tree = uproot.open(partition.uri, **open_options)[partition.tree]
    start, stop = _partition_range(partition, tree.num_entries)
    return tree.arrays(
        list(columns), entry_start=start, entry_stop=stop, library=library
    )


def _partition_range(partition, n_entries):
    """A partition's concrete ``(entry_start, entry_stop)`` against a tree's entry count."""
    if getattr(partition, "is_blind", False):
        partition = partition.resolve(n_entries)
    start, stop = partition.entry_start, partition.entry_stop
    if (
        stop < 0
    ):  # legacy blind sentinel in older serialized plans: step `start` of `-stop` steps
        n_steps, step = -stop, start
        start = (step * n_entries) // n_steps
        stop = ((step + 1) * n_entries) // n_steps
    return start, stop


def _evaluation_columns(array, source_node_id, common_keys):
    """The per-task read list: every branch the recorded graph SYNTACTICALLY accesses on the
    source. Compiled-IR evaluation replays every node — including field accesses whose BUFFERS
    the output never touches (e.g. the pz/E legs of a zip whose only consumed property is .pt) —
    so the buffer projection (what data is needed) UNDER-supplies evaluation (what fields must
    exist). The structure-only/offsets story is unchanged: a jagged branch read here is also its
    own offsets carrier."""
    needed = set()
    sentinel = object()

    def on_source(nid):
        return (sentinel, nid)

    def on_op(_nid, name, ins, params):
        is_source_input = any(
            isinstance(x, tuple)
            and len(x) == 2
            and x[0] is sentinel
            and x[1] == source_node_id
            for x in ins
        )
        if is_source_input:
            if name == "field":
                needed.add(str(params["field"]))
            elif name == "fields":
                needed.update(f for f in str(params["fields"]).split(",") if f)
            else:
                needed.update(
                    common_keys
                )  # a non-field op consumes the whole source record
        return None

    array.session.walk(
        array, source=on_source, op=on_op, external=lambda _n, _f, ins: None
    )
    if (
        not needed
    ):  # a bare source read (the array IS the source): every selected branch
        return tuple(common_keys)
    return tuple(k for k in common_keys if k in needed)


def _task_columns(array, source_node_id, source):
    """The branches one task must read to evaluate ``array``. A form-mapped source DECLARES them
    — its graph field names are not branch names — where an unmapped one reads every branch the
    graph names."""
    if isinstance(source, _MappedGraphedTTreeSource):
        return source.projected_columns((array,))
    return _evaluation_columns(array, source_node_id, source._common_keys)


def graphed_head(array, n=5):
    """EAGER peek at the first ``n`` rows of a recorded analysis, reading ONLY the first file's
    leading entries (and only the branches the graph accesses) and evaluating through the
    compiled IR — never the whole dataset. Clamps to the first file's entry count.

    The ``graphed`` analogue of a dask collection's ``head``.
    """
    graphed = uproot.extras.graphed()

    session = array.session
    uproot_sources = [
        (nid, s)
        for nid, s in session.sources().items()
        if isinstance(s, _GraphedTTreeSource)
    ]
    if len(uproot_sources) != 1:
        raise TypeError(
            f"graphed_head supports exactly one uproot.graphed source per array; "
            f"this array is backed by {len(uproot_sources)}"
        )
    nid, source = uproot_sources[0]
    columns = _task_columns(array, nid, source)
    compiled = graphed.compile_ir(session, array)

    obj = source._resolve(*source._file_tree[0])
    stop = min(int(n), obj.num_entries)
    chunk = source.read_range(obj, list(columns), 0, stop)
    (out,) = graphed.evaluate_ir(
        compiled, session.backend, {session.source_name(nid): chunk}
    )
    return out
