# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""
This module defines :doc:`uproot.writing._graphed_write.graphed_write`, the ``graphed`` analogue of
:doc:`uproot.writing._dask_write.dask_write` — and a SPECIALIZATION of the ``graphed.write``
partitioned-write base.

**Experimental**: tracks a pre-1.0 ``graphed``; the API may change without a deprecation cycle.

Like ``dask_write``, it produces **one output ROOT file per input partition**, written by the
same ``ak_to_root`` (a ``TTree`` through ``mktree``, the same compression names and tree
options). The partitions are BLIND (the driver opens no files — each worker resolves its entry
range against its own file); each write task evaluates the recorded graph over its chunk, writes
``{prefix}-{N:05d}.root`` and **reports its part path** up the plan's deterministic combine tree;
a step that resolves EMPTY writes no part file. With ``compute=False`` the Plan is returned
without running; with ``compute=True`` (the default) it is executed through a ``graphed-executors``
executor.

``graphed`` / ``graphed_executors`` are imported lazily, so importing ``uproot`` does not require
them.
"""

from __future__ import annotations

import functools

from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

import uproot
from uproot.writing._dask_write import ak_to_root


# ---- module-level so a spawned ProcessPoolExecutor worker can pickle/import them --------------
def _default_field_name(outer, inner):
    return inner if outer == "" else outer + "_" + inner


def _default_counter_name(counted):
    return "n" + counted


def _write_partition(
    partition,
    resources,
    *,
    base,
    sep,
    prefix,
    columns,
    bases,
    compiled,
    backend,
    externals,
    source,
    source_name,
    root_options,
):
    """Read this blind partition's chunk (file opened once per worker), EVALUATE the compiled
    graph over it, and write the evaluated record's fields as ``TBranches`` of its own part file,
    REPORTING the written path. ``N`` = the file's base + this partition's blind step, derived here
    from the partition alone (``graphed.write.blind_part_index``). An empty resolved step, or a
    file without the tree (``allow_missing``), writes nothing."""
    graphed = uproot.extras.graphed()

    tree = source.open_tree(partition, resources)
    if tree is None:
        return []
    start, stop = source.entry_range(partition, tree, columns)
    if stop <= start:
        return []
    chunk = source.read_range(tree, columns, start, stop)
    (evaluated,) = graphed.evaluate_ir(
        compiled,
        graphed.resolve_backend(backend),
        {source_name: chunk},
        externals=dict(externals),
    )
    index = graphed.write.blind_part_index(partition, dict(bases))
    path = f"{base}{sep}{prefix}-{index:05d}.root"
    ak_to_root(path, evaluated, **root_options)
    return [path]


def _select_executor(executor, max_workers):
    if isinstance(executor, str):
        graphed_executors = uproot.extras.graphed_executors()

        classes = {
            "process": graphed_executors.local.ProcessPoolExecutor,
            "thread": graphed_executors.local.ThreadExecutor,
        }
        if executor not in classes:
            raise ValueError(
                f"executor must be 'process' or 'thread', an executor instance, or an executor "
                f"class; got {executor!r}"
            )
        return classes[executor](max_workers=max_workers)
    if isinstance(executor, type):
        return executor(max_workers=max_workers)
    return executor  # an instance, used as-is


def graphed_write(
    array,
    destination,
    *,
    steps_per_file=1,
    prefix=None,
    tree_name="tree",
    title="",
    field_name=_default_field_name,
    initial_basket_capacity=10,
    counter_name=_default_counter_name,
    resize_factor=10.0,
    compression="zlib",
    compression_level=1,
    storage_options=None,
    compute=True,
    executor="process",
    max_workers=None,
    backend=None,
):
    """
    Args:
        array (``graphed.Array``): A record recorded over a ``uproot.graphed`` source — its fields
            become the ``TBranches``. A fieldless expression (``g.x * 2``) has no branch name and
            is refused: wrap it, ``gak.zip({"x2": g.x * 2})``.
        destination (path-like): Output **directory**, local or remote (``storage_options``
            reach its filesystem, as in ``dask_write``); the part files are written inside it.
        steps_per_file (int): Split each input ``TTree`` into this many contiguous output
            partitions (blind — resolved by each worker; the driver opens no files). Files given
            with explicit ``steps`` are not writable this way: a worker derives its part index from
            the partition alone, which explicit ranges do not allow.
        prefix (str or None): Part files are named ``f"{prefix or 'part'}-{N:05d}.root"``.
        tree_name, title, field_name, initial_basket_capacity, counter_name, resize_factor,
            compression, compression_level: As in :doc:`uproot.writing._dask_write.dask_write`
            (one ``TTree`` per part, through ``mktree``). ``field_name`` and ``counter_name`` must
            be picklable for the process executor (the defaults are).
        compute (bool): If ``True`` (default), execute the write task graph now and return the
            written paths (reported by the workers, in deterministic key order). If ``False``,
            return the ``graphed.core.Plan`` **without writing** — run it later with an executor.
        executor (str, executor instance or class): ``"process"`` (default,
            ``ProcessPoolExecutor``) or ``"thread"`` (``ThreadExecutor``); an executor **instance**
            is used as-is, a class is called as ``executor(max_workers=max_workers)``.
        max_workers (int or None): Worker count for an executor built here.
        backend (callable, str or None): The workers' evaluation backend, as in
            ``graphed.aggregate_plan``: a zero-arg factory/class or an importable ``"module:attr"``
            reference, resolved in the worker; default the session backend's type. A backend whose
            behavior holds lambdas (``vector``'s) does not pickle — pass it by reference.

    Produces one ROOT file per partition, mirroring :doc:`uproot.writing._dask_write.dask_write`,
    as a specialization of the ``graphed.write`` partitioned-write base. A reduction on the
    partitioned axis (an axis-0 slice, ``gak.sum``) is refused: each part would hold its own
    chunk's partial.

    **Experimental**: tracks a pre-1.0 ``graphed``; the API may change without a deprecation cycle.
    """
    graphed = uproot.extras.graphed()

    if not isinstance(array, graphed.Array):
        raise TypeError("graphed_write expects a uproot.graphed Array")

    from uproot._graphed import _uproot_source

    session = array.session
    nid, source = _uproot_source(array, "graphed_write")
    if not session.form(array).tt.fields:
        raise TypeError(
            "graphed_write writes a record's fields as TBranches; this array has no fields. "
            'Name the result: gak.zip({"name": expr})'
        )

    compiled = graphed.compile_ir(session, array)
    graphed.refuse_chunk_partials(compiled, as_outputs=True)
    partitions = source.partitions(steps_per_file)
    if not all(p.is_blind for p in partitions):
        raise TypeError(
            "graphed_write needs blind partitions (a worker derives its part index from the "
            "partition alone); 'files' with explicit 'steps' cannot be written — use "
            "steps_per_file"
        )
    bases = graphed.write.file_bases(
        [
            (file_path, object_path or "")
            for file_path, object_path in source._file_tree
        ],
        steps_per_file,
    )

    fs, root = url_to_fs(destination, **(storage_options or {}))
    fs.mkdirs(root, exist_ok=True)
    process = functools.partial(
        _write_partition,
        # a local destination reports plain paths; a remote one its full URLs
        base=root if isinstance(fs, LocalFileSystem) else fs.unstrip_protocol(root),
        sep=fs.sep,
        prefix=prefix or "part",
        columns=list(source.projected_columns((array,))),
        bases=tuple(bases.items()),
        compiled=compiled,
        backend=backend if backend is not None else type(session.backend),
        externals=tuple(
            graphed.aggregate.external_evaluators(session, compiled).items()
        ),
        source=source,
        source_name=session.source_name(nid),
        root_options={
            "tree_name": tree_name,
            "compression": compression,
            "compression_level": compression_level,
            "title": title,
            "counter_name": counter_name,
            "field_name": field_name,
            "initial_basket_capacity": initial_basket_capacity,
            "resize_factor": resize_factor,
            "storage_options": storage_options,
        },
    )
    plan = graphed.write.write_plan(partitions, process)

    if not compute:
        return plan
    return list(_select_executor(executor, max_workers).run(plan).value)
