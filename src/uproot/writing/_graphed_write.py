# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""
This module defines :doc:`uproot.writing._graphed_write.graphed_write`, the ``graphed`` analogue of
:doc:`uproot.writing._dask_write.dask_write` — and a SPECIALIZATION of the ``graphed.write``
partitioned-write base.

Like ``dask_write``, it produces **one output ROOT file per input partition**. The partitions are
BLIND (the driver opens no files — each worker resolves its entry range against its own file);
each write task writes ``{prefix}-{N:05d}.root`` and **reports its part path** up the plan's
deterministic combine tree (the base's contract: a write task returns the paths it wrote);
a step that resolves EMPTY (a file with fewer entries than ``steps_per_file``) is skipped, so no
empty part files are written (part numbering may then have gaps in that corner case). With
``compute=False`` the Plan (the write task graph) is returned without running; with
``compute=True`` (the default) it is executed through a ``graphed-executors`` executor — the
``ProcessPoolExecutor`` by default (``executor="thread"`` for the thread pool).

``graphed`` / ``graphed.core`` / ``graphed_executors.local`` are imported lazily, so importing ``uproot``
does not require them.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import uproot


def _recreate_kwargs(compression, compression_level):
    if compression is None:
        return {}
    codes = {
        "zlib": uproot.ZLIB,
        "lzma": uproot.LZMA,
        "lz4": uproot.LZ4,
        "zstd": uproot.ZSTD,
    }
    resolved = (
        codes[compression](compression_level)
        if isinstance(compression, str)
        else compression
    )
    return {"compression": resolved}


# ---- module-level so a spawned ProcessPoolExecutor worker can pickle/import it ----------------------
def _write_partition(
    partition,
    resources,
    *,
    destination,
    prefix,
    columns,
    tree_name,
    compression,
    compression_level,
    bases,
    compiled,
    backend,
    source,
    source_name,
):
    """Read this blind partition's chunk via uproot (file opened once per worker), EVALUATE the
    recorded graph over it, and write the evaluated record's fields to its own part file, REPORTING
    the written path. ``N`` = the file's base + this partition's blind step, derived here from the
    partition alone (``graphed.write.blind_part_index`` — only the O(#files) base table travels
    with the task). An empty resolved step writes nothing.

    The read copies branches no longer: ``compiled`` (the driver-compiled IR) is evaluated per
    partition with the raw chunk bound to the source, so a DERIVED column (a field computed from an
    expression, absent from the source branches) is materialized and written — mirroring the
    read-side ``uproot._graphed.graphed_head`` pattern."""
    graphed = uproot.extras.graphed()

    tree = source.open_tree(partition, resources)
    resolved = partition.resolve(tree.num_entries)
    if resolved.entry_stop <= resolved.entry_start:
        return []  # fewer entries than steps: skip, never write an empty part file
    chunk = source.read_range(tree, columns, resolved.entry_start, resolved.entry_stop)
    evaluated: Any  # a backend array (awkward/numpy); evaluate_ir is typed list[object]
    (evaluated,) = graphed.evaluate_ir(compiled, backend, {source_name: chunk})
    # a record graph yields named fields (the derived columns); a bare (non-record) expression
    # yields a fieldless array with no branch name to write it under — fall back to the source
    # columns it reads
    out_rec = evaluated if evaluated.fields else chunk
    record = {name: out_rec[name] for name in out_rec.fields}
    idx = graphed.write.blind_part_index(partition, dict(bases))
    path = graphed.write.part_path(
        destination, idx, prefix=prefix or "part", suffix=".root"
    )
    with uproot.recreate(
        path, **_recreate_kwargs(compression, compression_level)
    ) as out:
        out[tree_name] = record
    return [path]


def _select_executor(executor):
    graphed_executors = uproot.extras.graphed_executors()

    if isinstance(executor, str):
        return {
            "process": graphed_executors.local.ProcessPoolExecutor,
            "thread": graphed_executors.local.ThreadExecutor,
        }[executor]
    return executor  # an executor class passed directly (instantiated by the caller)


def graphed_write(
    array,
    destination,
    *,
    steps_per_file=1,
    prefix=None,
    tree_name="tree",
    compute=True,
    executor="process",
    max_workers=None,
    compression="zlib",
    compression_level=1,
):
    """
    Args:
        array (``graphed.Array``): A ``uproot.graphed`` read (the deferred record of ``TTree``
            branches) to write back out, partition by partition.
        destination (path-like): Output **directory**; the part files are written inside it.
        steps_per_file (int): Split each input ``TTree`` into this many contiguous output
            partitions (blind — resolved by each worker; the driver opens no files).
        prefix (str or None): Part files are named by ``graphed.write.part_path``:
            ``f"{prefix or 'part'}-{N:05d}.root"``.
        tree_name (str): Name the data is assigned to in each part file (a dict assignment, so
            an ``RNTuple`` since Uproot 5.7). Default ``"tree"``.
        compute (bool): If ``True`` (default), execute the write task graph now via a
            ``graphed-executors`` executor and return the written paths (REPORTED BY THE WORKERS,
            in deterministic key order). If ``False``, return the ``graphed.core.Plan`` (the write
            task graph) **without writing** — run it later with an executor.
        executor (str or executor class): ``"process"`` (default, ``ProcessPoolExecutor``) or
            ``"thread"`` (``ThreadExecutor``); an executor **class** may also be passed — it is
            called as ``executor(max_workers=max_workers)``, so an already-built instance is not
            accepted.
        max_workers (int or None): Worker count for the executor.
        compression, compression_level: ROOT compression for the part files (as in ``dask_write``).

    Produces one ROOT file per partition, mirroring :doc:`uproot.writing._dask_write.dask_write`,
    as a specialization of the ``graphed.write`` partitioned-write base.
    """
    graphed = uproot.extras.graphed()

    if not isinstance(array, graphed.Array):
        raise TypeError("graphed_write expects a uproot.graphed Array")

    from uproot._graphed import _GraphedTTreeSource, _task_columns

    session = array.session
    uproot_sources = [
        (nid, s)
        for nid, s in session.sources().items()  # the public accessor — no internals
        if isinstance(s, _GraphedTTreeSource)
    ]
    if not uproot_sources:
        raise TypeError(
            "graphed_write: the array is not backed by a uproot.graphed source"
        )
    if len(uproot_sources) > 1:
        raise TypeError(
            f"graphed_write supports exactly one uproot.graphed source per array; "
            f"this array is backed by {len(uproot_sources)}"
        )
    nid, source = uproot_sources[0]

    # Each worker EVALUATES the recorded graph (below), which replays every node the graph accesses
    # — including field reads whose buffers the output never touches — so the read list is the
    # SYNTACTIC evaluation columns (as on the read side), not the finer buffer projection which
    # under-supplies evaluation; a form-mapped source declares its own instead, in TBranch names.
    # Compile once in the driver; evaluate per partition in the worker.
    columns = _task_columns(array, nid, source)
    compiled = graphed.compile_ir(session, array)
    os.makedirs(destination, exist_ok=True)

    # the graphed.write base: blind partitions (no driver file opens) + the O(#files) base table
    partitions = source.partitions(steps_per_file)
    bases = graphed.write.file_bases(list(source._file_tree), steps_per_file)

    process = functools.partial(
        _write_partition,
        destination=destination,
        prefix=prefix,
        columns=columns,
        tree_name=tree_name,
        compression=compression,
        compression_level=compression_level,
        bases=tuple(bases.items()),
        compiled=compiled,
        backend=session.backend,
        source=source,
        source_name=session.source_name(nid),
    )
    plan = graphed.write.write_plan(partitions, process)

    if not compute:
        return plan
    executor_cls = _select_executor(executor)
    return list(executor_cls(max_workers=max_workers).run(plan).value)
