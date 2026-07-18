# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""
This module defines :doc:`uproot.writing._graphed_write.graphed_write`, the ``graphed`` analogue of
:doc:`uproot.writing._dask_write.dask_write` — and a SPECIALIZATION of the ``graphed.write``
partitioned-write base (P3.6 revision; freeze-UPROOT-2, user-authorized frozen amendments).

Like ``dask_write``, it produces **one output ROOT file per input partition**. The partitions are
BLIND (the driver opens no files — each worker resolves its entry range against its own file,
R7.9); each write task writes ``{prefix}-{N:05d}.root`` and **reports its part path** up the
plan's deterministic combine tree (the base's contract — previously the tasks returned nothing);
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

import uproot


def _is_graphed_array(obj):
    try:
        from graphed import Array
    except ImportError:
        return False
    return isinstance(obj, Array)


def _recreate_kwargs(compression, compression_level):
    if compression is None:
        return {}
    codes = {"zlib": uproot.ZLIB, "lzma": uproot.LZMA, "lz4": uproot.LZ4, "zstd": uproot.ZSTD}
    resolved = codes[compression](compression_level) if isinstance(compression, str) else compression
    return {"compression": resolved}


# ---- module-level so a spawned ProcessPoolExecutor worker can pickle/import it ----------------------
def _write_partition(
    partition, resources, *, destination, prefix, columns, tree_name, compression,
    compression_level, bases,
):
    """Read this blind partition's chunk via uproot (file opened once per worker) and write it to
    its own part file, REPORTING the written path. ``N`` = the file's base + this partition's
    blind step, derived here from the partition alone (``graphed.write.blind_part_index`` — only
    the O(#files) base table travels with the task). An empty resolved step writes nothing."""
    from graphed import write as gwrite

    tree = resources.open_once(partition.uri, uproot.open)[partition.tree]
    resolved = partition.resolve(tree.num_entries)
    if resolved.entry_stop <= resolved.entry_start:
        return []  # fewer entries than steps: skip, never write an empty part file
    chunk = uproot.read_graphed_partition(partition, columns, tree=tree)
    record = {name: chunk[name] for name in chunk.fields}
    idx = gwrite.blind_part_index(partition, dict(bases))
    path = gwrite.part_path(destination, idx, prefix=prefix or "part", suffix=".root")
    with uproot.recreate(path, **_recreate_kwargs(compression, compression_level)) as out:
        out[tree_name] = record
    return [path]


def _select_executor(executor):
    from graphed_executors.local import ProcessPoolExecutor, ThreadExecutor

    if isinstance(executor, str):
        return {"process": ProcessPoolExecutor, "thread": ThreadExecutor}[executor]
    return executor  # an executor class/instance passed directly


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
        tree_name (str): Name of the ``TTree`` written into each part file. Default ``"tree"``.
        compute (bool): If ``True`` (default), execute the write task graph now via a
            ``graphed-executors`` executor and return the written paths (REPORTED BY THE WORKERS,
            in deterministic key order). If ``False``, return the ``graphed.core.Plan`` (the write
            task graph) **without writing** — run it later with an executor.
        executor (str or executor): ``"process"`` (default, ``ProcessPoolExecutor``) or ``"thread"``
            (``ThreadExecutor``); an executor class/instance may also be passed.
        max_workers (int or None): Worker count for the executor.
        compression, compression_level: ROOT compression for the part files (as in ``dask_write``).

    Produces one ROOT file per partition, mirroring :doc:`uproot.writing._dask_write.dask_write`,
    as a specialization of the ``graphed.write`` partitioned-write base.
    """
    from graphed import write as gwrite

    if not _is_graphed_array(array):
        raise TypeError("graphed_write expects a uproot.graphed Array")

    from uproot._graphed import _GraphedTTreeSource, necessary_columns

    session = array.session
    uproot_sources = [
        (nid, s)
        for nid, s in session.sources().items()  # the public accessor (graphed M10) — no internals
        if isinstance(s, _GraphedTTreeSource)
    ]
    if not uproot_sources:
        raise TypeError("graphed_write: the array is not backed by a uproot.graphed source")
    if len(uproot_sources) > 1:
        raise TypeError(
            f"graphed_write supports exactly one uproot.graphed source per array; "
            f"this array is backed by {len(uproot_sources)}"
        )
    (nid, source) = uproot_sources[0]

    # write only the branches the recorded array actually carries (its necessary columns); a bare
    # source read projects to every common key, reproducing the old behavior
    projected = necessary_columns(array).get(session.source_name(nid), frozenset())
    columns = tuple(k for k in source._common_keys if k in projected) or tuple(source._common_keys)
    os.makedirs(destination, exist_ok=True)

    # the graphed.write base: blind partitions (no driver file opens) + the O(#files) base table
    partitions = source.partitions(steps_per_file)
    bases = gwrite.file_bases(list(source._file_tree), steps_per_file)

    process = functools.partial(
        _write_partition,
        destination=destination,
        prefix=prefix,
        columns=columns,
        tree_name=tree_name,
        compression=compression,
        compression_level=compression_level,
        bases=tuple(bases.items()),
    )
    plan = gwrite.write_plan(partitions, process)

    if not compute:
        return plan
    executor_cls = _select_executor(executor)
    return list(executor_cls(max_workers=max_workers).run(plan).value)
