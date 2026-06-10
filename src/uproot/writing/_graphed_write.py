# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""
This module defines :doc:`uproot.writing._graphed_write.graphed_write`, the ``graphed`` analogue of
:doc:`uproot.writing._dask_write.dask_write`.

Like ``dask_write``, it produces **one output ROOT file per input partition**. Each partition becomes a
side-effecting write *task* (it reads its chunk and writes ``{prefix-}part{N}.root``, returning
nothing); the tasks form a ``graphed_core.Plan``. With ``compute=False`` the Plan (the write task
graph) is returned without running; with ``compute=True`` (the default) it is executed through a
``graphed-exec-local`` executor — the ``ProcessExecutor`` by default (``executor="thread"`` for the
thread pool) — exactly the way a deferred-array ``.compute()`` hides a thread/process executor.

``graphed`` / ``graphed_core`` / ``graphed_exec_local`` are imported lazily, so importing ``uproot``
does not require them.

NOTE (P3.6 review, 2026-06-10): this module deliberately does NOT sit on the ``graphed.write``
partitioned-write base that ``graphed_to_parquet`` specializes. Two of its behaviors are FROZEN
pins predating the base: write tasks return ``None`` (the suite asserts ``result.value is None``),
where the base's tasks report their part paths; and ``_step_ranges`` SKIPS empty ranges when a
file has fewer entries than ``steps_per_file``, where the base's ``step_of`` math does not.
Aligning them would amend frozen tests — a recorded freeze bump to do deliberately, not as part
of a refactor.
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


def _step_ranges(n_entries, steps_per_file):
    """The non-empty contiguous step ranges of one file — the SAME rule the driver used, so a
    worker can recompute its own part index from its partition alone (no per-partition path table
    is pickled into every task)."""
    ranges = []
    for i in range(steps_per_file):
        start = (i * n_entries) // steps_per_file
        stop = ((i + 1) * n_entries) // steps_per_file
        if stop > start:
            ranges.append((start, stop))
    return ranges


# ---- module-level so a spawned ProcessExecutor worker can pickle/import them --------------------
def _write_partition(
    destination, prefix, columns, tree_name, compression, compression_level,
    steps_per_file, file_bases, partition, resources,
):
    """Read this partition's chunk via uproot (file opened once per worker) and write it to its own
    ``part{N}.root``. ``N`` = the file's base index + this chunk's position among the file's step
    ranges, recomputed here from the open file. Returns ``None`` — the write is the task's only
    effect."""
    tree = resources.open_once(partition.uri, uproot.open)[partition.tree]
    chunk = uproot.read_graphed_partition(partition, columns, tree=tree)
    record = {name: chunk[name] for name in chunk.fields}
    ranges = _step_ranges(tree.num_entries, steps_per_file)
    idx = file_bases[(partition.uri, partition.tree)] + ranges.index(
        (partition.entry_start, partition.entry_stop)
    )
    name = f"{prefix}-part{idx}.root" if prefix else f"part{idx}.root"
    path = os.path.join(destination, name)
    with uproot.recreate(path, **_recreate_kwargs(compression, compression_level)) as out:
        out[tree_name] = record
    return None


def _combine_none(a, b):
    return None


def _empty_none():
    return None


def _select_executor(executor):
    from graphed_exec_local import ProcessExecutor, ThreadExecutor

    if isinstance(executor, str):
        return {"process": ProcessExecutor, "thread": ThreadExecutor}[executor]
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
        steps_per_file (int): Split each input ``TTree`` into this many contiguous output partitions.
        prefix (str or None): If given, part files are ``f"{prefix}-part{N}.root"``; otherwise
            ``f"part{N}.root"``.
        tree_name (str): Name of the ``TTree`` written into each part file. Default ``"tree"``.
        compute (bool): If ``True`` (default), execute the write task graph now via a
            ``graphed-exec-local`` executor and return the list of written paths. If ``False``, return
            the ``graphed_core.Plan`` (the set of write tasks) **without writing** — run it later with
            an executor (or call again with ``compute=True``).
        executor (str or executor): ``"process"`` (default, ``ProcessExecutor``) or ``"thread"``
            (``ThreadExecutor``); an executor class/instance may also be passed.
        max_workers (int or None): Worker count for the executor.
        compression, compression_level: ROOT compression for the part files (as in ``dask_write``).

    Produces one ROOT file per partition, mirroring :doc:`uproot.writing._dask_write.dask_write`.
    """
    from graphed_core import Partition, Plan, Task

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

    # one partition (one output file) per (file x step); workers recompute their own part index
    # from (file base + step ranges), so only the O(#files) base table travels with the task
    tasks: list = []
    ordered_paths: list = []
    file_bases: dict = {}
    idx = 0
    for file_path, object_path in source._file_tree:
        n_entries = uproot.open(file_path)[object_path].num_entries
        file_bases[(file_path, object_path)] = idx
        for start, stop in _step_ranges(n_entries, steps_per_file):
            part = Partition(file_path, object_path, int(start), int(stop))
            name = f"{prefix}-part{idx}.root" if prefix else f"part{idx}.root"
            ordered_paths.append(os.path.join(destination, name))
            tasks.append(Task(idx, part))
            idx += 1

    process = functools.partial(
        _write_partition, destination, prefix, columns, tree_name, compression,
        compression_level, steps_per_file, file_bases,
    )
    plan = Plan(process=process, combine=_combine_none, empty=_empty_none, tasks=tasks)

    if not compute:
        return plan

    executor_cls = _select_executor(executor)
    executor_cls(max_workers=max_workers).run(plan)
    return ordered_paths
