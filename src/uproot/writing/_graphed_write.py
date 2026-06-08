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


# ---- module-level so a spawned ProcessExecutor worker can pickle/import them --------------------
def _write_partition(out_paths, columns, tree_name, compression, compression_level, partition, resources):
    """Read this partition's chunk via uproot (file opened once per worker) and write it to its own
    ``part{N}.root``. Returns ``None`` — the write is the task's only effect."""
    file = resources.open_once(partition.uri, uproot.open)
    chunk = file[partition.tree].arrays(
        list(columns),
        entry_start=partition.entry_start,
        entry_stop=partition.entry_stop,
        library="ak",
    )
    record = {name: chunk[name] for name in chunk.fields}
    path = out_paths[(partition.uri, partition.tree, partition.entry_start, partition.entry_stop)]
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

    from uproot._graphed import _GraphedTTreeSource

    session = array.session
    source = next(
        (
            session._sources[nid]
            for nid in session.source_ids()
            if isinstance(session._sources.get(nid), _GraphedTTreeSource)
        ),
        None,
    )
    if source is None:
        raise TypeError("graphed_write: the array is not backed by a uproot.graphed source")

    columns = tuple(source._common_keys)
    os.makedirs(destination, exist_ok=True)

    # one partition (one output file) per (file x step)
    tasks: list = []
    out_paths: dict = {}
    ordered_paths: list = []
    idx = 0
    for file_path, object_path in source._file_tree:
        n_entries = uproot.open(file_path)[object_path].num_entries
        for i in range(steps_per_file):
            start = (i * n_entries) // steps_per_file
            stop = ((i + 1) * n_entries) // steps_per_file
            if stop <= start:
                continue
            part = Partition(file_path, object_path, int(start), int(stop))
            name = f"{prefix}-part{idx}.root" if prefix else f"part{idx}.root"
            path = os.path.join(destination, name)
            out_paths[(file_path, object_path, int(start), int(stop))] = path
            ordered_paths.append(path)
            tasks.append(Task(idx, part))
            idx += 1

    process = functools.partial(
        _write_partition, out_paths, columns, tree_name, compression, compression_level
    )
    plan = Plan(process=process, combine=_combine_none, empty=_empty_none, tasks=tasks)

    if not compute:
        return plan

    executor_cls = _select_executor(executor)
    executor_cls(max_workers=max_workers).run(plan)
    return ordered_paths
