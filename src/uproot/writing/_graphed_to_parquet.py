# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""
This module defines :doc:`uproot.writing._graphed_to_parquet.graphed_to_parquet`: ROOT in,
parquet out, partition by partition — the first cross-format ``graphed`` pipeline.

Each (blind — no file is opened at planning time) ROOT partition becomes one write task: a worker
opens its file once, reads ONLY the branches the recorded graph needs (every branch the graph
SYNTACTICALLY accesses on the source — compiled-IR evaluation replays every node, so this is the
correct read list; the buffer projection, which tracks touched DATA, under-supplies it — and a
jagged branch read this way is also its own offsets carrier), evaluates the analysis
through the **compiled IR** (compiled once at the driver, never re-recorded per partition), and
writes one parquet part. The task graph is the ``graphed.write`` partitioned-write base
(specialized here with the ROOT reader + parquet codec), so workers report
their own part paths up the deterministic combine tree; ``compute=False`` returns the plan and
running it later produces the ``compute=True`` outputs bit-for-bit.

``graphed`` / ``graphed_awkward`` / ``graphed_exec_local`` / ``pyarrow`` are imported lazily, so
importing ``uproot`` does not require them.
"""
from __future__ import annotations

import functools
import os

import uproot


# ---- module-level so a spawned ProcessExecutor worker can pickle/import it ----------------------
def _convert_partition(
    partition, resources, *, destination, prefix, columns, column, compiled, source_name,
    behavior, bases,
):
    """Read this partition's branches, evaluate the compiled IR over the chunk, and write one
    parquet part. The part index is the file's base + this partition's blind step — derived here
    from the partition alone (only the O(#files) base table travels with the task)."""
    import awkward as ak
    from graphed import evaluate_ir
    from graphed_awkward import AwkwardBackend

    if isinstance(behavior, str):  # an importable "module:attr" reference (picklable; OpSpec-style)
        import importlib

        mod_name, _, attr = behavior.partition(":")
        behavior = getattr(importlib.import_module(mod_name), attr)
    tree = resources.open_once(partition.uri, uproot.open)[partition.tree]
    chunk = uproot.read_graphed_partition(partition, list(columns), tree=tree)
    (out,) = evaluate_ir(
        compiled, AwkwardBackend(behavior=behavior), {source_name: chunk}
    )
    from graphed import write as gwrite

    result = ak.Array(out)
    payload = result if result.fields else ak.Array({column: result})
    idx = gwrite.blind_part_index(partition, dict(bases))  # from the partition alone (R15.9)
    os.makedirs(destination, exist_ok=True)
    path = gwrite.part_path(destination, idx, prefix=prefix or "part", suffix=".parquet")
    ak.to_parquet(payload, path)
    return [path]


def graphed_to_parquet(
    array,
    destination,
    *,
    steps_per_file=1,
    prefix=None,
    compute=True,
    executor="process",
    max_workers=None,
    column="data",
    behavior=None,
):
    """
    Args:
        array (``graphed.Array``): A deferred expression recorded over ONE ``uproot.graphed``
            source (the analysis to evaluate and persist).
        destination (path-like): Output **directory**; parquet part files are written inside it.
        steps_per_file (int): Contiguous output partitions per input ``TTree`` (blind — resolved
            by each worker against the file's entry count; planning opens no files).
        prefix (str or None): Part files are named by ``graphed.write.part_path``:
            ``f"{prefix or 'part'}-{N:05d}.parquet"``.
        compute (bool): ``True`` (default) runs the write task graph now and returns the written
            paths (reported by the workers, in deterministic key order). ``False`` returns the
            ``graphed_core`` ``Plan`` without writing; run it later with any R7 executor.
        executor (str or executor): ``"process"`` (default) or ``"thread"``, or an executor
            class/instance.
        max_workers (int or None): Worker count.
        column (str): Field name for a non-record result (records keep their own fields).
        behavior: An awkward behavior dict — or, for process executors, an importable
            ``"module:attr"`` reference resolved in each worker (behavior dicts often contain
            lambdas, which do not pickle) — forwarded to each worker's evaluation backend (pair
            with ``uproot.graphed(..., behavior=...)``).

    Evaluates the recorded analysis per partition through the compiled IR and writes one parquet
    part per partition — ROOT in, parquet out.
    """
    from graphed import compile_ir
    from graphed import write as gwrite
    from graphed_core import Partition

    from uproot._graphed import _GraphedTTreeSource, _evaluation_columns
    from uproot.writing._graphed_write import _select_executor

    session = array.session
    uproot_sources = [
        (nid, s)
        for nid, s in session.sources().items()
        if isinstance(s, _GraphedTTreeSource)
    ]
    if len(uproot_sources) != 1:
        raise TypeError(
            f"graphed_to_parquet supports exactly one uproot.graphed source per array; "
            f"this array is backed by {len(uproot_sources)}"
        )
    (nid, source) = uproot_sources[0]
    source_name = session.source_name(nid)
    columns = _evaluation_columns(array, nid, source._common_keys)
    compiled = compile_ir(session, array)

    # blind partitions: one per (file x step), no file opened at planning time (R7.9); the
    # base table comes from the graphed.write base, keyed (uri, tree) for container formats
    partitions = tuple(
        Partition.blind(file_path, object_path, s, steps_per_file)
        for (file_path, object_path) in source._file_tree
        for s in range(steps_per_file)
    )
    bases = gwrite.file_bases(list(source._file_tree), steps_per_file)

    process = functools.partial(
        _convert_partition,
        destination=destination,
        prefix=prefix,
        columns=columns,
        column=column,
        compiled=compiled,
        source_name=source_name,
        behavior=behavior,
        bases=tuple(bases.items()),
    )
    plan = gwrite.write_plan(partitions, process)

    if not compute:
        return plan
    executor_cls = _select_executor(executor)
    return list(executor_cls(max_workers=max_workers).run(plan).value)
