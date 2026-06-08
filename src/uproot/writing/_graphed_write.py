# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""
This module defines :doc:`uproot.writing._graphed_write.graphed_write`, the ``graphed`` analogue of
:doc:`uproot.writing._dask_write.dask_write`: it computes a deferred ``graphed`` array and writes the
result to a ``TTree`` in a ROOT file via ``uproot``.

``graphed`` is imported lazily, so importing ``uproot`` does not require it.
"""
from __future__ import annotations

import uproot


def _is_graphed_array(obj):
    try:
        from graphed import Array
    except ImportError:
        return False
    return isinstance(obj, Array)


def graphed_write(
    array,
    destination,
    *,
    tree_name="tree",
    compute=True,
    project=True,
    on_fail="warn",
    title="",
    field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner,
    initial_basket_capacity=10,
    counter_name=lambda counted: "n" + counted,
    resize_factor=10.0,
    compression="zlib",
    compression_level=1,
):
    """
    Args:
        array (``graphed.Array`` or ``awkward.Array``): The deferred ``graphed`` array (or an already
            materialized awkward array) to write.
        destination (path-like): Output ROOT file path.
        tree_name (str): Name of the ``TTree`` to write. Default ``"tree"``.
        compute (bool): If ``True`` (default), compute the ``graphed`` array and write it now; if
            ``False``, return the computed awkward array without writing.
        project (bool): If ``True`` (default), read only the ``TBranches`` the recorded graph touches
            when computing a ``graphed`` array (``graphed``'s column projection).
        on_fail (str): The projection on-fail policy for opaque ops (``"raise"``/``"warn"``/``"pass"``).
        title, field_name, initial_basket_capacity, counter_name, resize_factor, compression,
        compression_level: Forwarded to ``uproot``'s ROOT writing (as in
            :doc:`uproot.writing._dask_write.dask_write`).

    Writes a (computed) ``graphed`` array to a ``TTree`` in ``destination``. This is the ``graphed``
    analogue of :doc:`uproot.writing._dask_write.dask_write`.
    """
    if _is_graphed_array(array):
        from uproot._graphed import _GraphedTTreeSource

        session = array.session
        if project:
            from graphed_awkward.projection import project as _project

            proj = _project(array, on_fail=on_fail)
            for nid in session.source_ids():
                data = session._sources.get(nid)
                if isinstance(data, _GraphedTTreeSource):
                    cols = proj.columns_for(session.source_name(nid))
                    if cols:
                        data.columns = sorted(cols)
        materialized = session.materialize(array)
    else:
        materialized = array

    if not compute:
        return materialized

    import awkward

    materialized = awkward.Array(materialized)
    record = {name: materialized[name] for name in materialized.fields}

    recreate_kwargs = {}
    if compression is not None:
        codes = {"zlib": uproot.ZLIB, "lzma": uproot.LZMA, "lz4": uproot.LZ4, "zstd": uproot.ZSTD}
        recreate_kwargs["compression"] = (
            codes[compression](compression_level) if isinstance(compression, str) else compression
        )

    with uproot.recreate(destination, **recreate_kwargs) as out:
        out[tree_name] = record if record else materialized

    return destination
