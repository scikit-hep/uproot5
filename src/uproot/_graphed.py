# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""
This module defines :doc:`uproot._graphed.graphed`, which reads ``TTrees`` into a deferred
`graphed <https://github.com/graphed-org/graphed-project-mvp>`__ array — the analogue of
:doc:`uproot._dask.dask` for the ``graphed`` task-graph system (MVP).

``uproot.graphed(files, library="ak")`` returns a deferred ``graphed`` ``Array`` (recorded on the
``graphed-awkward`` backend). Construction reads only metadata (the ``TTree`` form); the event data is
read by ``uproot`` lazily when the expression is computed. Because ``graphed`` computes the *necessary
buffers* of the recorded graph (column projection), ``uproot.graphed(...).compute()`` reads only the
``TBranches`` the analysis actually touches — the same necessary-columns optimization ``uproot.dask``
provides, expressed through ``graphed``.

``graphed`` (and its backends) are imported lazily, so importing ``uproot`` does not require them.
"""
from __future__ import annotations

import uproot
import uproot._util
import uproot.interpretation.library
from uproot._dask import _get_ttree_form
from uproot._util import no_filter


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
        keys = obj.keys(
            recursive=recursive,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            full_paths=full_paths,
            ignore_duplicates=True,
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


def compute(array, *, project=True, on_fail="warn"):
    """Read the data and evaluate ``array`` (the ``graphed`` analogue of ``dask``'s ``.compute()``).

    With ``project=True`` (default), only the ``TBranches`` the recorded graph touches are read from
    the file(s) — ``graphed``'s column projection driving ``uproot``'s read."""
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
    return session.materialize(array)
