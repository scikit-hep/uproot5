# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines behaviors for :doc:`uproot.behaviors.RNTuple.RNTuple` and
:doc:`uproot.behaviors.RNTuple.HasFields` (both ``RField`` and
``RNTuple``).

Most of the functionality of RNTuple-reading is implemented here.

See :doc:`uproot.models.RNTuple` for deserialization of the ``RNTuple``
objects themselves.
"""
from __future__ import annotations

import sys
import warnings
from collections.abc import Mapping

import numpy

import uproot
import uproot.interpretation.grouped
import uproot.language.python
import uproot.source.chunk
from uproot._util import no_filter, unset


def iterate(
    files,
    expressions=None,  # TODO: Not implemented yet
    cut=None,  # TODO: Not implemented yet
    *,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_field=no_filter,
    aliases=None,  # TODO: Not implemented yet
    language=uproot.language.python.python_language,  # TODO: Not implemented yet
    step_size="100 MB",
    decompression_executor=None,  # TODO: Not implemented yet
    library="ak",  # TODO: Not implemented yet
    ak_add_doc=False,  # TODO: Not implemented yet
    how=None,
    report=False,  # TODO: Not implemented yet
    allow_missing=False,  # TODO: Not implemented yet
    # For compatibility reasons we also accepts kwargs meant for TTrees
    filter_branch=unset,
    interpretation_executor=unset,
    custom_classes=unset,
    **options,
):
    """
    Args:
        files: See below.
        expressions (None, str, or list of str): Names of ``RFields`` or
            aliases to convert to arrays or mathematical expressions of them.
            Uses the ``language`` to evaluate. If None, all ``RFields``
            selected by the filters are included. (Not implemented yet.)
        cut (None or str): If not None, this expression filters all of the
            ``expressions``. (Not implemented yet.)
        filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by name.
        filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by type.
        filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool, or None): A
            filter to select ``RFields`` using the full
            :doc:`uproot.models.RNTuple.RField` object. If the function
            returns False or None, the ``RField`` is excluded; if the function
            returns True, it is included.
        aliases (None or dict of str \u2192 str): Mathematical expressions that
            can be used in ``expressions`` or other aliases.
            Uses the ``language`` engine to evaluate. (Not implemented yet.)
        language (:doc:`uproot.language.Language`): Language used to interpret
            the ``expressions`` and ``aliases``. (Not implemented yet.)
        step_size (int or str): If an integer, the maximum number of entries to
            include in each iteration step; if a string, the maximum memory size
            to include. The string must be a number followed by a memory unit,
            such as "100 MB".
        decompression_executor (None or Executor with a ``submit`` method): The
            executor that is used to decompress ``RPages``; if None, a
            :doc:`uproot.source.futures.TrivialExecutor` is created. (Not implemented yet.)
        library (str or :doc:`uproot.interpretation.library.Library`): The library
            that is used to represent arrays. Options are ``"np"`` for NumPy,
            ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas. (Not implemented yet.)
        ak_add_doc (bool | dict ): If True and ``library="ak"``, add the RField ``name``
            to the Awkward ``__doc__`` parameter of the array.
            if dict = {key:value} and ``library="ak"``, add the RField ``value`` to the
            Awkward ``key`` parameter of the array.
        how (None, str, or container type): Library-dependent instructions
            for grouping. The only recognized container types are ``tuple``,
            ``list``, and ``dict``. Note that the container *type itself*
            must be passed as ``how``, not an instance of that type (i.e.
            ``how=tuple``, not ``how=()``).
        report (bool): If True, this generator yields
            (arrays, :doc:`uproot.behaviors.TBranch.Report`) pairs; if False,
            it only yields arrays. The report has data about the ``TFile``,
            ``TTree``, and global and local entry ranges.
        allow_missing (bool): If True, skip over any files that do not contain
            the specified ``RNTuple``.
        filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
            for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
            and will be removed in a future version.
        interpretation_executor (None): This argument is not used and is only included for now
            for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
            and will be removed in a future version.
        custom_classes (None): This argument is not used and is only included for now
            for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
            and will be removed in a future version.
        options: See below.

    Iterates through contiguous chunks of entries from a set of files.

    For example:

    .. code-block:: python

        >>> for array in uproot.iterate("files*.root:ntuple", filter_names=["x", "y"], step_size=100):
        ...     # each of the following have 100 entries
        ...     array["x"], array["y"]

    Allowed types for the ``files`` parameter:

    * str/bytes: relative or absolute filesystem path or URL, without any colons
      other than Windows drive letter or URL schema.
      Examples: ``"rel/file.root"``, ``"C:\\abs\\file.root"``, ``"http://where/what.root"``
    * str/bytes: same with an object-within-ROOT path, separated by a colon.
      Example: ``"rel/file.root:tdirectory/rntuple"``
    * pathlib.Path: always interpreted as a filesystem path or URL only (no
      object-within-ROOT path), regardless of whether there are any colons.
      Examples: ``Path("rel:/file.root")``, ``Path("/abs/path:stuff.root")``
    * glob syntax in str/bytes and pathlib.Path.
      Examples: ``Path("rel/*.root")``, ``"/abs/*.root:tdirectory/rntuple"``
    * dict: keys are filesystem paths, values are objects-within-ROOT paths.
      Example: ``{{"/data_v1/*.root": "rntuple_v1", "/data_v2/*.root": "rntuple_v2"}}``
    * already-open RNTuple objects.
    * iterables of the above.

    Options (type; default): (Not implemented yet.)

    * handler (:doc:`uproot.source.chunk.Source` class; None)
    * timeout (float for HTTP, int for XRootD; 30)
    * max_num_elements (None or int; None)
    * num_workers (int; 1)
    * use_threads (bool; False on the emscripten platform (i.e. in a web browser), else True)
    * num_fallback_workers (int; 10)
    * begin_chunk_size (memory_size; 403, the smallest a ROOT file can be)
    * minimal_ttree_metadata (bool; True)

    See also :ref:`uproot.behaviors.RNTuple.HasFields.iterate` to iterate
    within a single file.

    Other file entry points:

    * :doc:`uproot.reading.open`: opens one file to read any of its objects.
    * :doc:`uproot.behaviors.RNTuple.iterate` (this function): iterates through
      chunks of contiguous entries in ``RNTuples``.
    * :doc:`uproot.behaviors.RNTuple.concatenate`: returns a single concatenated
      array from ``RNTuples``.
    * :doc:`uproot._dask.dask`: returns an unevaluated Dask array from ``RNTuples``.
    """
    files = uproot._util.regularize_files(files, steps_allowed=False, **options)
    library = uproot.interpretation.library._regularize_library(library)

    for file_path, object_path in files:
        hasfields = uproot._util.regularize_object_path(
            file_path, object_path, None, allow_missing, options
        )

        if hasfields is not None:
            with hasfields:
                try:
                    yield from hasfields.iterate(
                        expressions=expressions,
                        cut=cut,
                        filter_name=filter_name,
                        filter_typename=filter_typename,
                        filter_field=filter_field,
                        aliases=aliases,
                        language=language,
                        step_size=step_size,
                        decompression_executor=decompression_executor,
                        library=library,
                        ak_add_doc=ak_add_doc,
                        how=how,
                        report=report,
                        filter_branch=filter_branch,
                        interpretation_executor=interpretation_executor,
                    )

                except uproot.exceptions.KeyInFileError:
                    if allow_missing:
                        continue
                    else:
                        raise


def concatenate(
    files,
    expressions=None,  # TODO: Not implemented yet
    cut=None,  # TODO: Not implemented yet
    *,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_field=no_filter,
    aliases=None,  # TODO: Not implemented yet
    language=uproot.language.python.python_language,  # TODO: Not implemented yet
    entry_start=None,
    entry_stop=None,
    decompression_executor=None,  # TODO: Not implemented yet
    library="ak",  # TODO: Not implemented yet
    ak_add_doc=False,  # TODO: Not implemented yet
    how=None,
    allow_missing=False,
    # For compatibility reasons we also accepts kwargs meant for TTrees
    filter_branch=unset,
    interpretation_executor=unset,
    custom_classes=unset,
    **options,
):
    """
    Args:
        files: See below.
        expressions (None, str, or list of str): Names of ``RFields`` or
            aliases to convert to arrays or mathematical expressions of them.
            Uses the ``language`` to evaluate. If None, all ``RFields``
            selected by the filters are included. (Not implemented yet.)
        cut (None or str): If not None, this expression filters all of the
            ``expressions``. (Not implemented yet.)
        filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by name.
        filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by type.
        filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool, or None): A
            filter to select ``RFields`` using the full
            :doc:`uproot.models.RNTuple.RField` object. If the function
            returns False or None, the ``RField`` is excluded; if the function
            returns True, it is included.
        aliases (None or dict of str \u2192 str): Mathematical expressions that
            can be used in ``expressions`` or other aliases.
            Uses the ``language`` engine to evaluate. (Not implemented yet.)
        language (:doc:`uproot.language.Language`): Language used to interpret
            the ``expressions`` and ``aliases``. (Not implemented yet.)
        entry_start (None or int): The first entry to include. If None, start
            at zero. If negative, count from the end, like a Python slice.
        entry_stop (None or int): The first entry to exclude (i.e. one greater
            than the last entry to include). If None, stop at
            :ref:`uproot.behaviors.RNTuple.HasFields.num_entries`. If negative,
            count from the end, like a Python slice.
        decompression_executor (None or Executor with a ``submit`` method): The
            executor that is used to decompress ``RPages``; if None, a
            :doc:`uproot.source.futures.TrivialExecutor` is created. (Not implemented yet.)
        library (str or :doc:`uproot.interpretation.library.Library`): The library
            that is used to represent arrays. Options are ``"np"`` for NumPy,
            ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas. (Not implemented yet.)
        ak_add_doc (bool | dict ): If True and ``library="ak"``, add the RField ``name``
            to the Awkward ``__doc__`` parameter of the array.
            if dict = {key:value} and ``library="ak"``, add the RField ``value`` to the
            Awkward ``key`` parameter of the array.
        how (None, str, or container type): Library-dependent instructions
            for grouping. The only recognized container types are ``tuple``,
            ``list``, and ``dict``. Note that the container *type itself*
            must be passed as ``how``, not an instance of that type (i.e.
            ``how=tuple``, not ``how=()``).
        report (bool): If True, this generator yields
            (arrays, :doc:`uproot.behaviors.TBranch.Report`) pairs; if False,
            it only yields arrays. The report has data about the ``TFile``,
            ``TTree``, and global and local entry ranges.
        allow_missing (bool): If True, skip over any files that do not contain
            the specified ``RNTuple``.
        filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
            for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
            and will be removed in a future version.
        interpretation_executor (None): This argument is not used and is only included for now
            for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
            and will be removed in a future version.
        custom_classes (None): This argument is not used and is only included for now
            for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
            and will be removed in a future version.
        options: See below.

    Returns an array with data from a set of files concatenated into one.

    For example:

    .. code-block:: python

        >>> array = uproot.concatenate("files*.root:ntuple", filter_field=["x", "y"])

    Depending on the number of files, the number of selected ``RFields``, and
    the size of your computer's memory, this function might not have enough
    memory to run.

    Allowed types for the ``files`` parameter:

    * str/bytes: relative or absolute filesystem path or URL, without any colons
      other than Windows drive letter or URL schema.
      Examples: ``"rel/file.root"``, ``"C:\\abs\\file.root"``, ``"http://where/what.root"``
    * str/bytes: same with an object-within-ROOT path, separated by a colon.
      Example: ``"rel/file.root:tdirectory/rntuple"``
    * pathlib.Path: always interpreted as a filesystem path or URL only (no
      object-within-ROOT path), regardless of whether there are any colons.
      Examples: ``Path("rel:/file.root")``, ``Path("/abs/path:stuff.root")``
    * glob syntax in str/bytes and pathlib.Path.
      Examples: ``Path("rel/*.root")``, ``"/abs/*.root:tdirectory/rntuple"``
    * dict: keys are filesystem paths, values are objects-within-ROOT paths.
      Example: ``{{"/data_v1/*.root": "rntuple_v1", "/data_v2/*.root": "rntuple_v2"}}``
    * already-open RNTuple objects.
    * iterables of the above.

    Options (type; default): (Not implemented yet.)

    * handler (:doc:`uproot.source.chunk.Source` class; None)
    * timeout (float for HTTP, int for XRootD; 30)
    * max_num_elements (None or int; None)
    * num_workers (int; 1)
    * use_threads (bool; False on the emscripten platform (i.e. in a web browser), else True)
    * num_fallback_workers (int; 10)
    * begin_chunk_size (memory_size; 403, the smallest a ROOT file can be)
    * minimal_ttree_metadata (bool; True)

    Other file entry points:

    * :doc:`uproot.reading.open`: opens one file to read any of its objects.
    * :doc:`uproot.behaviors.RNTuple.iterate`: iterates through chunks of
      contiguous entries in ``RNTuples``.
    * :doc:`uproot.behaviors.RNTuple.concatenate` (this function): returns a
      single concatenated array from ``RNTuples``.
    * :doc:`uproot._dask.dask`: returns an unevaluated Dask array from ``RNTuples``.
    """
    files = uproot._util.regularize_files(files, steps_allowed=False, **options)
    library = uproot.interpretation.library._regularize_library(library)

    all_arrays = []
    global_start = 0
    global_stop = 0

    all_hasfields = []
    for file_path, object_path in files:
        _hasfields = uproot._util.regularize_object_path(
            file_path, object_path, None, allow_missing, options
        )
        if _hasfields is not None:
            all_hasfields.append(_hasfields)

    total_num_entries = sum(hasfields.num_entries for hasfields in all_hasfields)
    entry_start, entry_stop = uproot.behaviors.TBranch._regularize_entries_start_stop(
        total_num_entries, entry_start, entry_stop
    )
    for hasfields in all_hasfields:
        with hasfields:
            nentries = hasfields.num_entries
            global_stop += nentries

            if (
                global_start <= entry_start < global_stop
                or global_start < entry_stop <= global_stop
            ):
                # overlap, read only the overlapping entries
                local_entry_start = max(
                    0, entry_start - global_start
                )  # need to clip to 0
                local_entry_stop = entry_stop - global_start  # overflows are fine
            elif entry_start >= global_stop or entry_stop <= global_start:  # no overlap
                # outside of this file's range -> skip
                global_start = global_stop
                continue
            else:
                # read all entries
                local_entry_start = 0
                local_entry_stop = nentries

            try:
                arrays = hasfields.arrays(
                    expressions=expressions,
                    cut=cut,
                    filter_name=filter_name,
                    filter_typename=filter_typename,
                    filter_field=filter_field,
                    aliases=aliases,
                    language=language,
                    entry_start=local_entry_start,
                    entry_stop=local_entry_stop,
                    decompression_executor=decompression_executor,
                    array_cache=None,
                    library=library,
                    ak_add_doc=ak_add_doc,
                    how=how,
                    filter_branch=filter_branch,
                    interpretation_executor=interpretation_executor,
                )
                arrays = library.global_index(arrays, global_start)
            except uproot.exceptions.KeyInFileError:
                if allow_missing:
                    continue
                else:
                    raise

            all_arrays.append(arrays)
            global_start = global_stop

    return library.concatenate(all_arrays)


class HasFields(Mapping):
    """
    Abstract class of behaviors for anything that "has fields," namely
    :doc:`uproot.models.RNTuple.RNTuple` and
    :doc:`uproot.models.RNTuple.RField`, which mostly consist of array-reading
    methods.

    A :doc:`uproot.behaviors.RNTuple.HasFields` is a Python ``Mapping``, which
    uses square bracket syntax to extract subfields:

    .. code-block:: python

        my_rntuple["field"]
        my_rntuple["field"]["subfield"]
        my_rntuple["field.subfield"]
        my_rntuple["field.subfield.subsubfield"]
        my_rntuple["field/subfield/subsubfield"]
        my_rntuple["field\\subfield\\subsubfield"]
    """

    @property
    def ntuple(self):
        """
        The :doc:`uproot.models.RNTuple.RNTuple` that this
        :doc:`uproot.models.RNTuple.HasFields` is part of.
        """
        return self._ntuple

    @property
    def num_entries(self):
        """
        The number of entries in the ``RNTuple``.
        """
        if isinstance(self, uproot.behaviors.RNTuple.RNTuple):
            if self._num_entries is None:
                self._num_entries = sum(x.num_entries for x in self.cluster_summaries)
            return self._num_entries
        return self.ntuple.num_entries

    @property
    def fields(self):
        """
        The list of :doc:`uproot.models.RNTuple.RField` directly under
        this :doc:`uproot.models.RNTuple.RNTuple` or
        :doc:`uproot.models.RNTuple.RField` (i.e. not recursive).
        """
        if self._fields is None:
            rntuple = self.ntuple
            if isinstance(self, uproot.behaviors.RNTuple.RNTuple):
                fields = [
                    rntuple.all_fields[i]
                    for i, f in enumerate(rntuple.field_records)
                    if f.parent_field_id == i
                ]
            else:
                fields = [
                    rntuple.all_fields[i]
                    for i, f in enumerate(rntuple.field_records)
                    if f.parent_field_id == self._fid and f.parent_field_id != i
                ]
            self._fields = fields
        return self._fields

    @property
    def path(self):
        """
        The full path of the field in the :doc:`uproot.models.RNTuple.RNTuple`. When it is
        the ``RNTuple`` itself, this is ``"."``.
        """
        if isinstance(self, uproot.behaviors.RNTuple.RNTuple):
            return "."
        if self._path is None:
            path = self.name
            parent = self.parent
            field = self
            while not isinstance(parent, uproot.behaviors.RNTuple.RNTuple):
                path = f"{parent.name}.{path}"
                field = parent
                parent = field.parent
            self._path = path
        return self._path

    def to_akform(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns the an Awkward Form with the structure of the data in the ``RNTuple`` or ``RField``.
        """
        ak = uproot.extras.awkward()

        keys = self.keys(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            filter_branch=filter_branch,
        )
        rntuple = self.ntuple

        top_names = []
        record_list = []
        if self is rntuple:
            for field in self.fields:
                # the field needs to be in the keys or be a parent of a field in the keys
                if any(key.startswith(field.name) for key in keys):
                    top_names.append(field.name)
                    record_list.append(rntuple.field_form(field.field_id, keys))
        else:
            # Always use the full path for keys
            # Also include the field itself
            keys = [self.path] + [f"{self.path}.{k}" for k in keys]
            # The field needs to be in the keys or be a parent of a field in the keys
            if any(key.startswith(self.path) for key in keys):
                top_names.append(self.name)
                record_list.append(rntuple.field_form(self.field_id, keys))

        form = ak.forms.RecordForm(record_list, top_names, form_key="toplevel")
        return form

    def arrays(
        self,
        expressions=None,  # TODO: Not implemented yet
        cut=None,  # TODO: Not implemented yet
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        aliases=None,  # TODO: Not implemented yet
        language=uproot.language.python.python_language,  # TODO: Not implemented yet
        entry_start=None,
        entry_stop=None,
        decompression_executor=None,  # TODO: Not implemented yet
        array_cache="inherit",  # TODO: Not implemented yet
        library="ak",  # TODO: Not implemented yet
        ak_add_doc=False,
        how=None,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        interpretation_executor=None,
        filter_branch=unset,
    ):
        """
        Args:
            expressions (None, str, or list of str): Names of ``RFields`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``RFields``
                selected by the filters are included. (Not implemented yet.)
            cut (None or str): If not None, this expression filters all of the
                ``expressions``. (Not implemented yet.)
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool, or None): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. If the function
                returns False or None, the ``RField`` is excluded; if the function
                returns True, it is included.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases.
                Uses the ``language`` engine to evaluate. (Not implemented yet.)
            language (:doc:`uproot.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``. (Not implemented yet.)
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.RNTuple.RNTuple.num_entries`. If negative,
                count from the end, like a Python slice.
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``RPages``; if None, the
                file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
                is used. (Not implemented yet.)
            array_cache ("inherit", None, MutableMapping, or memory size): Cache of arrays;
                if "inherit", use the file's cache; if None, do not use a cache;
                if a memory size, create a new cache of this size. (Not implemented yet.)
            library (str or :doc:`uproot.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas. (Not implemented yet.)
            ak_add_doc (bool | dict ): If True and ``library="ak"``, add the RField ``name``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value} and ``library="ak"``, add the RField ``value`` to the
                Awkward ``key`` parameter of the array.
            how (None, str, or container type): Library-dependent instructions
                for grouping. The only recognized container types are ``tuple``,
                ``list``, and ``dict``. Note that the container *type itself*
                must be passed as ``how``, not an instance of that type (i.e.
                ``how=tuple``, not ``how=()``).
            interpretation_executor (None): This argument is not used and is only included for now
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns a group of arrays from the ``RNTuple``.

        For example:

        .. code-block:: python

            >>> my_ntuple.arrays()
            <Array [{my_vector: [1, 2]}, {...}] type='2 * {my_vector: var * int64}'>

        See also :ref:`uproot.behaviors.RNTuple.HasFields.array` to read a single
        ``RField`` as an array.

        See also :ref:`uproot.behaviors.RNTuple.HasFields.iterate` to iterate over
        the array in contiguous ranges of entries.
        """
        # This temporarily provides basic functionality while expressions are properly implemented
        if expressions is not None:
            if filter_name == no_filter:
                filter_name = expressions
            else:
                raise ValueError(
                    "Expressions are not supported yet. They are currently equivalent to filter_name."
                )

        entry_start, entry_stop = (
            uproot.behaviors.TBranch._regularize_entries_start_stop(
                self.num_entries, entry_start, entry_stop
            )
        )
        library = uproot.interpretation.library._regularize_library(library)

        clusters = self.ntuple.cluster_summaries
        cluster_starts = numpy.array([c.num_first_entry for c in clusters])
        start_cluster_idx = (
            numpy.searchsorted(cluster_starts, entry_start, side="right") - 1
        )
        stop_cluster_idx = numpy.searchsorted(cluster_starts, entry_stop, side="right")
        cluster_num_entries = numpy.sum(
            [c.num_entries for c in clusters[start_cluster_idx:stop_cluster_idx]]
        )

        form = self.to_akform(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            filter_branch=filter_branch,
        )

        # only read columns mentioned in the awkward form
        target_cols = []
        container_dict = {}
        _recursive_find(form, target_cols)
        for key in target_cols:
            if "column" in key and "union" not in key:
                key_nr = int(key.split("-")[1])
                dtype_byte = self.ntuple.column_records[key_nr].type

                content = self.ntuple.read_col_pages(
                    key_nr,
                    range(start_cluster_idx, stop_cluster_idx),
                    dtype_byte=dtype_byte,
                    pad_missing_element=True,
                )
                if "cardinality" in key:
                    content = numpy.diff(content)
                if dtype_byte == uproot.const.rntuple_col_type_to_num_dict["switch"]:
                    kindex, tags = uproot.models.RNTuple._split_switch_bits(content)
                    # Find invalid variants and adjust buffers accordingly
                    invalid = numpy.flatnonzero(tags == -1)
                    if len(invalid) > 0:
                        kindex = numpy.delete(kindex, invalid)
                        tags = numpy.delete(tags, invalid)
                        invalid -= numpy.arange(len(invalid))
                        optional_index = numpy.insert(
                            numpy.arange(len(kindex), dtype=numpy.int64), invalid, -1
                        )
                    else:
                        optional_index = numpy.arange(len(kindex), dtype=numpy.int64)
                    container_dict[f"{key}-index"] = optional_index
                    container_dict[f"{key}-union-index"] = kindex
                    container_dict[f"{key}-union-tags"] = tags
                else:
                    # don't distinguish data and offsets
                    container_dict[f"{key}-data"] = content
                    container_dict[f"{key}-offsets"] = content
        cluster_offset = cluster_starts[start_cluster_idx]
        entry_start -= cluster_offset
        entry_stop -= cluster_offset
        arrays = uproot.extras.awkward().from_buffers(
            form, cluster_num_entries, container_dict, allow_noncanonical_form=True
        )[entry_start:entry_stop]

        # no longer needed; save memory
        del container_dict

        # FIXME: This is not right, but it might temporarily work
        if library.name == "np":
            return arrays.to_numpy()

        # TODO: This should be done with library.group, if possible
        if how is tuple:
            arrays = tuple(arrays[f] for f in arrays.fields)
        elif how is list:
            arrays = [arrays[f] for f in arrays.fields]
        elif how is dict:
            arrays = {f: arrays[f] for f in arrays.fields}
        elif how is not None:
            raise ValueError(
                f"unrecognized 'how' parameter: {how}. Options are None, tuple, list and dict."
            )

        return arrays

    def __array__(self, *args, **kwargs):
        if isinstance(self, uproot.behaviors.RNTuple.RNTuple):
            out = self.arrays(library="np")
        else:
            out = self.array(library="np")
        if args == () and kwargs == {}:
            return out
        else:
            return numpy.array(out, *args, **kwargs)

    def iterate(
        self,
        expressions=None,  # TODO: Not implemented yet
        cut=None,  # TODO: Not implemented yet
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        aliases=None,  # TODO: Not implemented yet
        language=uproot.language.python.python_language,  # TODO: Not implemented yet
        entry_start=None,
        entry_stop=None,
        step_size="100 MB",
        decompression_executor=None,  # TODO: Not implemented yet
        library="ak",  # TODO: Not implemented yet
        ak_add_doc=False,  # TODO: Not implemented yet
        how=None,
        report=False,  # TODO: Not implemented yet
        # For compatibility reasons we also accepts kwargs meant for TTrees
        interpretation_executor=None,
        filter_branch=unset,
    ):
        """
        Args:
            expressions (None, str, or list of str): Names of ``RFields`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``RFields``
                selected by the filters are included. (Not implemented yet.)
            cut (None or str): If not None, this expression filters all of the
                ``expressions``. (Not implemented yet.)
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``EFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool, or None): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. If the function
                returns False or None, the ``RField`` is excluded; if the function
                returns True, it is included.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases.
                Uses the ``language`` engine to evaluate. (Not implemented yet.)
            language (:doc:`uproot.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``. (Not implemented yet.)
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.RNTuple.HasFields.num_entries`. If negative,
                count from the end, like a Python slice.
            step_size (int or str): If an integer, the maximum number of entries to
                include in each iteration step; if a string, the maximum memory size
                to include. The string must be a number followed by a memory unit,
                such as "100 MB".
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``RPages``; if None, the
                file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
                is used. (Not implemented yet.)
            library (str or :doc:`uproot.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas. (Not implemented yet.)
            ak_add_doc (bool | dict ): If True and ``library="ak"``, add the RField ``name``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value} and ``library="ak"``, add the RField ``value`` to the
                Awkward ``key`` parameter of the array. (Not implemented yet.)
            how (None, str, or container type): Library-dependent instructions
                for grouping. The only recognized container types are ``tuple``,
                ``list``, and ``dict``. Note that the container *type itself*
                must be passed as ``how``, not an instance of that type (i.e.
                ``how=tuple``, not ``how=()``).
            report (bool): If True, this generator yields
                (arrays, :doc:`uproot.behaviors.TBranch.Report`) pairs; if False,
                it only yields arrays. The report has data about the ``TFile``,
                ``RNTuple``, and global and local entry ranges. (Not implemented yet.)
            interpretation_executor (None): This argument is not used and is only included for now
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Iterates through contiguous chunks of entries from the ``RNTuple``.

        For example:

        .. code-block:: python

            >>> for array in rntuple.iterate(filter_name=["x", "y"], step_size=100):
            ...     # each of the following have 100 entries
            ...     array["x"], array["y"]

        See also :ref:`uproot.behaviors.RNTuple.HasFields.arrays` to read
        everything in a single step, without iteration.

        See also :doc:`uproot.behaviors.RNTuple.iterate` to iterate over many
        files.
        """
        # This temporarily provides basic functionality while expressions are properly implemented
        if expressions is not None:
            if filter_name == no_filter:
                filter_name = expressions
            else:
                raise ValueError(
                    "Expressions are not supported yet. They are currently equivalent to filter_name."
                )

        entry_start, entry_stop = (
            uproot.behaviors.TBranch._regularize_entries_start_stop(
                self.ntuple.num_entries, entry_start, entry_stop
            )
        )

        akform = self.to_akform(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            filter_branch=filter_branch,
        )

        step_size = _regularize_step_size(
            self.ntuple, akform, step_size, entry_start, entry_stop
        )
        # TODO: This can be done more efficiently
        for start in range(0, self.num_entries, step_size):
            yield self.arrays(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_field=filter_field,
                entry_start=start,
                entry_stop=start + step_size,
                library=library,
                how=how,
                filter_branch=filter_branch,
            )

    def keys(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        ignore_duplicates=False,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            ignore_duplicates (bool): If True, return a set of the keys; otherwise, return the full list of keys.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns the names of the subfields as a list of strings.
        """
        return list(
            self.iterkeys(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_field=filter_field,
                recursive=recursive,
                full_paths=full_paths,
                ignore_duplicates=ignore_duplicates,
                filter_branch=filter_branch,
            )
        )

    def values(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns the subfields as a list of
        :doc:`uproot.behaviors.RField.RField`.

        (Note: with ``recursive=False``, this is the same as
        :ref:`uproot.behaviors.TBranch.HasFields.fields`.)
        """
        return list(
            self.itervalues(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_field=filter_field,
                recursive=recursive,
                filter_branch=filter_branch,
            )
        )

    def items(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns (name, field) pairs of the subfields as a list of 2-tuples
        of (str, :doc:`uproot.behaviors.RField.RField`).
        """
        return list(
            self.iteritems(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_field=filter_field,
                recursive=recursive,
                full_paths=full_paths,
                filter_branch=filter_branch,
            )
        )

    def typenames(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns (name, typename) pairs of the subfields as a dict of
        str \u2192 str.
        """
        return dict(
            self.itertypenames(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_field=filter_field,
                recursive=recursive,
                full_paths=full_paths,
                filter_branch=filter_branch,
            )
        )

    def iterkeys(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        ignore_duplicates=False,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            ignore_duplicates (bool): If True, return a set of the keys; otherwise, return the full list of keys.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.


        Returns the names of the subfields as an iterator over strings.
        """
        for k, _ in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            recursive=recursive,
            full_paths=full_paths,
            ignore_duplicates=ignore_duplicates,
            filter_branch=filter_branch,
        ):
            yield k

    def itervalues(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns the subfields as an iterator over
        :doc:`uproot.behaviors.RField.RField`.

        (Note: with ``recursive=False``, this is the same as
        :ref:`uproot.behaviors.RField.HasFields.fields`.)
        """
        for _, v in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            recursive=recursive,
            full_paths=False,
            filter_branch=filter_branch,
        ):
            yield v

    def iteritems(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        ignore_duplicates=False,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            ignore_duplicates (bool): If True, return a set of the keys; otherwise, return the full list of keys.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.


        Returns (name, field) pairs of the subfields as an iterator over
        2-tuples of (str, :doc:`uproot.behaviors.RField.RField`).
        """
        if filter_branch is not unset:
            warnings.warn(
                "the filter_branch kwarg should not be used for RNTuples",
                DeprecationWarning,
                stacklevel=1,
            )
            filter_field = filter_branch

        filter_name = uproot._util.regularize_filter(filter_name)
        filter_typename = uproot._util.regularize_filter(filter_typename)
        if filter_field is None:
            filter_field = no_filter
        elif callable(filter_field):
            pass
        else:
            raise TypeError(
                f"filter_field must be None or a function: RField -> bool, not {filter_field!r}"
            )

        keys_set = set()

        for field in self.fields:
            if (
                (
                    filter_name is no_filter
                    or _filter_name_deep(filter_name, self, field)
                )
                and (filter_typename is no_filter or filter_typename(field.typename))
                and (filter_field is no_filter or filter_field(field))
            ):
                if ignore_duplicates and field.name in keys_set:
                    pass
                else:
                    keys_set.add(field.name)
                    yield field.name, field

            if recursive:
                for k1, v in field.iteritems(
                    recursive=recursive,
                    filter_name=no_filter,
                    filter_typename=filter_typename,
                    filter_field=filter_field,
                    full_paths=full_paths,
                ):
                    k2 = f"{field.name}.{k1}" if full_paths else k1
                    if filter_name is no_filter or _filter_name_deep(
                        filter_name, self, v
                    ):
                        if ignore_duplicates and k2 in keys_set:
                            pass
                        else:
                            keys_set.add(k2)
                            yield k2, v

    def itertypenames(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns (name, typename) pairs of the subfields as an iterator over
        2-tuples of (str, str).
        """
        for k, v in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=full_paths,
        ):
            yield k, v.typename

    def _ipython_key_completions_(self):
        """
        Supports key-completion in an IPython or Jupyter kernel.
        """
        return self.iterkeys()

    def num_entries_for(
        self,
        memory_size,
        expressions=None,  # TODO: Not implemented yet
        cut=None,  # TODO: Not implemented yet
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        aliases=None,  # TODO: Not implemented yet
        language=uproot.language.python.python_language,  # TODO: Not implemented yet
        entry_start=None,
        entry_stop=None,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            memory_size (int or str): An integer is interpreted as a number of
                bytes and a string must be a number followed by a unit, such as
                "100 MB".
            expressions (None, str, or list of str): Names of ``RFields`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``RFields``
                selected by the filters are included. (Not implemented yet.)
            cut (None or str): If not None, this expression filters all of the
                ``expressions``. (Not implemented yet.)
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RField.RField` \u2192 bool, or None): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RField.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases.
                Uses the ``language`` engine to evaluate.
            language (:doc:`uproot.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.RNTuple.HasFields.num_entries`. If negative,
                count from the end, like a Python slice.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns an *approximate* step size as a number of entries to read
        a given ``memory_size`` in each step.

        This method does not actually read the ``RField`` data or compute any
        expressions to arrive at its estimate. It only uses metadata from the
        already-loaded ``RNTuple``; it only needs ``language`` to parse the
        expressions, not to evaluate them.

        This is the algorithm that
        :ref:`uproot.behaviors.RNTuple.HasFields.iterate` uses to convert a
        ``step_size`` expressed in memory units into a number of entries.
        """
        # This temporarily provides basic functionality while expressions are properly implemented
        if expressions is not None:
            if filter_name == no_filter:
                filter_name = expressions
            else:
                raise ValueError(
                    "Expressions are not supported yet. They are currently equivalent to filter_name."
                )

        target_num_bytes = uproot._util.memory_size(memory_size)

        entry_start, entry_stop = (
            uproot.behaviors.TBranch._regularize_entries_start_stop(
                self.ntuple.num_entries, entry_start, entry_stop
            )
        )

        akform = self.to_akform(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            filter_branch=filter_branch,
        )

        return _num_entries_for(self, akform, target_num_bytes, entry_start, entry_stop)

    def __getitem__(self, where):
        original_where = where

        if uproot._util.isint(where):
            return self.fields[where]
        elif isinstance(where, str):
            where = uproot._util.ensure_str(where)
            where = where.replace("/", ".").replace("\\", ".")
        else:
            raise TypeError(f"where must be an integer or a string, not {where!r}")

        if where.startswith("."):
            recursive = False
            where = where[1:]
        else:
            recursive = True

        if self._lookup is None:
            self._lookup = {f.name: f for f in self.fields}
        got = self._lookup.get(where)
        if got is not None:
            return got

        if "." in where:
            this = self
            try:
                for piece in where.split("."):
                    if piece != "":
                        this = this[piece]
            except uproot.KeyInFileError:
                raise uproot.KeyInFileError(
                    original_where,
                    keys=self.keys(recursive=recursive),
                    file_path=self._file.file_path,  # TODO
                    object_path=self.object_path,  # TODO
                ) from None
            return this

        elif recursive:
            got = _get_recursive(self, where)
            if got is not None:
                return got
            else:
                raise uproot.KeyInFileError(
                    original_where,
                    keys=self.keys(recursive=recursive),
                    file_path=self._file.file_path,
                    object_path=self.object_path,
                )

        else:
            raise uproot.KeyInFileError(
                original_where,
                keys=self.keys(recursive=recursive),
                file_path=self._file.file_path,
                object_path=self.object_path,
            )

    def __iter__(self):
        yield from self.fields

    def __len__(self):
        return len(self.fields)

    def show(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        name_width=20,
        typename_width=24,
        path_width=30,
        stream=sys.stdout,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        full_paths=unset,
        filter_branch=unset,
        interpretation_width=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool, or None): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, recursively descend into subfields.
            name_width (int): Number of characters to reserve for the ``TBranch``
                names.
            typename_width (int): Number of characters to reserve for the C++
                typenames.
            interpretation_width (int): Number of characters to reserve for the
                :doc:`uproot.interpretation.Interpretation` displays.
            stream (object with a ``write(str)`` method): Stream to write the
                output to.
            full_paths (None): This argument is not used and is only included for now
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.
            interpretation_width (None): This argument is not used and is only included for now
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Interactively display the ``RFields``.

        For example,

        .. code-block::

            >>> my_ntuple.show()
            name                 | typename                 | path
            ---------------------+--------------------------+-------------------------------
            my_int               | std::int64_t             | my_int
            my_vec               | std::vector<std::int6... | my_vec
            _0                   | std::int64_t             | my_vec._0
        """
        if name_width < 3:
            raise ValueError("'name_width' must be at least 3")
        if typename_width < 3:
            raise ValueError("'typename_width' must be at least 3")
        if path_width < 3:
            raise ValueError("'path_width' must be at least 3")

        formatter = f"{{0:{name_width}.{name_width}}} | {{1:{typename_width}.{typename_width}}} | {{2:{path_width}.{path_width}}}"

        stream.write(formatter.format("name", "typename", "path"))
        stream.write(
            "\n"
            + "-" * name_width
            + "-+-"
            + "-" * typename_width
            + "-+-"
            + "-" * path_width
            + "\n"
        )

        if isinstance(self, uproot.models.RNTuple.RField):
            stream.write(formatter.format(self.name, self.typename, self.path) + "\n")

        for field in self.itervalues(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            recursive=recursive,
            filter_branch=filter_branch,
        ):
            name = field.name
            typename = field.typename
            path = field.path

            if len(name) > name_width:
                name = name[: name_width - 3] + "..."
            if len(typename) > typename_width:
                typename = typename[: typename_width - 3] + "..."
            if len(path) > path_width:
                path = path[: path_width - 3] + "..."

            stream.write(formatter.format(name, typename, path).rstrip(" ") + "\n")

    @property
    def source(self) -> uproot.source.chunk.Source | None:
        """Returns the associated source of data for this container, if it exists

        Returns: uproot.source.chunk.Source or None
        """
        if isinstance(self.ntuple._file, uproot.reading.ReadOnlyFile):
            return self.ntuple._file.source
        return None


class RNTuple(HasFields):
    """
    Behaviors for an ``RNTuple``, which mostly consist of array-reading methods.

    Since a :doc:`uproot.behaviors.RNTuple.RNTuple` is a
    :doc:`uproot.behaviors.RNTuple.HasFields`, it is also a Python
    ``Mapping``, which uses square bracket syntax to extract subbranches:

    .. code-block:: python

        my_rntuple["field"]
        my_rntuple["field"]["subfield"]
        my_rntuple["field.subfield"]
        my_rntuple["field.subfield.subsubfield"]
        my_rntuple["field/subfield/subsubfield"]
        my_rntuple["field\\subfield\\subsubfield"]
    """

    def __repr__(self):
        if len(self) == 0:
            return f"<{self.classname} {self.name!r} at 0x{id(self):012x}>"
        else:
            return f"<{self.classname} {self.name!r} ({len(self)} fields) at 0x{id(self):012x}>"

    @property
    def name(self):
        """
        Name of the ``RNTuple``.
        """
        return self.header.ntuple_name

    @property
    def object_path(self):
        """
        Object path of the ``RNTuple``.
        """
        return self.parent.object_path

    @property
    def cache_key(self):
        """
        String that uniquely specifies this ``RNTuple`` in its path, to use as
        part of object and array cache keys.
        """
        return f"{self.parent.cache_key}{self.name};{self.parent.fCycle}"


def _filter_name_deep(filter_name, hasfields, field):
    shallow = name = field.name
    if filter_name(name):
        return True
    while field is not hasfields:
        field = field.parent
        if field is not hasfields:
            name = field.name + "." + name
    if name != shallow and filter_name(name):
        return True
    return filter_name("." + name)


def _keys_deep(hasbranches):
    out = set()
    for branch in hasbranches.itervalues(recursive=True):
        name = branch.name
        out.add(name)
        while branch is not hasbranches:
            branch = branch.parent  # noqa: PLW2901 (overwriting branch)
            if branch is not hasbranches:
                name = branch.name + "/" + name
        out.add(name)
        out.add("/" + name)
    return out


def _get_recursive(hasfields, where):
    if hasfields._lookup is None:
        hasfields._lookup = {f.name: f for f in hasfields.fields}
    got = hasfields._lookup.get(where)
    if got is not None:
        return got
    for field in hasfields.fields:
        got = _get_recursive(field, where)
        if got is not None:
            return got
    else:
        return None


def _num_entries_for(ntuple, akform, target_num_bytes, entry_start, entry_stop):
    # TODO: there might be a better way to estimate the number of entries
    clusters = ntuple.cluster_summaries
    cluster_starts = numpy.array([c.num_first_entry for c in clusters])

    start_cluster_idx = (
        numpy.searchsorted(cluster_starts, entry_start, side="right") - 1
    )
    stop_cluster_idx = numpy.searchsorted(cluster_starts, entry_stop, side="right")

    target_cols = []
    _recursive_find(akform, target_cols)

    total_bytes = 0
    for key in target_cols:
        if "column" in key and "union" not in key:
            key_nr = int(key.split("-")[1])
            for cluster in range(start_cluster_idx, stop_cluster_idx):
                pages = ntuple.page_link_list[cluster][key_nr].pages
                total_bytes += sum(page.locator.num_bytes for page in pages)

    total_entries = entry_stop - entry_start
    if total_bytes == 0:
        num_entries = 0
    else:
        num_entries = round(target_num_bytes * total_entries / total_bytes)
    if num_entries <= 0:
        return 1
    else:
        return num_entries


def _regularize_step_size(ntuple, akform, step_size, entry_start, entry_stop):
    if uproot._util.isint(step_size):
        return step_size
    target_num_bytes = uproot._util.memory_size(
        step_size,
        "number of entries or memory size string with units "
        f"(such as '100 MB') required, not {step_size!r}",
    )
    return _num_entries_for(ntuple, akform, target_num_bytes, entry_start, entry_stop)


def _recursive_find(form, res):
    ak = uproot.extras.awkward()

    if hasattr(form, "form_key"):
        res.append(form.form_key)
    if hasattr(form, "contents"):
        for c in form.contents:
            _recursive_find(c, res)
    if hasattr(form, "content") and issubclass(type(form.content), ak.forms.Form):
        _recursive_find(form.content, res)
