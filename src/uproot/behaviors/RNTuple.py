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
            keys = [f"{self.path}.{k}" for k in keys]
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
        library="ak",
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
                ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
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
        return uproot.extras.awkward().from_buffers(
            form, cluster_num_entries, container_dict, allow_noncanonical_form=True
        )[entry_start:entry_stop]

        # return _ak_add_doc(
        #     library.group(output, expression_context, how), self, ak_add_doc
        # )

    def __array__(self, *args, **kwargs):
        if isinstance(self, uproot.behaviors.RNTuple.RNTuple):
            out = self.arrays(library="np")
        else:
            out = self.array(library="np")
        if args == () and kwargs == {}:
            return out
        else:
            return numpy.array(out, *args, **kwargs)

    # def iterate(
    #     self,
    #     expressions=None,
    #     cut=None,
    #     *,
    #     filter_name=no_filter,
    #     filter_typename=no_filter,
    #     filter_branch=no_filter,
    #     aliases=None,
    #     language=uproot.language.python.python_language,
    #     entry_start=None,
    #     entry_stop=None,
    #     step_size="100 MB",
    #     decompression_executor=None,
    #     interpretation_executor=None,
    #     library="ak",
    #     ak_add_doc=False,
    #     how=None,
    #     report=False,
    # ):
    #     """
    #     Args:
    #         expressions (None, str, or list of str): Names of ``TBranches`` or
    #             aliases to convert to arrays or mathematical expressions of them.
    #             Uses the ``language`` to evaluate. If None, all ``TBranches``
    #             selected by the filters are included.
    #         cut (None or str): If not None, this expression filters all of the
    #             ``expressions``.
    #         filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
    #             filter to select ``TBranches`` by name.
    #         filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
    #             filter to select ``TBranches`` by type.
    #         filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
    #             filter to select ``TBranches`` using the full
    #             :doc:`uproot.behaviors.TBranch.TBranch` object. If the function
    #             returns False or None, the ``TBranch`` is excluded; if the function
    #             returns True, it is included with its standard
    #             :ref:`uproot.behaviors.TBranch.TBranch.interpretation`; if an
    #             :doc:`uproot.interpretation.Interpretation`, this interpretation
    #             overrules the standard one.
    #         aliases (None or dict of str \u2192 str): Mathematical expressions that
    #             can be used in ``expressions`` or other aliases (without cycles).
    #             Uses the ``language`` engine to evaluate. If None, only the
    #             :ref:`uproot.behaviors.TBranch.TBranch.aliases` are available.
    #         language (:doc:`uproot.language.Language`): Language used to interpret
    #             the ``expressions`` and ``aliases``.
    #         entry_start (None or int): The first entry to include. If None, start
    #             at zero. If negative, count from the end, like a Python slice.
    #         entry_stop (None or int): The first entry to exclude (i.e. one greater
    #             than the last entry to include). If None, stop at
    #             :ref:`uproot.behaviors.TTree.TTree.num_entries`. If negative,
    #             count from the end, like a Python slice.
    #         step_size (int or str): If an integer, the maximum number of entries to
    #             include in each iteration step; if a string, the maximum memory size
    #             to include. The string must be a number followed by a memory unit,
    #             such as "100 MB".
    #         decompression_executor (None or Executor with a ``submit`` method): The
    #             executor that is used to decompress ``TBaskets``; if None, the
    #             file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
    #             is used.
    #         interpretation_executor (None or Executor with a ``submit`` method): The
    #             executor that is used to interpret uncompressed ``TBasket`` data as
    #             arrays; if None, the file's :ref:`uproot.reading.ReadOnlyFile.interpretation_executor`
    #             is used.
    #         library (str or :doc:`uproot.interpretation.library.Library`): The library
    #             that is used to represent arrays. Options are ``"np"`` for NumPy,
    #             ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
    #         ak_add_doc (bool | dict ): If True and ``library="ak"``, add the TBranch ``title``
    #             to the Awkward ``__doc__`` parameter of the array.
    #             if dict = {key:value} and ``library="ak"``, add the TBranch ``value`` to the
    #             Awkward ``key`` parameter of the array.
    #         how (None, str, or container type): Library-dependent instructions
    #             for grouping. The only recognized container types are ``tuple``,
    #             ``list``, and ``dict``. Note that the container *type itself*
    #             must be passed as ``how``, not an instance of that type (i.e.
    #             ``how=tuple``, not ``how=()``).
    #         report (bool): If True, this generator yields
    #             (arrays, :doc:`uproot.behaviors.TBranch.Report`) pairs; if False,
    #             it only yields arrays. The report has data about the ``TFile``,
    #             ``TTree``, and global and local entry ranges.

    #     Iterates through contiguous chunks of entries from the ``TTree``.

    #     For example:

    #     .. code-block:: python

    #         >>> for array in tree.iterate(["x", "y"], step_size=100):
    #         ...     # each of the following have 100 entries
    #         ...     array["x"], array["y"]

    #     See also :ref:`uproot.behaviors.TBranch.HasBranches.arrays` to read
    #     everything in a single step, without iteration.

    #     See also :doc:`uproot.behaviors.TBranch.iterate` to iterate over many
    #     files.
    #     """
    #     keys = _keys_deep(self)
    #     if isinstance(self, TBranch) and expressions is None and len(keys) == 0:
    #         filter_branch = uproot._util.regularize_filter(filter_branch)
    #         yield from self.parent.iterate(
    #             expressions=expressions,
    #             cut=cut,
    #             filter_name=filter_name,
    #             filter_typename=filter_typename,
    #             filter_branch=lambda branch: branch is self and filter_branch(branch),
    #             aliases=aliases,
    #             language=language,
    #             entry_start=entry_start,
    #             entry_stop=entry_stop,
    #             step_size=step_size,
    #             decompression_executor=decompression_executor,
    #             interpretation_executor=interpretation_executor,
    #             library=library,
    #             how=how,
    #             report=report,
    #         )

    #     else:
    #         entry_start, entry_stop = _regularize_entries_start_stop(
    #             self.tree.num_entries, entry_start, entry_stop
    #         )
    #         decompression_executor, interpretation_executor = _regularize_executors(
    #             decompression_executor, interpretation_executor, self._file
    #         )
    #         library = uproot.interpretation.library._regularize_library(library)

    #         aliases = _regularize_aliases(self, aliases)
    #         (
    #             arrays,
    #             expression_context,
    #             branchid_interpretation,
    #         ) = _regularize_expressions(
    #             self,
    #             expressions,
    #             cut,
    #             filter_name,
    #             filter_typename,
    #             filter_branch,
    #             keys,
    #             aliases,
    #             language,
    #             (lambda branchname, interpretation: None),
    #         )

    #         entry_step = _regularize_step_size(
    #             self, step_size, entry_start, entry_stop, branchid_interpretation
    #         )

    #         previous_baskets = {}
    #         for sub_entry_start in range(entry_start, entry_stop, entry_step):
    #             sub_entry_stop = min(sub_entry_start + entry_step, entry_stop)
    #             if sub_entry_stop - sub_entry_start == 0:
    #                 continue

    #             ranges_or_baskets = []
    #             checked = set()
    #             for _, context in expression_context:
    #                 for branch in context["branches"]:
    #                     if branch.cache_key not in checked and not isinstance(
    #                         branchid_interpretation[branch.cache_key],
    #                         uproot.interpretation.grouped.AsGrouped,
    #                     ):
    #                         checked.add(branch.cache_key)
    #                         for (
    #                             basket_num,
    #                             range_or_basket,
    #                         ) in branch.entries_to_ranges_or_baskets(
    #                             sub_entry_start, sub_entry_stop
    #                         ):
    #                             previous_basket = previous_baskets.get(
    #                                 (branch.cache_key, basket_num)
    #                             )
    #                             if previous_basket is None:
    #                                 ranges_or_baskets.append(
    #                                     (branch, basket_num, range_or_basket)
    #                                 )
    #                             else:
    #                                 ranges_or_baskets.append(
    #                                     (branch, basket_num, previous_basket)
    #                                 )

    #             arrays = {}
    #             interp_options = {"ak_add_doc": ak_add_doc}
    #             _ranges_or_baskets_to_arrays(
    #                 self,
    #                 ranges_or_baskets,
    #                 branchid_interpretation,
    #                 sub_entry_start,
    #                 sub_entry_stop,
    #                 decompression_executor,
    #                 interpretation_executor,
    #                 library,
    #                 arrays,
    #                 True,
    #                 interp_options,
    #             )

    #             _fix_asgrouped(
    #                 arrays,
    #                 expression_context,
    #                 branchid_interpretation,
    #                 library,
    #                 how,
    #                 ak_add_doc,
    #             )

    #             output = language.compute_expressions(
    #                 self,
    #                 arrays,
    #                 expression_context,
    #                 keys,
    #                 aliases,
    #                 self.file.file_path,
    #                 self.object_path,
    #             )

    #             # no longer needed; save memory
    #             del arrays

    #             minimized_expression_context = [
    #                 (e, c)
    #                 for e, c in expression_context
    #                 if c["is_primary"] and not c["is_cut"]
    #             ]

    #             out = _ak_add_doc(
    #                 library.group(output, minimized_expression_context, how),
    #                 self,
    #                 ak_add_doc,
    #             )

    #             # no longer needed; save memory
    #             del output

    #             next_baskets = {}
    #             for branch, basket_num, basket in ranges_or_baskets:
    #                 basket_entry_start, basket_entry_stop = basket.entry_start_stop
    #                 if basket_entry_stop > sub_entry_stop:
    #                     next_baskets[branch.cache_key, basket_num] = basket

    #             previous_baskets = next_baskets

    #             # no longer needed; save memory
    #             popper = [out]
    #             del out

    #             if report:
    #                 yield popper.pop(), Report(self, sub_entry_start, sub_entry_stop)
    #             else:
    #                 yield popper.pop()

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

    # def num_entries_for(
    #     self,
    #     memory_size,
    #     expressions=None,
    #     cut=None,
    #     *,
    #     filter_name=no_filter,
    #     filter_typename=no_filter,
    #     filter_branch=no_filter,
    #     aliases=None,
    #     language=uproot.language.python.python_language,
    #     entry_start=None,
    #     entry_stop=None,
    # ):
    #     """
    #     Args:
    #         memory_size (int or str): An integer is interpreted as a number of
    #             bytes and a string must be a number followed by a unit, such as
    #             "100 MB".
    #         expressions (None, str, or list of str): Names of ``TBranches`` or
    #             aliases to convert to arrays or mathematical expressions of them.
    #             Uses the ``language`` to evaluate. If None, all ``TBranches``
    #             selected by the filters are included.
    #         cut (None or str): If not None, this expression filters all of the
    #             ``expressions``.
    #         filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
    #             filter to select ``TBranches`` by name.
    #         filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
    #             filter to select ``TBranches`` by type.
    #         filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
    #             filter to select ``TBranches`` using the full
    #             :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
    #             included if the function returns True, excluded if it returns False.
    #         aliases (None or dict of str \u2192 str): Mathematical expressions that
    #             can be used in ``expressions`` or other aliases (without cycles).
    #             Uses the ``language`` engine to evaluate. If None, only the
    #             :ref:`uproot.behaviors.TBranch.TBranch.aliases` are available.
    #         language (:doc:`uproot.language.Language`): Language used to interpret
    #             the ``expressions`` and ``aliases``.
    #         entry_start (None or int): The first entry to include. If None, start
    #             at zero. If negative, count from the end, like a Python slice.
    #         entry_stop (None or int): The first entry to exclude (i.e. one greater
    #             than the last entry to include). If None, stop at
    #             :ref:`uproot.behaviors.TTree.TTree.num_entries`. If negative,
    #             count from the end, like a Python slice.

    #     Returns an *approximate* step size as a number of entries to read
    #     a given ``memory_size`` in each step.

    #     This method does not actually read the ``TBranch`` data or compute any
    #     expressions to arrive at its estimate. It only uses metadata from the
    #     already-loaded ``TTree``; it only needs ``language`` to parse the
    #     expressions, not to evaluate them.

    #     In addition, the estimate is based on compressed ``TBasket`` sizes
    #     (the amount of data that would have to be read), not uncompressed
    #     ``TBasket`` sizes (the amount of data that the final arrays would use
    #     in memory, without considering ``cuts``).

    #     This is the algorithm that
    #     :ref:`uproot.behaviors.TBranch.HasBranches.iterate` uses to convert a
    #     ``step_size`` expressed in memory units into a number of entries.
    #     """
    #     target_num_bytes = uproot._util.memory_size(memory_size)

    #     entry_start, entry_stop = _regularize_entries_start_stop(
    #         self.tree.num_entries, entry_start, entry_stop
    #     )

    #     keys = _keys_deep(self)
    #     aliases = _regularize_aliases(self, aliases)
    #     arrays, expression_context, branchid_interpretation = _regularize_expressions(
    #         self,
    #         expressions,
    #         cut,
    #         filter_name,
    #         filter_typename,
    #         filter_branch,
    #         keys,
    #         aliases,
    #         language,
    #         (lambda branchname, interpretation: None),
    #     )

    #     return _hasbranches_num_entries_for(
    #         self, target_num_bytes, entry_start, entry_stop, branchid_interpretation
    #     )

    # def common_entry_offsets(
    #     self,
    #     *,
    #     filter_name=no_filter,
    #     filter_typename=no_filter,
    #     filter_branch=no_filter,
    #     recursive=True,
    # ):
    #     """
    #     Args:
    #         filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
    #             filter to select ``TBranches`` by name.
    #         filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
    #             filter to select ``TBranches`` by type.
    #         filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
    #             filter to select ``TBranches`` using the full
    #             :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
    #             included if the function returns True, excluded if it returns False.
    #         recursive (bool): If True, descend into any nested subbranches.
    #             If False, only consider branches directly accessible under this
    #             object. (Only applies when ``branches=None``.)

    #     Returns entry offsets in which ``TBasket`` boundaries align in the
    #     specified set of branches.

    #     If this :doc:`uproot.behaviors.TBranch.TBranch` has no subbranches,
    #     the output is identical to
    #     :ref:`uproot.behaviors.TBranch.TBranch.entry_offsets`.
    #     """
    #     common_offsets = None
    #     for branch in self.itervalues(
    #         filter_name=filter_name,
    #         filter_typename=filter_typename,
    #         filter_branch=filter_branch,
    #         recursive=recursive,
    #     ):
    #         if common_offsets is None:
    #             common_offsets = set(branch.entry_offsets)
    #         else:
    #             common_offsets = common_offsets.intersection(set(branch.entry_offsets))
    #     return sorted(common_offsets)

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


def _hasbranches_num_entries_for(
    hasbranches, target_num_bytes, entry_start, entry_stop, branchid_interpretation
):
    total_bytes = 0.0
    for branch in hasbranches.itervalues(recursive=True):
        if branch.cache_key in branchid_interpretation:
            entry_offsets = branch.entry_offsets
            start = entry_offsets[0]
            for basket_num, stop in enumerate(entry_offsets[1:]):
                if entry_start < stop and start <= entry_stop:
                    total_bytes += branch.basket_compressed_bytes(basket_num)
                start = stop

    total_entries = entry_stop - entry_start
    if total_bytes == 0:
        num_entries = 0
    else:
        num_entries = int(round(target_num_bytes * total_entries / total_bytes))
    if num_entries <= 0:
        return 1
    else:
        return num_entries


def _regularize_step_size(
    hasbranches, step_size, entry_start, entry_stop, branchid_interpretation
):
    if uproot._util.isint(step_size):
        return step_size
    target_num_bytes = uproot._util.memory_size(
        step_size,
        "number of entries or memory size string with units "
        f"(such as '100 MB') required, not {step_size!r}",
    )
    return _hasbranches_num_entries_for(
        hasbranches, target_num_bytes, entry_start, entry_stop, branchid_interpretation
    )


def _ak_add_doc(array, hasbranches, ak_add_doc):
    if type(array).__module__ == "awkward.highlevel":
        if isinstance(ak_add_doc, bool):
            if ak_add_doc:
                array.layout.parameters["__doc__"] = hasbranches.title
        elif isinstance(ak_add_doc, dict):
            array.layout.parameters.update(
                {
                    key: hasbranches.__getattribute__(value)
                    for key, value in ak_add_doc.items()
                }
            )
    return array


def _recursive_find(form, res):
    ak = uproot.extras.awkward()

    if hasattr(form, "form_key"):
        res.append(form.form_key)
    if hasattr(form, "contents"):
        for c in form.contents:
            _recursive_find(c, res)
    if hasattr(form, "content") and issubclass(type(form.content), ak.forms.Form):
        _recursive_find(form.content, res)
