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

import queue
import sys
import threading
import warnings
from collections.abc import Iterable, Mapping, MutableMapping

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

    # def arrays(
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
    #     decompression_executor=None,
    #     interpretation_executor=None,
    #     array_cache="inherit",
    #     library="ak",
    #     ak_add_doc=False,
    #     how=None,
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
    #         decompression_executor (None or Executor with a ``submit`` method): The
    #             executor that is used to decompress ``TBaskets``; if None, the
    #             file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
    #             is used.
    #         interpretation_executor (None or Executor with a ``submit`` method): The
    #             executor that is used to interpret uncompressed ``TBasket`` data as
    #             arrays; if None, the file's :ref:`uproot.reading.ReadOnlyFile.interpretation_executor`
    #             is used.
    #         array_cache ("inherit", None, MutableMapping, or memory size): Cache of arrays;
    #             if "inherit", use the file's cache; if None, do not use a cache;
    #             if a memory size, create a new cache of this size.
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

    #     Returns a group of arrays from the ``TTree``.

    #     For example:

    #     .. code-block:: python

    #         >>> my_tree["x"].array()
    #         <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>
    #         >>> my_tree["y"].array()
    #         <Array [17.4, -16.6, -16.6, ... 1.2, 1.2, 1.2] type='2304 * float64'>

    #     See also :ref:`uproot.behaviors.TBranch.TBranch.array` to read a single
    #     ``TBranch`` as an array.

    #     See also :ref:`uproot.behaviors.TBranch.HasBranches.iterate` to iterate over
    #     the array in contiguous ranges of entries.
    #     """
    #     keys = _keys_deep(self)
    #     if isinstance(self, TBranch) and expressions is None and len(keys) == 0:
    #         filter_branch = uproot._util.regularize_filter(filter_branch)
    #         return self.parent.arrays(
    #             expressions=expressions,
    #             cut=cut,
    #             filter_name=filter_name,
    #             filter_typename=filter_typename,
    #             filter_branch=lambda branch: branch is self and filter_branch(branch),
    #             aliases=aliases,
    #             language=language,
    #             entry_start=entry_start,
    #             entry_stop=entry_stop,
    #             decompression_executor=decompression_executor,
    #             interpretation_executor=interpretation_executor,
    #             array_cache=array_cache,
    #             library=library,
    #             how=how,
    #         )

    #     entry_start, entry_stop = _regularize_entries_start_stop(
    #         self.tree.num_entries, entry_start, entry_stop
    #     )
    #     decompression_executor, interpretation_executor = _regularize_executors(
    #         decompression_executor, interpretation_executor, self._file
    #     )
    #     array_cache = _regularize_array_cache(array_cache, self._file)
    #     library = uproot.interpretation.library._regularize_library(library)

    #     def get_from_cache(branchname, interpretation):
    #         if array_cache is not None:
    #             cache_key = f"{self.cache_key}:{branchname}:{interpretation.cache_key}:{entry_start}-{entry_stop}:{library.name}"
    #             return array_cache.get(cache_key)
    #         else:
    #             return None

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
    #         get_from_cache,
    #     )

    #     ranges_or_baskets = []
    #     checked = set()
    #     for _, context in expression_context:
    #         for branch in context["branches"]:
    #             if branch.cache_key not in checked and not isinstance(
    #                 branchid_interpretation[branch.cache_key],
    #                 uproot.interpretation.grouped.AsGrouped,
    #             ):
    #                 checked.add(branch.cache_key)
    #                 for (
    #                     basket_num,
    #                     range_or_basket,
    #                 ) in branch.entries_to_ranges_or_baskets(entry_start, entry_stop):
    #                     ranges_or_baskets.append((branch, basket_num, range_or_basket))

    #     interp_options = {"ak_add_doc": ak_add_doc}
    #     _ranges_or_baskets_to_arrays(
    #         self,
    #         ranges_or_baskets,
    #         branchid_interpretation,
    #         entry_start,
    #         entry_stop,
    #         decompression_executor,
    #         interpretation_executor,
    #         library,
    #         arrays,
    #         False,
    #         interp_options,
    #     )

    #     # no longer needed; save memory
    #     del ranges_or_baskets

    #     _fix_asgrouped(
    #         arrays,
    #         expression_context,
    #         branchid_interpretation,
    #         library,
    #         how,
    #         ak_add_doc,
    #     )

    #     if array_cache is not None:
    #         checked = set()
    #         for expression, context in expression_context:
    #             for branch in context["branches"]:
    #                 if branch.cache_key not in checked:
    #                     checked.add(branch.cache_key)
    #                     interpretation = branchid_interpretation[branch.cache_key]
    #                     if branch is not None:
    #                         cache_key = f"{self.cache_key}:{expression}:{interpretation.cache_key}:{entry_start}-{entry_stop}:{library.name}"
    #                     array_cache[cache_key] = arrays[branch.cache_key]

    #     output = language.compute_expressions(
    #         self,
    #         arrays,
    #         expression_context,
    #         keys,
    #         aliases,
    #         self.file.file_path,
    #         self.object_path,
    #     )

    #     # no longer needed; save memory
    #     del arrays

    #     expression_context = [
    #         (e, c) for e, c in expression_context if c["is_primary"] and not c["is_cut"]
    #     ]

    #     return _ak_add_doc(
    #         library.group(output, expression_context, how), self, ak_add_doc
    #     )

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

    # @property
    # def source(self) -> uproot.source.chunk.Source | None:
    #     """Returns the associated source of data for this container, if it exists

    #     Returns: uproot.source.chunk.Source or None
    #     """
    #     if isinstance(self, uproot.model.Model) and isinstance(
    #         self._file, uproot.reading.ReadOnlyFile
    #     ):
    #         return self._file.source
    #     return None


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
        return self.parent.fName

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

    # def show(
    #     self,
    #     *,
    #     filter_name=no_filter,
    #     filter_typename=no_filter,
    #     filter_branch=no_filter,
    #     recursive=True,
    #     full_paths=True,
    #     name_width=20,
    #     typename_width=24,
    #     interpretation_width=30,
    #     stream=sys.stdout,
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
    #         recursive (bool): If True, recursively descend into the branches'
    #             branches.
    #         full_paths (bool): If True, include the full path to each subbranch
    #             with slashes (``/``); otherwise, use the descendant's name as
    #             the display name.
    #         name_width (int): Number of characters to reserve for the ``TBranch``
    #             names.
    #         typename_width (int): Number of characters to reserve for the C++
    #             typenames.
    #         interpretation_width (int): Number of characters to reserve for the
    #             :doc:`uproot.interpretation.Interpretation` displays.
    #         stream (object with a ``write(str)`` method): Stream to write the
    #             output to.

    #     Interactively display the ``TBranches``.

    #     For example,

    #     .. code-block::

    #         name                 | typename             | interpretation
    #         ---------------------+----------------------+-----------------------------------
    #         event_number         | int32_t              | AsDtype('>i4')
    #         trigger_isomu24      | bool                 | AsDtype('bool')
    #         eventweight          | float                | AsDtype('>f4')
    #         MET                  | TVector2             | AsStridedObjects(Model_TVector2_v3
    #         jetp4                | std::vector<TLorentz | AsJagged(AsStridedObjects(Model_TL
    #         jetbtag              | std::vector<float>   | AsJagged(AsDtype('>f4'), header_by
    #         jetid                | std::vector<bool>    | AsJagged(AsDtype('bool'), header_b
    #     """
    #     if name_width < 3:
    #         raise ValueError("'name_width' must be at least 3")
    #     if typename_width < 3:
    #         raise ValueError("'typename_width' must be at least 3")
    #     if interpretation_width < 3:
    #         raise ValueError("'interpretation_width' must be at least 3")

    #     formatter = f"{{0:{name_width}.{name_width}}} | {{1:{typename_width}.{typename_width}}} | {{2:{interpretation_width}.{interpretation_width}}}"

    #     stream.write(formatter.format("name", "typename", "interpretation"))
    #     stream.write(
    #         "\n"
    #         + "-" * name_width
    #         + "-+-"
    #         + "-" * typename_width
    #         + "-+-"
    #         + "-" * interpretation_width
    #         + "\n"
    #     )

    #     if isinstance(self, TBranch):
    #         stream.write(
    #             formatter.format(self.name, self.typename, repr(self.interpretation))
    #             + "\n"
    #         )

    #     for name, branch in self.iteritems(
    #         filter_name=filter_name,
    #         filter_typename=filter_typename,
    #         filter_branch=filter_branch,
    #         recursive=recursive,
    #         full_paths=full_paths,
    #     ):
    #         typename = branch.typename
    #         interp = repr(branch.interpretation)

    #         if len(name) > name_width:
    #             name = (
    #                 name[: name_width - 3] + "..."
    #             )
    #         if len(typename) > typename_width:
    #             typename = typename[: typename_width - 3] + "..."
    #         if len(interp) > interpretation_width:
    #             interp = interp[: interpretation_width - 3] + "..."

    #         stream.write(formatter.format(name, typename, interp).rstrip(" ") + "\n")

    def array(
        self,
        interpretation=None,
        entry_start=None,
        entry_stop=None,
        *,
        decompression_executor=None,
        interpretation_executor=None,
        array_cache="inherit",
        library="ak",
        ak_add_doc=False,
    ):
        """
        Args:
            interpretation (None or :doc:`uproot.interpretation.Interpretation`): An
                interpretation of the ``TBranch`` data as an array. If None, the
                standard :ref:`uproot.behaviors.TBranch.TBranch.interpretation`
                is used, which is derived from
                :doc:`uproot.interpretation.identify.interpretation_of`.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``TBaskets``; if None, the
                file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
                is used.
            interpretation_executor (None or Executor with a ``submit`` method): The
                executor that is used to interpret uncompressed ``TBasket`` data as
                arrays; if None, the file's :ref:`uproot.reading.ReadOnlyFile.interpretation_executor`
                is used.
            array_cache ("inherit", None, MutableMapping, or memory size): Cache of arrays;
                if "inherit", use the file's cache; if None, do not use a cache;
                if a memory size, create a new cache of this size.
            library (str or :doc:`uproot.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
            ak_add_doc (bool | dict ): If True and ``library="ak"``, add the TBranch ``title``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value} and ``library="ak"``, add the TBranch ``value`` to the
                Awkward ``key`` parameter of the array.

        Returns the ``TBranch`` data as an array.

        For example:

        .. code-block:: python

            >>> array = tree
            >>> array = tree.arrays(["x", "y"])    # only reads branches "x" and "y"
            >>> array
            <Array [{x: -41.2, y: 17.4}, ... {x: 32.5, y: 1.2}], type='2304 * {"x": float64,...'>
            >>> array["x"]
            <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>
            >>> array["y"]
            <Array [17.4, -16.6, -16.6, ... 1.2, 1.2, 1.2] type='2304 * float64'>

        See also :ref:`uproot.behaviors.TBranch.HasBranches.arrays` to read
        multiple ``TBranches`` into a group of arrays or an array-group.
        """
        if interpretation is None:
            interpretation = self.interpretation
        else:
            interpretation = _regularize_interpretation(interpretation)

        entry_start, entry_stop = _regularize_entries_start_stop(
            self.num_entries, entry_start, entry_stop
        )
        decompression_executor, interpretation_executor = _regularize_executors(
            decompression_executor, interpretation_executor, self._file
        )
        array_cache = _regularize_array_cache(array_cache, self._file)
        library = uproot.interpretation.library._regularize_library(library)

        def get_from_cache(branchname, interpretation):
            if array_cache is not None:
                cache_key = f"{self.cache_key}:{branchname}:{interpretation.cache_key}:{entry_start}-{entry_stop}:{library.name}"
                return array_cache.get(cache_key)
            else:
                return None

        arrays = {}
        expression_context = []
        branchid_interpretation = {}
        _regularize_branchname(
            self,
            self.name,
            self,
            interpretation,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            True,
            False,
        )

        ranges_or_baskets = []
        checked = set()
        for _, context in expression_context:
            for branch in context["branches"]:
                if branch.cache_key not in checked and not isinstance(
                    branchid_interpretation[branch.cache_key],
                    uproot.interpretation.grouped.AsGrouped,
                ):
                    checked.add(branch.cache_key)
                    for (
                        basket_num,
                        range_or_basket,
                    ) in branch.entries_to_ranges_or_baskets(entry_start, entry_stop):
                        ranges_or_baskets.append((branch, basket_num, range_or_basket))

        interp_options = {"ak_add_doc": ak_add_doc}
        _ranges_or_baskets_to_arrays(
            self,
            ranges_or_baskets,
            branchid_interpretation,
            entry_start,
            entry_stop,
            decompression_executor,
            interpretation_executor,
            library,
            arrays,
            False,
            interp_options,
        )

        _fix_asgrouped(
            arrays,
            expression_context,
            branchid_interpretation,
            library,
            None,
            ak_add_doc,
        )

        if array_cache is not None:
            cache_key = f"{self.cache_key}:{self.name}:{interpretation.cache_key}:{entry_start}-{entry_stop}:{library.name}"
            array_cache[cache_key] = arrays[self.cache_key]

        return arrays[self.cache_key]

    def __array__(self, *args, **kwargs):
        out = self.array(library="np")
        if args == () and kwargs == {}:
            return out
        else:
            return numpy.array(out, *args, **kwargs)


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


def _regularize_entries_start_stop(num_entries, entry_start, entry_stop):
    if entry_start is None:
        entry_start = 0
    elif entry_start < 0:
        entry_start += num_entries
    entry_start = min(num_entries, max(0, entry_start))

    if entry_stop is None:
        entry_stop = num_entries
    elif entry_stop < 0:
        entry_stop += num_entries
    entry_stop = min(num_entries, max(0, entry_stop))

    if entry_stop < entry_start:
        entry_stop = entry_start

    return int(entry_start), int(entry_stop)


def _regularize_executors(decompression_executor, interpretation_executor, file):
    if file is None:
        if decompression_executor is None:
            decompression_executor = uproot.source.futures.TrivialExecutor()
        if interpretation_executor is None:
            interpretation_executor = uproot.source.futures.TrivialExecutor()
    else:
        if decompression_executor is None:
            decompression_executor = file.decompression_executor
        if interpretation_executor is None:
            interpretation_executor = file.interpretation_executor
    return decompression_executor, interpretation_executor


def _regularize_array_cache(array_cache, file):
    if isinstance(array_cache, MutableMapping):
        return array_cache
    elif isinstance(array_cache, str) and array_cache == "inherit":
        return file._array_cache
    elif array_cache is None:
        return None
    elif uproot._util.isint(array_cache) or isinstance(array_cache, str):
        return uproot.cache.LRUArrayCache(array_cache)
    else:
        raise TypeError("array_cache must be None, a MutableMapping, or a memory size")


def _regularize_aliases(hasbranches, aliases):
    if aliases is None:
        return hasbranches.aliases
    else:
        new_aliases = dict(hasbranches.aliases)
        new_aliases.update(aliases)
        return new_aliases


def _regularize_interpretation(interpretation):
    if isinstance(interpretation, uproot.interpretation.Interpretation):
        return interpretation
    elif isinstance(interpretation, numpy.dtype):
        return uproot.interpretation.numerical.AsDtype(interpretation)
    else:
        dtype = numpy.dtype(interpretation)
        dtype = dtype.newbyteorder(">")
        return uproot.interpretation.numerical.AsDtype(interpretation)


def _regularize_branchname(
    hasbranches,
    branchname,
    branch,
    interpretation,
    get_from_cache,
    arrays,
    expression_context,
    branchid_interpretation,
    is_primary,
    is_cut,
):
    got = get_from_cache(branchname, interpretation)
    if got is not None:
        arrays[branch.cache_key] = got

    is_jagged = isinstance(interpretation, uproot.interpretation.jagged.AsJagged)

    if isinstance(interpretation, uproot.interpretation.grouped.AsGrouped):
        branches = []
        for subname, subinterp in interpretation.subbranches.items():
            _regularize_branchname(
                hasbranches,
                subname,
                branch[subname],
                subinterp,
                get_from_cache,
                arrays,
                expression_context,
                branchid_interpretation,
                False,
                is_cut,
            )
            branches.extend(expression_context[-1][1]["branches"])

        branches.append(branch)
        arrays[branch.cache_key] = None

    else:
        branches = [branch]

    if branch.cache_key in branchid_interpretation:
        if (
            branchid_interpretation[branch.cache_key].cache_key
            != interpretation.cache_key
        ):
            raise ValueError(
                "a branch cannot be loaded with multiple interpretations: "
                f"{branchid_interpretation[branch.cache_key]!r} and {interpretation!r}"
            )
    else:
        branchid_interpretation[branch.cache_key] = interpretation

    c = {
        "is_primary": is_primary,
        "is_cut": is_cut,
        "is_jagged": is_jagged,
        "is_branch": True,
        "branches": branches,
    }
    expression_context.append((branchname, c))


def _regularize_expression(
    hasbranches,
    expression,
    keys,
    aliases,
    language,
    get_from_cache,
    arrays,
    expression_context,
    branchid_interpretation,
    symbol_path,
    is_cut,
    rename,
):
    is_primary = symbol_path == ()

    branch = hasbranches.get(expression)
    if branch is not None:
        _regularize_branchname(
            hasbranches,
            expression,
            branch,
            branch.interpretation,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            is_primary,
            is_cut,
        )

    else:
        # the value of `expression` is either what we want to compute or a lookup value for it
        to_compute = aliases.get(expression, expression)

        is_jagged = False
        expression_branches = []
        for symbol in language.free_symbols(
            to_compute,
            keys,
            aliases,
            hasbranches.file.file_path,
            hasbranches.object_path,
        ):
            if symbol in symbol_path:
                raise ValueError(
                    """symbol {} is recursively defined with aliases:

    {}

in file {} at {}""".format(
                        repr(symbol),
                        "\n    ".join(f"{k}: {v}" for k, v in aliases.items()),
                        hasbranches.file.file_path,
                        hasbranches.object_path,
                    )
                )

            _regularize_expression(
                hasbranches,
                symbol,
                keys,
                aliases,
                language,
                get_from_cache,
                arrays,
                expression_context,
                branchid_interpretation,
                (*symbol_path, symbol),
                False,
                None,
            )
            if expression_context[-1][1]["is_jagged"]:
                is_jagged = True
            expression_branches.extend(expression_context[-1][1]["branches"])

        c = {
            "is_primary": is_primary,
            "is_cut": is_cut,
            "is_jagged": is_jagged,
            "is_branch": False,
            "branches": expression_branches,
        }
        if rename is not None:
            c["rename"] = rename
        expression_context.append((expression, c))


def _regularize_expressions(
    hasbranches,
    expressions,
    cut,
    filter_name,
    filter_typename,
    filter_branch,
    keys,
    aliases,
    language,
    get_from_cache,
):
    arrays = {}
    expression_context = []
    branchid_interpretation = {}

    if expressions is None:
        for branchname, branch in hasbranches.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=True,
            full_paths=False,
        ):
            if not isinstance(
                branch.interpretation,
                (
                    uproot.interpretation.identify.UnknownInterpretation,
                    uproot.interpretation.grouped.AsGrouped,
                ),
            ):
                _regularize_expression(
                    hasbranches,
                    language.getter_of(branchname),
                    keys,
                    aliases,
                    language,
                    get_from_cache,
                    arrays,
                    expression_context,
                    branchid_interpretation,
                    (),
                    False,
                    branchname,
                )

    elif isinstance(expressions, str):
        _regularize_expression(
            hasbranches,
            expressions,
            keys,
            aliases,
            language,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            (),
            False,
            None,
        )

    elif isinstance(expressions, Iterable):
        if isinstance(expressions, dict):
            items = expressions.items()
        else:
            items = []
            for expression in expressions:
                if isinstance(expression, str):
                    items.append((expression, None))
                elif isinstance(expression, tuple) and len(expression) == 2:
                    items.append(expression)
                else:
                    raise TypeError(
                        "iterable of expressions must be strings or "
                        "name, Interpretation pairs (length-2 tuples), not "
                        + repr(expression)
                    )

        for expression, interp in items:
            if interp is None:
                _regularize_expression(
                    hasbranches,
                    expression,
                    keys,
                    aliases,
                    language,
                    get_from_cache,
                    arrays,
                    expression_context,
                    branchid_interpretation,
                    (),
                    False,
                    None,
                )
            else:
                branch = hasbranches[expression]
                interp = (  # noqa: PLW2901 (overwriting interp)
                    _regularize_interpretation(interp)
                )
                _regularize_branchname(
                    hasbranches,
                    expression,
                    branch,
                    interp,
                    get_from_cache,
                    arrays,
                    expression_context,
                    branchid_interpretation,
                    True,
                    False,
                )

    else:
        raise TypeError(
            "expressions must be None (for all branches), a string (single "
            "branch or expression), a list of strings (multiple), or a dict "
            "or list of name, Interpretation pairs (branch names and their "
            f"new Interpretation), not {expressions!r}"
        )

    if cut is None:
        pass
    elif isinstance(cut, str):
        _regularize_expression(
            hasbranches,
            cut,
            keys,
            aliases,
            language,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            (),
            True,
            None,
        )

    return arrays, expression_context, branchid_interpretation


_basket_arrays_lock = threading.Lock()


def _ranges_or_baskets_to_arrays(
    hasbranches,
    ranges_or_baskets,
    branchid_interpretation,
    entry_start,
    entry_stop,
    decompression_executor,
    interpretation_executor,
    library,
    arrays,
    update_ranges_or_baskets,
    interp_options,
):
    notifications = queue.Queue()

    branchid_arrays = {}
    branchid_num_baskets = {}
    ranges = []
    range_args = {}
    range_original_index = {}
    original_index = 0
    branchid_to_branch = {}

    for cache_key in branchid_interpretation:
        branchid_num_baskets[cache_key] = 0

    for branch, basket_num, range_or_basket in ranges_or_baskets:
        branchid_num_baskets[branch.cache_key] += 1

        if branch.cache_key not in branchid_arrays:
            branchid_arrays[branch.cache_key] = {}

        if isinstance(range_or_basket, tuple) and len(range_or_basket) == 2:
            range_or_basket = (  # noqa: PLW2901 (overwriting range_or_basket)
                int(range_or_basket[0]),
                int(range_or_basket[1]),
            )
            ranges.append(range_or_basket)
            range_args[range_or_basket] = (branch, basket_num)
            range_original_index[range_or_basket] = original_index
        else:
            notifications.put(range_or_basket)

        original_index += 1  # noqa: SIM113 (don't use `enumerate` for `original_index`)

        branchid_to_branch[branch.cache_key] = branch

    for cache_key, interpretation in branchid_interpretation.items():
        if branchid_num_baskets[cache_key] == 0 and cache_key not in arrays:
            arrays[cache_key] = interpretation.final_array(
                {}, 0, 0, [0], library, None, interp_options
            )

        # check for CannotBeAwkward errors on the main thread before reading any data
        if (
            isinstance(library, uproot.interpretation.library.Awkward)
            and isinstance(interpretation, uproot.interpretation.objects.AsObjects)
            and cache_key in branchid_to_branch
        ):
            branchid_to_branch[cache_key]._awkward_check(interpretation)

    hasbranches._file.source.chunks(ranges, notifications=notifications)

    def replace(ranges_or_baskets, original_index, basket):
        branch, basket_num, range_or_basket = ranges_or_baskets[original_index]
        ranges_or_baskets[original_index] = branch, basket_num, basket

    def chunk_to_basket(chunk, branch, basket_num):
        try:
            cursor = uproot.source.cursor.Cursor(chunk.start)
            basket = uproot.models.TBasket.Model_TBasket.read(
                chunk,
                cursor,
                {"basket_num": basket_num},
                hasbranches._file,
                hasbranches._file,
                branch,
            )
            original_index = range_original_index[(chunk.start, chunk.stop)]
            if update_ranges_or_baskets:
                replace(ranges_or_baskets, original_index, basket)
        except Exception:
            notifications.put(sys.exc_info())
        else:
            notifications.put(basket)

    forth_context = {x: threading.local() for x in branchid_interpretation}

    def basket_to_array(basket):
        try:
            assert basket.basket_num is not None
            branch = basket.parent
            interpretation = branchid_interpretation[branch.cache_key]
            basket_arrays = branchid_arrays[branch.cache_key]

            context = dict(branch.context)
            context["forth"] = forth_context[branch.cache_key]

            basket_array = interpretation.basket_array(
                basket.data,
                basket.byte_offsets,
                basket,
                branch,
                context,
                basket.member("fKeylen"),
                library,
                interp_options,
            )
            if basket.num_entries != len(basket_array):
                raise ValueError(
                    f"""basket {basket.basket_num} in tree/branch {branch.object_path} has the wrong number of entries """
                    f"""(expected {basket.num_entries}, obtained {len(basket_array)}) when interpreted as {interpretation}
    in file {branch.file.file_path}"""
                )

            basket_num = basket.basket_num
            basket = None

            with _basket_arrays_lock:
                basket_arrays[basket_num] = basket_array
                len_basket_arrays = len(basket_arrays)

            if len_basket_arrays == branchid_num_baskets[branch.cache_key]:
                arrays[branch.cache_key] = interpretation.final_array(
                    basket_arrays,
                    entry_start,
                    entry_stop,
                    branch.entry_offsets,
                    library,
                    branch,
                    interp_options,
                )
                with _basket_arrays_lock:
                    # no longer needed, save memory
                    basket_arrays.clear()

        except Exception:
            notifications.put(sys.exc_info())
        else:
            notifications.put(None)

    while len(arrays) < len(branchid_interpretation):
        obj = notifications.get()

        if isinstance(obj, uproot.source.chunk.Chunk):
            args = range_args[(obj.start, obj.stop)]
            decompression_executor.submit(chunk_to_basket, obj, *args)

        elif isinstance(obj, uproot.models.TBasket.Model_TBasket):
            interpretation_executor.submit(basket_to_array, obj)

        elif obj is None:
            pass

        elif isinstance(obj, tuple) and len(obj) == 3:
            uproot.source.futures.delayed_raise(*obj)

        else:
            raise AssertionError(obj)

        obj = None  # release before blocking


def _fix_asgrouped(
    arrays, expression_context, branchid_interpretation, library, how, ak_add_doc
):
    index_start = 0
    for index_stop, (_, context) in enumerate(expression_context):
        if context["is_branch"]:
            branch = context["branches"][-1]
            interpretation = branchid_interpretation[branch.cache_key]
            if isinstance(interpretation, uproot.interpretation.grouped.AsGrouped):
                assert arrays[branch.cache_key] is None

                limited_context = dict(expression_context[index_start:index_stop])

                subarrays = {}
                subcontext = []
                for subname in interpretation.subbranches:
                    subbranch = branch[subname]
                    subarrays[subname] = arrays[subbranch.cache_key]
                    subcontext.append((subname, limited_context[subname]))

                arrays[branch.cache_key] = _ak_add_doc(
                    library.group(subarrays, subcontext, how), branch, ak_add_doc
                )

                index_start = index_stop


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
