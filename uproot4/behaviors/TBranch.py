# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import re
import sys
import threading

try:
    from collections.abc import Mapping
    from collections.abc import MutableMapping
    from collections.abc import Iterable
except ImportError:
    from collections import Mapping
    from collections import MutableMapping
    from collections import Iterable
try:
    import queue
except ImportError:
    import Queue as queue

import uproot4.source.cursor
import uproot4.interpretation
import uproot4.interpretation.numerical
import uproot4.interpretation.jagged
import uproot4.interpretation.library
import uproot4.interpretation.identify
import uproot4.reading
import uproot4.compute.python
import uproot4.models.TBasket
import uproot4.models.TObjArray
import uproot4._util
from uproot4._util import no_filter


def _get_recursive(hasbranches, where):
    for branch in hasbranches.branches:
        if branch.name == where:
            return branch
        got = _get_recursive(branch, where)
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


def _regularize_executors(decompression_executor, interpretation_executor):
    if decompression_executor is None:
        decompression_executor = uproot4.decompression_executor
    if interpretation_executor is None:
        interpretation_executor = uproot4.interpretation_executor
    return decompression_executor, interpretation_executor


def _regularize_array_cache(array_cache, file):
    if isinstance(array_cache, MutableMapping):
        return array_cache
    elif array_cache is None:
        return file._array_cache
    else:
        raise TypeError("array_cache must be None or a MutableMapping")


def _regularize_interpretation(interpretation):
    if isinstance(interpretation, uproot4.interpretation.Interpretation):
        return interpretation
    elif isinstance(interpretation, numpy.dtype):
        return uproot4.interpretation.numerical.AsDtype(interpretation)
    else:
        dtype = numpy.dtype(interpretation)
        dtype = dtype.newbyteorder(">")
        return uproot4.interpretation.numerical.AsDtype(interpretation)


_expression_is_branch = re.compile(
    r"^\s*[A-Za-z_][A-Za-z_0-9]*(\s*\.\s*[A-Za-z_][A-Za-z_0-9]*)*\s*$"
)
_expression_strip_dot_whitespace = re.compile(r"\s*\.\s*")



# def _regularize_name(
#     hasbranches,
#     expression,
#     value,
#     filter_name,
#     filter_typename,
#     filter_branch,
#     aliases,
#     functions,
#     name_interp_branch,
#     original_jagged,
#     symbol_path,
# ):
#     if symbol_path is not None:
#         if aliases is not None and expression in aliases:
#             aliased_expression = aliases[expression]
#         else:
#             aliased_expression = expression

#         if _expression_is_branch.match(aliased_expression):
#             name = _expression_strip_dot_whitespace.sub(".", aliased_expression.strip())
#             try:
#                 branch = hasbranches[name]
#             except KeyError:
#                 raise uproot4.KeyInFileError(
#                     name, hasbranches.file.file_path, object_path=hasbranches.object_path
#                 )
#             else:
#                 if value is None:
#                     value = branch.interpretation
#                 is_jagged = isinstance(value, uproot4.interpretation.jagged.AsJagged)
#                 name_interp_branch.append((expression, value, branch))
#                 original_jagged.append((expression, is_jagged))

#         else:
#             start = len(name_interp_branch)
#             for symbol in uproot4._util.free_symbols(
#                 aliased_expression,
#                 functions,
#                 hasbranches.file.file_path,
#                 hasbranches.object_path,
#             ):
#                 if symbol in symbol_path:
#                     raise ValueError(
#                         """symbol {0} is recursively defined with aliases:

#     {1}

# in file {2} at {3}""".format(
#                         repr(symbol),
#                         "\n    ".join("{0}: {1}".format(k, v) for k, v in aliases.items()),
#                         hasbranches.file.file_path,
#                         hasbranches.object_path,
#                     )
#                 )

#                 _regularize_name(
#                     hasbranches,
#                     symbol,
#                     value,
#                     filter_name,
#                     filter_typename,
#                     filter_branch,
#                     aliases,
#                     functions,
#                     name_interp_branch,
#                     [],
#                     symbol_path + (symbol,),
#                 )

#             is_jagged = any(
#                 isinstance(interp, uproot4.interpretation.jagged.AsJagged)
#                 for _, interp, _ in name_interp_branch[start:]
#             )
#             original_jagged.append((expression, is_jagged))

#     else:
#         try:
#             branch = hasbranches[expression]
#         except KeyError:
#             raise uproot4.KeyInFileError(
#                 expression, hasbranches.file.file_path, object_path=hasbranches.object_path
#             )
#         else:
#             if value is None:
#                 value = branch.interpretation
#             is_jagged = isinstance(value, uproot4.interpretation.jagged.AsJagged)
#             name_interp_branch.append((expression, value, branch))
#             original_jagged.append((expression, is_jagged))


# def _regularize_names(
#     hasbranches, names, filter_name, filter_typename, filter_branch, aliases, functions
# ):
#     filter_name = uproot4._util.regularize_filter(filter_name)
#     filter_typename = uproot4._util.regularize_filter(filter_typename)
#     if filter_branch is None:
#         filter_branch = no_filter
#     elif callable(filter_branch):
#         pass
#     else:
#         raise TypeError(
#             "filter_branch must be None or a function: TBranch -> bool, not {0}".format(
#                 repr(filter_branch)
#             )
#         )

#     if aliases is None:
#         aliases = hasbranches.aliases
#     else:
#         new_aliases = dict(hasbranches.aliases)
#         new_aliases.update(aliases)
#         aliases = new_aliases

#     name_interp_branch, original_jagged = [], []

#     if names is None:
#         for name, branch in hasbranches.iteritems(recursive=True):
#             _regularize_name(
#                 hasbranches,
#                 name,
#                 branch.interpretation,
#                 filter_name,
#                 filter_typename,
#                 filter_branch,
#                 aliases,
#                 functions,
#                 name_interp_branch,
#                 original_jagged,
#                 None,
#             )

#     elif uproot4._util.isstr(names):
#         _regularize_name(
#             hasbranches,
#             names,
#             None,
#             filter_name,
#             filter_typename,
#             filter_branch,
#             aliases,
#             functions,
#             name_interp_branch,
#             original_jagged,
#             (),
#         )

#     elif isinstance(names, dict):
#         for name, value in names.items():
#             if uproot4._util.isstr(name):
#                 _regularize_name(
#                     hasbranches,
#                     name,
#                     _regularize_interpretation(value),
#                     filter_name,
#                     filter_typename,
#                     filter_branch,
#                     aliases,
#                     functions,
#                     name_interp_branch,
#                     original_jagged,
#                     None,
#                 )
#             else:
#                 raise TypeError(
#                     "keys of a {{name: Interpretation}} dict must be "
#                     "strings, not {0}".format(repr(original_key))
#                 )

#     elif isinstance(names, Iterable):
#         for item in names:
#             if uproot4._util.isstr(item):
#                 _regularize_name(
#                     hasbranches,
#                     item,
#                     None,
#                     filter_name,
#                     filter_typename,
#                     filter_branch,
#                     aliases,
#                     functions,
#                     name_interp_branch,
#                     original_jagged,
#                     (),
#                 )
#             elif isinstance(item, tuple) and len(item) == 2 and uproot4._util.isstr(item[0]):
#                 _regularize_name(
#                     hasbranches,
#                     item[0],
#                     _regularize_interpretation(item[1]),
#                     filter_name,
#                     filter_typename,
#                     filter_branch,
#                     aliases,
#                     functions,
#                     name_interp_branch,
#                     original_jagged,
#                     None,
#                 )
#             else:
#                 raise TypeError(
#                     "items in a names list must be strings or "
#                     "(name, Interpretation) pairs, not {0}".format(
#                         repr(original_key)
#                     )
#                 )

#     else:
#         raise TypeError(
#             "a names list must be a string, a list of strings, or a "
#             "{{name: Interpretation}} dict, not {0}".format(repr(names))
#         )

#     return name_interp_branch, original_jagged


def _regularize_branchname(
    hasbranches,
    branchname,
    branch,
    interp,
    get_from_cache,
    arrays,
    expression_context,
    name_branch_interp,
):
    got = get_from_cache(branchname, interp)
    if got is not None:
        arrays[id(branch)] = got

    expression_context.append((branchname, {"is_jagged": isinstance(interp, uproot4.interpretation.jagged.AsJagged)}))

    name_branch_interp.append((branchname, branch, interp))


def _regularize_expression(
    hasbranches,
    expressions,
    filter_name,
    filter_typename,
    filter_branch,
    aliases,
    compute,
    get_from_cache,
    arrays,
    expression_context,
    name_branch_interp,
):
    raise NotImplementedError


def _regularize_expressions(
    hasbranches,
    expressions,
    filter_name,
    filter_typename,
    filter_branch,
    aliases,
    compute,
    get_from_cache,
):
    filter_name = uproot4._util.regularize_filter(filter_name)
    filter_typename = uproot4._util.regularize_filter(filter_typename)
    if filter_branch is None:
        filter_branch = no_filter
    elif not callable(filter_branch):
        raise TypeError(
            "filter_branch must be None or a function: TBranch -> bool, not {0}".format(
                repr(filter_branch)
            )
        )
    if aliases is None:
        aliases = hasbranches.aliases
    else:
        new_aliases = dict(hasbranches.aliases)
        new_aliases.update(aliases)
        aliases = new_aliases

    arrays = {}
    expression_context = []
    name_branch_interp = []

    if expressions is None:
        for branchname, branch in hasbranches.iteritems(
            recursive=True,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
        ):
            _regularize_branchname(
                hasbranches,
                branchname,
                branch,
                branch.interpretation,
                get_from_cache,
                arrays,
                expression_context,
                name_branch_interp,
            )

    elif uproot4._util.isstr(expressions):
        _regularize_expression(
            hasbranches,
            expressions,
            filter_name,
            filter_typename,
            filter_branch,
            aliases,
            compute,
            get_from_cache,
            arrays,
            expression_context,
            name_branch_interp,
        )

    elif isinstance(expressions, Iterable):
        if isinstance(expressions, dict):
            items = expressions.items()
        else:
            items = []
            for expression in expressions:
                if uproot4._util.isstr(expression):
                    items.append((expression, None))
                elif isinstance(expression, tuple) and len(tuple) == 2:
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
                    filter_name,
                    filter_typename,
                    filter_branch,
                    aliases,
                    compute,
                    get_from_cache,
                    arrays,
                    expression_context,
                    name_branch_interp,
                )
            else:
                branch = hasbranches[expression]
                if (filter_name(expression) and
                    filter_typename(branch.typename) and
                    filter_branch(branch)
                ):
                    interp = _regularize_interpretation(interp)
                    _regularize_branchname(
                        hasbranches,
                        expression,
                        branch,
                        interp,
                        get_from_cache,
                        arrays,
                        expression_context,
                        name_branch_interp,
                    )

    else:
        raise TypeError(
            "expressions must be None (for all branches), a string (single "
            "branch or expression), a list of strings (multiple), or a dict "
            "or list of name, Interpretation pairs (branch names and their "
            "new Interpretation), not {0}".format(repr(expressions))
        )

    return arrays, expression_context, name_branch_interp


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
):
    notifications = queue.Queue()

    branchid_arrays = {}
    branchid_num_baskets = {}
    ranges = []
    range_args = {}
    range_original_index = {}
    original_index = 0

    for branch, basket_num, range_or_basket in ranges_or_baskets:
        if id(branch) not in branchid_arrays:
            branchid_arrays[id(branch)] = {}
            branchid_num_baskets[id(branch)] = 0
        branchid_num_baskets[id(branch)] += 1

        if isinstance(range_or_basket, tuple) and len(range_or_basket) == 2:
            ranges.append(range_or_basket)
            range_args[range_or_basket] = (branch, basket_num)
            range_original_index[range_or_basket] = original_index
        else:
            notifications.put(range_or_basket)

        original_index += 1

    hasbranches._file.source.chunks(ranges, notifications=notifications)

    def replace(ranges_or_baskets, original_index, basket):
        branch, basket_num, range_or_basket = ranges_or_baskets[
            original_index
        ]
        ranges_or_baskets[original_index] = branch, basket_num, basket

    def chunk_to_basket(chunk, branch, basket_num):
        try:
            cursor = uproot4.source.cursor.Cursor(chunk.start)
            basket = uproot4.models.TBasket.Model_TBasket.read(
                chunk, cursor, {"basket_num": basket_num}, hasbranches._file, branch
            )
            original_index = range_original_index[(chunk.start, chunk.stop)]
            replace(ranges_or_baskets, original_index, basket)
        except Exception:
            notifications.put(sys.exc_info())
        else:
            notifications.put(basket)

    def basket_to_array(basket):
        try:
            assert basket.basket_num is not None
            branch = basket.parent
            interpretation = branchid_interpretation[id(branch)]
            basket_arrays = branchid_arrays[id(branch)]
            basket_arrays[basket.basket_num] = interpretation.basket_array(
                basket.data, basket.byte_offsets, basket, branch
            )
            if len(basket_arrays) == branchid_num_baskets[id(branch)]:
                arrays[id(branch)] = interpretation.final_array(
                    basket_arrays,
                    entry_start,
                    entry_stop,
                    branch.entry_offsets,
                    library,
                    branch,
                )
        except Exception:
            notifications.put(sys.exc_info())
        else:
            notifications.put(None)

    while len(arrays) < len(branchid_interpretation):
        try:
            obj = notifications.get(timeout=0.001)
        except queue.Empty:
            continue

        if isinstance(obj, uproot4.source.chunk.Chunk):
            chunk = obj
            args = range_args[(chunk.start, chunk.stop)]
            decompression_executor.submit(chunk_to_basket, chunk, *args)

        elif isinstance(obj, uproot4.models.TBasket.Model_TBasket):
            basket = obj
            interpretation_executor.submit(basket_to_array, basket)

        elif obj is None:
            pass

        elif isinstance(obj, tuple) and len(obj) == 3:
            uproot4.source.futures.delayed_raise(*obj)

        else:
            raise AssertionError(obj)


# def _compute_expressions(
#     arrays,
#     original_jagged,
#     aliases,
#     functions,
#     file_path,
#     object_path,
# ):
#     print("arrays", list(arrays))

#     scope = {
#         "arrays": arrays,
#         "aliases": {},
#         "functions": functions,
#     }
#     for k, v in aliases.items():
#         print("alias", k)
#         scope["aliases"][k] = uproot4._util.branch_expression(
#             v,
#             aliases,
#             functions,
#             scope,
#             file_path,
#             object_path,
#         )

#     output = {}

#     for original_key, is_jagged in original_jagged:
#         print("expression", original_key)

#         expression = uproot4._util.branch_expression(
#             original_key,
#             aliases,
#             functions,
#             scope,
#             file_path,
#             object_path,
#         )
#         output[original_key] = expression()

#     return output


class HasBranches(Mapping):
    @property
    def branches(self):
        return self.member("fBranches")

    def __getitem__(self, where):
        original_where = where

        got = self._lookup.get(original_where)
        if got is not None:
            return got

        if uproot4._util.isint(where):
            return self.branches[where]
        elif uproot4._util.isstr(where):
            where = uproot4._util.ensure_str(where)
        else:
            raise TypeError(
                "where must be an integer or a string, not {0}".format(repr(where))
            )

        if where.startswith("/"):
            recursive = False
            where = where[1:]
        else:
            recursive = True

        if "/" in where:
            where = "/".join([x for x in where.split("/") if x != ""])
            for k, v in self.iteritems(recursive=True):
                if where == k:
                    self._lookup[original_where] = v
                    return v
            else:
                raise uproot4.KeyInFileError(original_where, self._file.file_path)

        elif recursive:
            got = _get_recursive(self, where)
            if got is not None:
                self._lookup[original_where] = got
                return got
            else:
                raise uproot4.KeyInFileError(original_where, self._file.file_path)

        else:
            for branch in self.branches:
                if branch.name == where:
                    self._lookup[original_where] = branch
                    return branch
            else:
                raise uproot4.KeyInFileError(original_where, self._file.file_path)

    def iteritems(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
    ):
        filter_name = uproot4._util.regularize_filter(filter_name)
        filter_typename = uproot4._util.regularize_filter(filter_typename)
        if filter_branch is None:
            filter_branch = no_filter
        elif callable(filter_branch):
            pass
        else:
            raise TypeError(
                "filter_branch must be None or a function: TBranch -> bool, not {0}".format(
                    repr(filter_branch)
                )
            )
        for branch in self.branches:
            if (
                filter_name(branch.name)
                and filter_typename(branch.typename)
                and filter_branch(branch)
            ):
                yield branch.name, branch

            if recursive:
                for k1, v in branch.iteritems(
                    recursive=recursive,
                    filter_name=no_filter,
                    filter_typename=filter_typename,
                    filter_branch=filter_branch,
                ):
                    k2 = "{0}/{1}".format(branch.name, k1)
                    if filter_name(k2):
                        yield k2, v

    def items(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
    ):
        return list(
            self.iteritems(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
            )
        )

    def iterkeys(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
    ):
        for k, v in self.iteritems(
            recursive=recursive,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
        ):
            yield k

    def keys(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
    ):
        return list(
            self.iterkeys(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
            )
        )

    def _ipython_key_completions_(self):
        "Support key-completion in an IPython or Jupyter kernel."
        return self.iterkeys()

    def itervalues(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
    ):
        for k, v in self.iteritems(
            recursive=recursive,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
        ):
            yield v

    def values(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
    ):
        return list(
            self.itervalues(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
            )
        )

    def itertypenames(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
    ):
        for k, v in self.iteritems(
            recursive=recursive,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
        ):
            yield k, v.typename

    def typenames(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
    ):
        return dict(
            self.itertypenames(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
            )
        )

    def __iter__(self):
        for x in self.branches:
            yield x

    def __len__(self):
        return len(self.branches)

    # def arrays(
    #     self,
    #     names=None,
    #     entry_start=None,
    #     entry_stop=None,
    #     filter_name=no_filter,
    #     filter_typename=no_filter,
    #     filter_branch=no_filter,
    #     aliases=None,
    #     functions=None,
    #     decompression_executor=None,
    #     interpretation_executor=None,
    #     array_cache=None,
    #     library="ak",
    #     how=None,
    # ):
    #     if functions is None:
    #         functions = {}
    #     name_interp_branch, original_jagged = _regularize_names(
    #         self, names, filter_name, filter_typename, filter_branch, aliases, functions
    #     )
    #     branchid_interpretation = {}
    #     for name, interp, branch in name_interp_branch:
    #         branchid_interpretation[id(branch)] = interp

    #     entry_start, entry_stop = _regularize_entries_start_stop(
    #         self.tree.num_entries, entry_start, entry_stop
    #     )
    #     decompression_executor, interpretation_executor = _regularize_executors(
    #         decompression_executor, interpretation_executor
    #     )
    #     array_cache = _regularize_array_cache(array_cache, self._file)
    #     library = uproot4.interpretation.library._regularize_library(library)

    #     output = {}

    #     branches_seen = set()
    #     name_interp_branch_toget = []
    #     if array_cache is not None:
    #         for name, interp, branch in name_interp_branch:
    #             if id(branch) not in branches_seen:
    #                 branches_seen.add(id(branch))
    #                 cache_key = "{0}:{1}:{2}-{3}:{4}".format(
    #                     branch.cache_key,
    #                     interp.cache_key,
    #                     entry_start,
    #                     entry_stop,
    #                     library.name,
    #                 )
    #                 got = array_cache.get(cache_key)
    #                 if got is None:
    #                     name_interp_branch_toget.append((name, interp, branch))
    #                 else:
    #                     output[name] = got

    #     else:
    #         for name, interp, branch in name_interp_branch:
    #             if id(branch) not in branches_seen:
    #                 branches_seen.add(id(branch))
    #                 name_interp_branch_toget.append((name, interp, branch))

    #     ranges_or_baskets = []
    #     for name, interp, branch in name_interp_branch_toget:
    #         for basket_num, range_or_basket in branch.entries_to_ranges_or_baskets(
    #             entry_start, entry_stop
    #         ):
    #             ranges_or_baskets.append((name, branch, basket_num, range_or_basket))

    #     self._ranges_or_baskets_to_arrays(
    #         ranges_or_baskets,
    #         branchid_interpretation,
    #         entry_start,
    #         entry_stop,
    #         decompression_executor,
    #         interpretation_executor,
    #         library,
    #         output,
    #     )

    #     if array_cache is not None:
    #         for name, interp, branch in name_interp_branch_toget:
    #             cache_key = "{0}:{1}:{2}-{3}:{4}".format(
    #                 branch.cache_key,
    #                 interp.cache_key,
    #                 entry_start,
    #                 entry_stop,
    #                 library.name,
    #             )
    #             array_cache[cache_key] = output[name]

    #     if functions is not None:
    #         output = _compute_expressions(
    #             output,
    #             original_jagged,
    #             aliases,
    #             functions,
    #             self.file.file_path,
    #             self.object_path,
    #         )

    #     return library.group(output, original_jagged, how)

    def arrays(
        self,
        expressions=None,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        aliases=None,
        compute=uproot4.compute.python.ComputePython(),
        entry_start=None,
        entry_stop=None,
        decompression_executor=None,
        interpretation_executor=None,
        array_cache=None,
        library="ak",
        how=None,
    ):
        entry_start, entry_stop = _regularize_entries_start_stop(
            self.tree.num_entries, entry_start, entry_stop
        )
        decompression_executor, interpretation_executor = _regularize_executors(
            decompression_executor, interpretation_executor
        )
        array_cache = _regularize_array_cache(array_cache, self._file)
        library = uproot4.interpretation.library._regularize_library(library)

        def get_from_cache(branchname, interp):
            cache_key = "{0}:{1}:{2}-{3}:{4}".format(
                self.cache_key,
                interp.cache_key,
                entry_start,
                entry_stop,
                library.name,
            )
            return array_cache.get(cache_key)

        arrays, expression_context, name_branch_interp = _regularize_expressions(
            self,
            expressions,
            filter_name,
            filter_typename,
            filter_branch,
            aliases,
            compute,
            get_from_cache,
        )

        raise NotImplementedError


class TBranch(HasBranches):
    def postprocess(self, chunk, cursor, context):
        fWriteBasket = self.member("fWriteBasket")

        self._lookup = {}
        self._interpretation = None
        self._count_branch = None
        self._count_leaf = None
        self._streamer = None

        self._num_normal_baskets = 0
        for i, x in enumerate(self.member("fBasketSeek")):
            if x == 0 or i == fWriteBasket:
                break
            self._num_normal_baskets += 1

        if (
            self.member("fEntries")
            == self.member("fBasketEntry")[self._num_normal_baskets]
        ):
            self._embedded_baskets = []
            self._embedded_baskets_lock = None

        elif self.has_member("fBaskets"):
            self._embedded_baskets = []
            for basket in self.member("fBaskets"):
                if basket is not None:
                    basket._basket_num = self._num_normal_baskets + len(
                        self._embedded_baskets
                    )
                    self._embedded_baskets.append(basket)
            self._embedded_baskets_lock = None

        else:
            self._embedded_baskets = None
            self._embedded_baskets_lock = threading.Lock()

        if "fIOFeatures" in self._parent.members:
            self._tree_iofeatures = self._parent.member("fIOFeatures").member("fIOBits")

        return self

    @property
    def tree(self):
        import uproot4.behaviors.TTree

        out = self
        while not isinstance(out, uproot4.behaviors.TTree.TTree):
            out = out.parent
        return out

    @property
    def aliases(self):
        return self.tree.aliases

    @property
    def cache_key(self):
        if isinstance(self._parent, uproot4.behaviors.TTree.TTree):
            return self.parent.cache_key + ":" + self.name
        else:
            return self.parent.cache_key + "/" + self.name

    @property
    def object_path(self):
        if isinstance(self._parent, uproot4.behaviors.TTree.TTree):
            return self.parent.object_path + ":" + self.name
        else:
            return self.parent.object_path + "/" + self.name

    @property
    def entry_offsets(self):
        if self._num_normal_baskets == 0:
            out = [0]
        else:
            out = self.member("fBasketEntry")[: self._num_normal_baskets + 1].tolist()
        num_entries_normal = out[-1]

        for basket in self.embedded_baskets:
            out.append(out[-1] + basket.num_entries)

        if out[-1] != self.num_entries and self.interpretation is not None:
            raise ValueError(
                """entries in normal baskets ({0}) plus embedded baskets ({1}) """
                """don't add up to expected number of entries ({2})
in file {3}""".format(
                    num_entries_normal,
                    sum(basket.num_entries for basket in self.embedded_baskets),
                    self.num_entries,
                    self._file.file_path,
                )
            )
        else:
            return out

    @property
    def embedded_baskets(self):
        if self._embedded_baskets is None:
            cursor = self._cursor_baskets.copy()
            baskets = uproot4.models.TObjArray.Model_TObjArrayOfTBaskets.read(
                self.tree.chunk, cursor, {}, self._file, self
            )
            with self._embedded_baskets_lock:
                self._embedded_baskets = []
                for basket in baskets:
                    if basket is not None:
                        basket._basket_num = self._num_normal_baskets + len(
                            self._embedded_baskets
                        )
                        self._embedded_baskets.append(basket)

        return self._embedded_baskets

    @property
    def name(self):
        return self.member("fName")

    @property
    def title(self):
        return self.member("fTitle")

    @property
    def typename(self):
        if self._streamer is not None:
            return self._streamer.typename

        def leaf_to_typename(leaf):
            if leaf.classname == "TLeafElement":
                return "???"
            else:
                return "{0}/{1}".format(leaf.member("fTitle"), leaf.classname[-1])

        if len(self.member("fLeaves")) == 1:
            return leaf_to_typename(self.member("fLeaves")[0])
        else:
            leaf_list = [leaf_to_typename(leaf) for leaf in self.member("fLeaves")]
            return ":".join(leaf_list)

    @property
    def streamer(self):
        return self._streamer

    @property
    def interpretation(self):
        if self._interpretation is None:
            self._interpretation = uproot4.interpretation.identify.interpretation_of(
                self, {}
            )
        return self._interpretation

    @property
    def count_branch(self):
        if self._count_branch is None:
            raise NotImplementedError
        return self._count_branch

    @property
    def count_leaf(self):
        if self._count_leaf is None:
            raise NotImplementedError
        return self._count_leaf

    @property
    def num_entries(self):
        return int(self.member("fEntries"))  # or fEntryNumber?

    @property
    def num_baskets(self):
        return self._num_normal_baskets + len(self.embedded_baskets)

    def __repr__(self):
        if len(self) == 0:
            return "<TBranch {0} at 0x{1:012x}>".format(repr(self.name), id(self))
        else:
            return "<TBranch {0} ({1} subbranches) at 0x{2:012x}>".format(
                repr(self.name), len(self), id(self)
            )

    def basket_chunk_bytes(self, basket_num):
        if 0 <= basket_num < self._num_normal_baskets:
            return int(self.member("fBasketBytes")[basket_num])
        elif 0 <= basket_num < self.num_baskets:
            raise IndexError(
                """branch {0} has {1} normal baskets; cannot get """
                """basket chunk {2} because only normal baskets have chunks
in file {3}""".format(
                    repr(self.name),
                    self._num_normal_baskets,
                    basket_num,
                    self._file.file_path,
                )
            )
        else:
            raise IndexError(
                """branch {0} has {1} baskets; cannot get basket chunk {2}
in file {3}""".format(
                    repr(self.name), self.num_baskets, basket_num, self._file.file_path
                )
            )

    def basket_chunk_cursor(self, basket_num):
        if 0 <= basket_num < self._num_normal_baskets:
            start = self.member("fBasketSeek")[basket_num]
            stop = start + self.basket_chunk_bytes(basket_num)
            cursor = uproot4.source.cursor.Cursor(start)
            chunk = self._file.source.chunk(start, stop)
            return chunk, cursor
        elif 0 <= basket_num < self.num_baskets:
            raise IndexError(
                """branch {0} has {1} normal baskets; cannot get chunk and """
                """cursor for basket {2} because only normal baskets have cursors
in file {3}""".format(
                    repr(self.name),
                    self._num_normal_baskets,
                    basket_num,
                    self._file.file_path,
                )
            )
        else:
            raise IndexError(
                """branch {0} has {1} baskets; cannot get cursor and chunk """
                """for basket {2}
in file {3}""".format(
                    repr(self.name), self.num_baskets, basket_num, self._file.file_path
                )
            )

    def basket_key(self, basket_num):
        start = self.member("fBasketSeek")[basket_num]
        stop = start + uproot4.reading.ReadOnlyKey._format_big.size
        cursor = uproot4.source.cursor.Cursor(start)
        chunk = self._file.source.chunk(start, stop)
        return uproot4.reading.ReadOnlyKey(
            chunk, cursor, {}, self._file, self, read_strings=False
        )

    def basket(self, basket_num):
        if 0 <= basket_num < self._num_normal_baskets:
            chunk, cursor = self.basket_chunk_cursor(basket_num)
            return uproot4.models.TBasket.Model_TBasket.read(
                chunk, cursor, {"basket_num": basket_num}, self._file, self
            )
        elif 0 <= basket_num < self.num_baskets:
            return self.embedded_baskets[basket_num - self._num_normal_baskets]
        else:
            raise IndexError(
                """branch {0} has {1} baskets; cannot get basket {2}
in file {3}""".format(
                    repr(self.name), self.num_baskets, basket_num, self._file.file_path
                )
            )

    def entries_to_ranges_or_baskets(self, entry_start, entry_stop):
        entry_offsets = self.entry_offsets
        out = []
        start = entry_offsets[0]
        for basket_num, stop in enumerate(entry_offsets[1:]):
            if entry_start < stop and start <= entry_stop:
                if 0 <= basket_num < self._num_normal_baskets:
                    byte_start = self.member("fBasketSeek")[basket_num]
                    byte_stop = byte_start + self.basket_chunk_bytes(basket_num)
                    out.append((basket_num, (byte_start, byte_stop)))
                elif 0 <= basket_num < self.num_baskets:
                    out.append((basket_num, self.basket(basket_num)))
                else:
                    raise AssertionError((self.name, basket_num))
            start = stop
        return out

    def array(
        self,
        interpretation=None,
        entry_start=None,
        entry_stop=None,
        decompression_executor=None,
        interpretation_executor=None,
        array_cache=None,
        library="ak",
    ):
        if interpretation is None:
            interpretation = self.interpretation
        branchid_interpretation = {id(self): interpretation}

        entry_start, entry_stop = _regularize_entries_start_stop(
            self.num_entries, entry_start, entry_stop
        )
        decompression_executor, interpretation_executor = _regularize_executors(
            decompression_executor, interpretation_executor
        )
        array_cache = _regularize_array_cache(array_cache, self._file)
        library = uproot4.interpretation.library._regularize_library(library)

        cache_key = "{0}:{1}:{2}-{3}:{4}".format(
            self.cache_key,
            interpretation.cache_key,
            entry_start,
            entry_stop,
            library.name,
        )
        if array_cache is not None:
            got = array_cache.get(cache_key)
            if got is not None:
                return got

        ranges_or_baskets = []
        for basket_num, range_or_basket in self.entries_to_ranges_or_baskets(
            entry_start, entry_stop
        ):
            ranges_or_baskets.append((self, basket_num, range_or_basket))

        arrays = {}
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
        )

        if array_cache is not None:
            array_cache[cache_key] = arrays[id(self)]

        return arrays[id(self)]
