# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import re
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

import numpy

import uproot4.source.cursor
import uproot4.streamers
import uproot4.stl_containers
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


def _regularize_aliases(hasbranches, aliases):
    if aliases is None:
        return hasbranches.aliases
    else:
        new_aliases = dict(hasbranches.aliases)
        new_aliases.update(aliases)
        return new_aliases


def _regularize_interpretation(interpretation):
    if isinstance(interpretation, uproot4.interpretation.Interpretation):
        return interpretation
    elif isinstance(interpretation, numpy.dtype):
        return uproot4.interpretation.numerical.AsDtype(interpretation)
    else:
        dtype = numpy.dtype(interpretation)
        dtype = dtype.newbyteorder(">")
        return uproot4.interpretation.numerical.AsDtype(interpretation)


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
        arrays[id(branch)] = got

    is_jagged = isinstance(interpretation, uproot4.interpretation.jagged.AsJagged)

    if id(branch) in branchid_interpretation:
        if branchid_interpretation[id(branch)].cache_key != interpretation.cache_key:
            raise ValueError(
                "a branch cannot be loaded with multiple interpretations: "
                "{0} and {1}".format(
                    repr(branchid_interpretation[id(branch)]), repr(interpretation)
                )
            )
        else:
            expression_context.append(
                (
                    branchname,
                    {
                        "is_primary": is_primary,
                        "is_cut": is_cut,
                        "is_duplicate": True,
                        "is_jagged": is_jagged,
                        "branch": branch,
                    },
                )
            )

    else:
        expression_context.append(
            (
                branchname,
                {
                    "is_primary": is_primary,
                    "is_cut": is_cut,
                    "is_duplicate": False,
                    "is_jagged": is_jagged,
                    "branch": branch,
                },
            )
        )
        branchid_interpretation[id(branch)] = interpretation


def _regularize_expression(
    hasbranches,
    expression,
    keys,
    aliases,
    compute,
    get_from_cache,
    arrays,
    expression_context,
    branchid_interpretation,
    symbol_path,
    is_cut,
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
        if expression in aliases:
            to_compute = aliases[expression]
        else:
            to_compute = expression

        is_jagged = False
        for symbol in compute.free_symbols(
            to_compute,
            keys,
            aliases,
            hasbranches.file.file_path,
            hasbranches.object_path,
        ):
            if symbol in symbol_path:
                raise ValueError(
                    """symbol {0} is recursively defined with aliases:

    {1}

in file {2} at {3}""".format(
                        repr(symbol),
                        "\n    ".join(
                            "{0}: {1}".format(k, v) for k, v in aliases.items()
                        ),
                        hasbranches.file.file_path,
                        hasbranches.object_path,
                    )
                )

            _regularize_expression(
                hasbranches,
                symbol,
                keys,
                aliases,
                compute,
                get_from_cache,
                arrays,
                expression_context,
                branchid_interpretation,
                symbol_path + (symbol,),
                False,
            )
            if expression_context[-1][1]["is_jagged"]:
                is_jagged = True

        expression_context.append(
            (
                expression,
                {"is_primary": is_primary, "is_cut": is_cut, "is_jagged": is_jagged},
            )
        )


def _regularize_expressions(
    hasbranches,
    expressions,
    cut,
    filter_name,
    filter_typename,
    filter_branch,
    keys,
    aliases,
    compute,
    get_from_cache,
):
    arrays = {}
    expression_context = []
    branchid_interpretation = {}

    if expressions is None:
        for branchname, branch in hasbranches.iteritems(
            recursive=True,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            full_paths=True,
        ):
            if not isinstance(
                branch.interpretation,
                uproot4.interpretation.identify.UnknownInterpretation,
            ):
                _regularize_branchname(
                    hasbranches,
                    branchname,
                    branch,
                    branch.interpretation,
                    get_from_cache,
                    arrays,
                    expression_context,
                    branchid_interpretation,
                    True,
                    False,
                )

    elif uproot4._util.isstr(expressions):
        _regularize_expression(
            hasbranches,
            expressions,
            keys,
            aliases,
            compute,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            (),
            False,
        )

    elif isinstance(expressions, Iterable):
        if isinstance(expressions, dict):
            items = expressions.items()
        else:
            items = []
            for expression in expressions:
                if uproot4._util.isstr(expression):
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
                    compute,
                    get_from_cache,
                    arrays,
                    expression_context,
                    branchid_interpretation,
                    (),
                    False,
                )
            else:
                branch = hasbranches[expression]
                interp = _regularize_interpretation(interp)
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
            "new Interpretation), not {0}".format(repr(expressions))
        )

    if cut is None:
        pass
    elif uproot4._util.isstr(cut):
        _regularize_expression(
            hasbranches,
            cut,
            keys,
            aliases,
            compute,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            (),
            True,
        )

    return arrays, expression_context, branchid_interpretation


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
        branch, basket_num, range_or_basket = ranges_or_baskets[original_index]
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
                basket.data,
                basket.byte_offsets,
                basket,
                branch,
                branch.context,
                basket.member("fKeylen"),
            )
            if basket.num_entries != len(basket_arrays[basket.basket_num]):
                raise ValueError(
                    """basket {0} in tree/branch {1} has the wrong number of entries """
                    """(expected {2}, obtained {3}) when interpreted as {4}
    in file {5}""".format(
                        basket.basket_num,
                        branch.object_path,
                        basket.num_entries,
                        len(basket_arrays[basket.basket_num]),
                        interpretation,
                        branch.file.file_path,
                    )
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
            for k, v in self.iteritems(recursive=True, full_paths=True):
                if where == k:
                    self._lookup[original_where] = v
                    return v
            else:
                raise uproot4.KeyInFileError(
                    original_where,
                    keys=self.keys(recursive=recursive),
                    file_path=self._file.file_path,
                    object_path=self.object_path,
                )

        elif recursive:
            got = _get_recursive(self, where)
            if got is not None:
                self._lookup[original_where] = got
                return got
            else:
                raise uproot4.KeyInFileError(
                    original_where,
                    file_path=self._file.file_path,
                    keys=self.keys(recursive=recursive),
                    object_path=self.object_path,
                )

        else:
            for branch in self.branches:
                if branch.name == where:
                    self._lookup[original_where] = branch
                    return branch
            else:
                raise uproot4.KeyInFileError(
                    original_where,
                    keys=self.keys(recursive=recursive),
                    file_path=self._file.file_path,
                    object_path=self.object_path,
                )

    def iteritems(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        full_paths=True,
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
                (filter_name is no_filter or filter_name(branch.name))
                and (filter_typename is no_filter or filter_typename(branch.typename))
                and (filter_branch is no_filter or filter_branch(branch))
            ):
                yield branch.name, branch

            if recursive:
                for k1, v in branch.iteritems(
                    recursive=recursive,
                    filter_name=no_filter,
                    filter_typename=filter_typename,
                    filter_branch=filter_branch,
                    full_paths=full_paths,
                ):
                    if full_paths:
                        k2 = "{0}/{1}".format(branch.name, k1)
                    else:
                        k2 = k1
                    if filter_name is no_filter or filter_name(k2):
                        yield k2, v

    def items(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        full_paths=True,
    ):
        return list(
            self.iteritems(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                full_paths=full_paths,
            )
        )

    def iterkeys(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        full_paths=True,
    ):
        for k, v in self.iteritems(
            recursive=recursive,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            full_paths=full_paths,
        ):
            yield k

    def keys(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        full_paths=True,
    ):
        return list(
            self.iterkeys(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                full_paths=full_paths,
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
            full_paths=False,
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
        full_paths=True,
    ):
        for k, v in self.iteritems(
            recursive=recursive,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            full_paths=full_paths,
        ):
            yield k, v.typename

    def typenames(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        full_paths=True,
    ):
        return dict(
            self.itertypenames(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                full_paths=full_paths,
            )
        )

    def __iter__(self):
        for x in self.branches:
            yield x

    def __len__(self):
        return len(self.branches)

    def show(
        self,
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        full_paths=True,
        name_width=20,
        typename_width=20,
        interpretation_width=34,
        stream=sys.stdout,
    ):
        """
        Args:
            stream: Object with a `write` method for writing the output.
        """
        formatter = "{{0:{0}.{0}}} | {{1:{1}.{1}}} | {{2:{2}.{2}}}\n".format(
            name_width, typename_width, interpretation_width,
        )

        stream.write(formatter.format("name", "typename", "interpretation"))
        stream.write(
            "-" * name_width
            + "-+-"
            + "-" * typename_width
            + "-+-"
            + "-" * interpretation_width
            + "\n"
        )

        if isinstance(self, TBranch):
            stream.write(
                formatter.format(self.name, self.typename, repr(self.interpretation))
            )

        for name, branch in self.iteritems(
            recursive=recursive,
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            full_paths=full_paths,
        ):
            stream.write(
                formatter.format(name, branch.typename, repr(branch.interpretation))
            )

    def arrays(
        self,
        expressions=None,
        cut=None,
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

        def get_from_cache(branchname, interpretation):
            if array_cache is not None:
                cache_key = "{0}:{1}:{2}:{3}-{4}:{5}".format(
                    self.cache_key,
                    branchname,
                    interpretation.cache_key,
                    entry_start,
                    entry_stop,
                    library.name,
                )
                return array_cache.get(cache_key)
            else:
                return None

        keys = set(self.keys(recursive=True, full_paths=False))
        aliases = _regularize_aliases(self, aliases)
        arrays, expression_context, branchid_interpretation = _regularize_expressions(
            self,
            expressions,
            cut,
            filter_name,
            filter_typename,
            filter_branch,
            keys,
            aliases,
            compute,
            get_from_cache,
        )

        ranges_or_baskets = []
        for expression, context in expression_context:
            branch = context.get("branch")
            if branch is not None and not context["is_duplicate"]:
                for basket_num, range_or_basket in branch.entries_to_ranges_or_baskets(
                    entry_start, entry_stop
                ):
                    ranges_or_baskets.append((branch, basket_num, range_or_basket))

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
            for expression, context in expression_context:
                branch = context.get("branch")
                if branch is not None:
                    interpretation = branchid_interpretation[id(branch)]
                    if branch is not None:
                        cache_key = "{0}:{1}:{2}:{3}-{4}:{5}".format(
                            self.cache_key,
                            expression,
                            interpretation.cache_key,
                            entry_start,
                            entry_stop,
                            library.name,
                        )
                    array_cache[cache_key] = arrays[id(branch)]

        output = compute.compute_expressions(
            arrays,
            expression_context,
            keys,
            aliases,
            self.file.file_path,
            self.object_path,
        )

        expression_context = [
            (e, c) for e, c in expression_context if c["is_primary"] and not c["is_cut"]
        ]

        return library.group(output, expression_context, how)


_branch_clean_name = re.compile(r"(.*\.)*([^\.\[\]]*)(\[.*\])*")
_branch_clean_parent_name = re.compile(r"(.*\.)*([^\.\[\]]*)\.([^\.\[\]]*)(\[.*\])*")


class TBranch(HasBranches):
    def postprocess(self, chunk, cursor, context):
        fWriteBasket = self.member("fWriteBasket")

        self._lookup = {}
        self._interpretation = None
        self._count_branch = None
        self._count_leaf = None
        self._typename = None
        self._streamer = None
        self._context = dict(context)
        self._context["breadcrumbs"] = ()
        self._context["in_TBranch"] = True

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
        out = self
        while not isinstance(out, uproot4.behaviors.TTree.TTree):
            out = out.parent
        return out

    @property
    def context(self):
        return self._context

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
        if self.interpretation is None:
            return "unknown"
        else:
            return self.interpretation.typename

    @property
    def top_level(self):
        return isinstance(self.parent, uproot4.behaviors.TTree.TTree)

    @property
    def streamer(self):
        if self._streamer is None:
            clean_name = _branch_clean_name.match(self.name).group(2)
            clean_parentname = _branch_clean_parent_name.match(self.name)
            fParentName = self.member("fParentName", none_if_missing=True)
            fClassName = self.member("fClassName", none_if_missing=True)

            if fParentName is not None and fParentName != "":
                matches = self._file.streamers.get(fParentName)

                if matches is not None:
                    streamerinfo = matches[max(matches)]

                    for element in streamerinfo.elements:
                        if element.name == clean_name:
                            self._streamer = element
                            break

                    if self._streamer is None and clean_parentname is not None:
                        clean_parentname = clean_parentname.group(2)
                        for element in streamerinfo.elements:
                            if element.name == clean_parentname:
                                substreamerinfo = self._file.streamer_named(
                                    element.typename
                                )
                                for subelement in substreamerinfo.elements:
                                    if subelement.name == clean_name:
                                        self._streamer = subelement
                                        break
                                break

            elif fClassName is not None and fClassName != "":
                matches = self._file.streamers.get(fClassName)
                if matches is not None:
                    self._streamer = matches[max(matches)]

        return self._streamer

    @property
    def interpretation(self):
        if self._interpretation is None:
            try:
                self._interpretation = uproot4.interpretation.identify.interpretation_of(
                    self, {}
                )
            except uproot4.interpretation.identify.UnknownInterpretation as err:
                self._interpretation = err
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
        else:
            interpretation = _regularize_interpretation(interpretation)
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
