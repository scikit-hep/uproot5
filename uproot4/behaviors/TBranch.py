# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

import uproot4.source.cursor
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


class HasBranches(Mapping):
    @property
    def branches(self):
        return self.member("fBranches")

    def __getitem__(self, where):
        original_where = where

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
                    return v
            else:
                raise uproot4.KeyInFileError(original_where, self._file.file_path)

        elif recursive:
            got = _get_recursive(self, where)
            if got is not None:
                return got
            else:
                raise uproot4.KeyInFileError(original_where, self._file.file_path)

        else:
            for branch in self.branches:
                if branch.name == where:
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


class TBranch(HasBranches):
    @property
    def name(self):
        return self.member("fName")

    @property
    def title(self):
        return self.member("fTitle")

    @property
    def typename(self):
        return "FIXME"

    @property
    def interpretation(self):
        if self._interpretation is None:
            raise NotImplementedError
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
        return int(self.member("fEntries"))   # or fEntryNumber?

    @property
    def num_baskets(self):
        # FIXME: recover (which should be defined in the MODEL)
        return self._num_good_baskets + len(self._recovered_baskets)

    def __repr__(self):
        return "<TBranch {0} ({1} subbranches)>".format(repr(self.name), len(self))

    def basket_cursor(self, basket_num):
        if 0 <= basket_num < self._num_good_baskets:
            return uproot4.source.cursor.Cursor(self.member("fBasketSeek")[basket_num])
        elif 0 <= basket_num < self.num_baskets:
            raise NotImplementedError
        else:
            raise IndexError(
                """branch {0} has {1} baskets; cannot get basket {2}
in file {3}""".format(self.name, self.num_baskets, basket_num, self._file.file_path)
            )

    def basket_chunk_bytes(self, basket_num):
        if 0 <= basket_num < self._num_good_baskets:
            return int(self.member("fBasketBytes")[basket_num])
        elif 0 <= basket_num < self.num_baskets:
            raise NotImplementedError
        else:
            raise IndexError(
                """branch {0} has {1} baskets; cannot get basket {2}
in file {3}""".format(self.name, self.num_baskets, basket_num, self._file.file_path)
            )

    def basket_compressed_bytes(self, basket_num):
        raise NotImplementedError

    def basket_uncompressed_bytes(self, basket_num):
        raise NotImplementedError

    def basket_chunk(self, basket_num):
        start = self.basket_cursor(basket_num).index
        stop = start + self.basket_chunk_bytes(basket_num)

        return self._file.source.chunk(start, stop)

