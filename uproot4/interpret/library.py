# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.interpret.jagged
import uproot4.interpret.objects


class Library(object):
    """
    Indicates the type of array to produce.

       * `imported`: The imported library or raises a helpful "how to"
             message if it could not be imported.
       * `empty(shape, dtype)`: Make (possibly temporary) multi-basket storage.
       * `finalize(array, branch)`: Convert the internal storage form into one
             appropriate for this library (NumPy array, Pandas Series, etc.).
       * `group(arrays, names, how)`: Combine arrays into a group, either
             a generic tuple or a grouping style appropriate for this library
             (NumPy array dict, Awkward RecordArray, Pandas DataFrame, etc.).
    """

    @property
    def imported(self):
        raise AssertionError

    def empty(self, shape, dtype):
        raise AssertionError

    def finalize(self, array, branch):
        raise AssertionError

    def group(self, arrays, names, how):
        if how is tuple:
            return tuple(arrays[name] for name in names)
        elif how is list:
            return [arrays[name] for name in names]
        elif how is dict or how is None:
            return arrays
        else:
            raise TypeError(
                "for library {0}, how must be tuple, list, dict, or None (for "
                "dict)".format(self.name)
            )

    def __repr__(self):
        return repr(self.name)

    def __eq__(self, other):
        return type(_libraries[self.name]) is type(_libraries[other.name])  # noqa: E721


class NumPy(Library):
    name = "np"

    @property
    def imported(self):
        import numpy

        return numpy

    def empty(self, shape, dtype):
        return numpy.empty(shape, dtype)

    def finalize(self, array, branch):
        if isinstance(array, uproot4.interpret.jagged.JaggedArray):
            out = numpy.zeros(len(array), dtype=numpy.object)
            for i, x in enumerate(array):
                out[i] = x
            return out

        else:
            return array


class Awkward(Library):
    name = "ak"

    @property
    def imported(self):
        try:
            import awkward1
        except ImportError:
            raise ImportError(
                """install the 'awkward1' package with:

    pip install awkward1"""
            )
        else:
            return awkward1


class Pandas(Library):
    name = "pd"

    @property
    def imported(self):
        try:
            import pandas
        except ImportError:
            raise ImportError(
                """install the 'pandas' package with:

    pip install pandas

or

    conda install pandas"""
            )
        else:
            return pandas

    def empty(self, shape, dtype):
        return numpy.empty(shape, dtype)

    def finalize(self, array, branch):
        pandas = self.imported

        if isinstance(array, uproot4.interpret.jagged.JaggedArray):
            compact = array.compact
            index = pandas.MultiIndex.from_arrays(
                [compact.parents, compact.localindex], names=["entry", "subentry"]
            )
            return pandas.Series(compact.content, index=index)

        elif isinstance(array, uproot4.interpret.objects.ObjectArray):
            out = numpy.zeros(len(array), dtype=numpy.object)
            for i, x in enumerate(array):
                out[i] = x
            return pandas.Series(out)

        else:
            return pandas.Series(array)

    def group(self, arrays, names, how):
        pandas = self.imported
        if how is tuple:
            return tuple(arrays[name] for name in names)
        elif how is list:
            return [arrays[name] for name in names]
        elif how is dict:
            return arrays
        elif uproot4._util.isstr(how) or how is None:
            if all(isinstance(x.index, pandas.RangeIndex) for x in arrays.values()):
                return pandas.DataFrame(data=arrays, columns=names)
            indexes = []
            groups = []
            for name in names:
                array = arrays[name]
                if isinstance(array.index, pandas.MultiIndex):
                    for index, group in zip(indexes, groups):
                        if numpy.array_equal(array.index, index):
                            group.append(name)
                            break
                    else:
                        indexes.append(array.index)
                        groups.append([name])
            if how is None:
                flat_index = None
                dfs = [[] for x in indexes]
                group_names = [[] for x in indexes]
                for index, group, df, gn in zip(indexes, groups, dfs, group_names):
                    for name in names:
                        array = arrays[name]
                        if isinstance(array.index, pandas.RangeIndex):
                            if flat_index is None or len(flat_index) != len(
                                array.index
                            ):
                                flat_index = pandas.MultiIndex.from_arrays(
                                    [array.index]
                                )
                            df.append(
                                pandas.Series(array.values, index=flat_index).reindex(
                                    index
                                )
                            )
                            gn.append(name)
                        elif name in group:
                            df.append(array)
                            gn.append(name)
                out = []
                for index, df, gn in zip(indexes, dfs, group_names):
                    out.append(
                        pandas.DataFrame(
                            data=dict(zip(gn, df)), index=index, columns=gn
                        )
                    )
                if len(out) == 1:
                    return out[0]
                else:
                    return tuple(out)
            else:
                out = None
                for index, group in zip(indexes, groups):
                    only = dict((name, arrays[name]) for name in group)
                    df = pandas.DataFrame(data=only, index=index, columns=group)
                    if out is None:
                        out = df
                    else:
                        out = pandas.merge(
                            out, df, how=how, left_index=True, right_index=True
                        )
                flat_names = [
                    name
                    for name in names
                    if isinstance(arrays[name].index, pandas.RangeIndex)
                ]
                if len(flat_names) > 0:
                    flat_index = pandas.MultiIndex.from_arrays(
                        [arrays[flat_names[0]].index]
                    )
                    only = dict(
                        (name, pandas.Series(arrays[name].values, index=flat_index))
                        for name in flat_names
                    )
                    df = pandas.DataFrame(
                        data=only, index=flat_index, columns=flat_names
                    )
                    out = pandas.merge(
                        df.reindex(out.index),
                        out,
                        how=how,
                        left_index=True,
                        right_index=True,
                    )
                return out

        else:
            raise TypeError(
                "for library {0}, how must be tuple, list, dict, str (for "
                "pandas.merge's 'how' parameter, or None (for one or more"
                "DataFrames without merging)".format(self.name)
            )


class CuPy(Library):
    name = "cp"

    @property
    def imported(self):
        try:
            import cupy
        except ImportError:
            raise ImportError(
                """install the 'cupy' package with:

    pip install cupy

or

    conda install cupy"""
            )
        else:
            return cupy

    def empty(self, shape, dtype):
        cupy = self.imported
        return cupy.empty(shape, dtype)

    def finalize(self, array, branch):
        cupy = self.imported

        if isinstance(array, uproot4.interpret.jagged.JaggedArray):
            raise TypeError("jagged arrays and objects are not supported by CuPy")

        else:
            return cupy.array(array)


_libraries = {
    NumPy.name: NumPy(),
    Awkward.name: Awkward(),
    Pandas.name: Pandas(),
    CuPy.name: CuPy(),
}

_libraries["numpy"] = _libraries[NumPy.name]
_libraries["Numpy"] = _libraries[NumPy.name]
_libraries["NumPy"] = _libraries[NumPy.name]
_libraries["NUMPY"] = _libraries[NumPy.name]

_libraries["awkward1"] = _libraries[Awkward.name]
_libraries["Awkward1"] = _libraries[Awkward.name]
_libraries["AWKWARD1"] = _libraries[Awkward.name]
_libraries["awkward"] = _libraries[Awkward.name]
_libraries["Awkward"] = _libraries[Awkward.name]
_libraries["AWKWARD"] = _libraries[Awkward.name]

_libraries["pandas"] = _libraries[Pandas.name]
_libraries["Pandas"] = _libraries[Pandas.name]
_libraries["PANDAS"] = _libraries[Pandas.name]

_libraries["cupy"] = _libraries[CuPy.name]
_libraries["Cupy"] = _libraries[CuPy.name]
_libraries["CuPy"] = _libraries[CuPy.name]
_libraries["CUPY"] = _libraries[CuPy.name]


def _regularize_library(library):
    if isinstance(library, Library):
        return _libraries[library.name]

    elif isinstance(library, type) and issubclass(library, Library):
        return _libraries[library().name]

    else:
        try:
            return _libraries[library]
        except KeyError:
            raise ValueError("unrecognized library: {0}".format(repr(library)))
