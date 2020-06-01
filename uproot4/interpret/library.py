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
       * `wrap_numpy(array)`: Wraps a NumPy array into the native type for
             this library.
       * `wrap_jagged(array)`: Wraps a jagged array into the native type for
             this library.
       * `wrap_python(array)`: Wraps an array of Python objects into the native
             type for this library.
    """

    @property
    def imported(self):
        raise AssertionError

    def empty(self, shape, dtype):
        raise AssertionError

    def finalize(self, array, branch):
        raise AssertionError

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
