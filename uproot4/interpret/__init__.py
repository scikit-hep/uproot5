# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import


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

    def wrap_numpy(self, array):
        raise AssertionError

    def wrap_jagged(self, array):
        raise AssertionError

    def wrap_python(self, array):
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

    def wrap_numpy(self, array):
        return array

    def wrap_jagged(self, array):
        return self.wrap_python(array)

    def wrap_python(self, array):
        numpy = self.imported
        out = numpy.zeros(len(array), dtype=numpy.object)
        for i, x in enumerate(array):
            out[i] = x
        return out


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

    def wrap_numpy(self, array):
        pandas = self.imported
        return pandas.Series(pandas)

    def wrap_jagged(self, array):
        array = array.compact
        pandas = self.imported
        index = pandas.MultiIndex.from_arrays(
            [array.parents, array.localindex], names=["entry", "subentry"]
        )
        return pandas.Series(array.content, index=index)

    def wrap_python(self, array):
        pandas = self.imported
        return pandas.Series(array)


_libraries = {
    NumPy.name: NumPy(),
    Awkward.name: Awkward(),
    Pandas.name: Pandas(),
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


def _regularize_library(library):
    try:
        return _libraries[library]
    except KeyError:
        raise ValueError("unrecognized library: {0}".format(repr(library)))


class Interpretation(object):
    """
    Abstract class for interpreting TTree basket data as arrays (NumPy and
    Awkward). The following methods must be defined:

       * `cache_key`: Used to distinguish the same array read with different
             interpretations in a cache.
       * `numpy_dtype`: Data type (including any shape elements after the first
             dimension) of the NumPy array that would be created.
       * `awkward_form`: Form of the Awkward Array that would be created
             (requires `awkward1`); used by the `ak.type` function.
       * `empty_array(library)`: An empty, finalized array, as defined by this
             Interpretation and Library.
       * `num_items(num_bytes, num_entries)`: Predict the number of items.
       * `basket_array(data, byte_offsets)`: Create a basket_array from a
             basket's `data` and `byte_offsets`.
       * `fillable_array(num_items, num_entries)`: Create the array that is
             incrementally filled by baskets as they arrive. This may include
             excess at the beginning of the first basket and the end of the
             last basket to cover full baskets (trimmed later).
       * `fill(basket_array, fillable_array, item_start, item_stop, entry_start,
             entry_stop)`: Copy data from the basket_array to fillable_array
             with possible transformations (e.g. big-to-native endian, shift
             offsets).
       * `trim(fillable_array, entry_start, entry_stop)`: Remove any excess
             entries in the first and last baskets.
       * `finalize(fillable_array, library)`: Return an array in the desired
             form for a given Library.
    """

    @property
    def cache_key(self):
        raise AssertionError

    @property
    def numpy_dtype(self):
        raise AssertionError

    @property
    def awkward_form(self):
        raise AssertionError

    def empty_array(self, library):
        raise AssertionError

    def num_items(self, num_bytes, num_entries):
        raise AssertionError

    def basket_array(self, data, byte_offsets):
        raise AssertionError

    def fillable_array(self, num_items, num_entries):
        raise AssertionError

    def fill(
        self,
        basket_array,
        fillable_array,
        item_start,
        item_stop,
        entry_start,
        entry_stop,
    ):
        raise AssertionError

    def trim(self, fillable_array, entry_start, entry_stop):
        raise AssertionError

    def finalize(self, fillable_array, library):
        raise AssertionError

    def hook_before_basket_array(self, data, byte_offsets):
        pass

    def hook_after_basket_array(self, data, byte_offsets, basket_array):
        pass

    def hook_before_fillable_array(self, num_items, num_entries):
        pass

    def hook_after_fillable_array(self, num_items, num_entries, fillable_array):
        pass

    def hook_before_fill(
        self,
        basket_array,
        fillable_array,
        item_start,
        item_stop,
        entry_start,
        entry_stop,
    ):
        pass

    def hook_after_fill(
        self,
        basket_array,
        fillable_array,
        item_start,
        item_stop,
        entry_start,
        entry_stop,
    ):
        pass

    def hook_before_trim(self, fillable_array, entry_start, entry_stop):
        pass

    def hook_after_trim(self, fillable_array, entry_start, entry_stop, trimmed_array):
        pass

    def hook_before_finalize(self, fillable_array, library):
        pass

    def hook_after_finalize(self, fillable_array, library, final_array):
        pass
