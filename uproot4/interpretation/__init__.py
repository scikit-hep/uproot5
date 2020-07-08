# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import


class Interpretation(object):
    """
    Abstract class for interpreting TTree basket data as arrays (NumPy and
    Awkward). The following methods must be defined:

       * `cache_key`: Used to distinguish the same array read with different
             interpretations in a cache.
       * `typename`: Name of the C++ data type from which this Interpretation
             was derived.
       * `numpy_dtype`: Data type (including any shape elements after the first
             dimension) of the NumPy array that would be created.
       * `awkward_form(file, index_format="i64", header=False, tobject_header=True)`:
             Form of the Awkward Array that would be created (requires `awkward1`).
       * `basket_array(data, byte_offsets, basket, branch, context, cursor_offset)`:
             Create a basket_array from a basket's `data` and `byte_offsets`.
       * `final_array(basket_arrays, entry_start, entry_stop, entry_offsets,
             library, branch)`: Combine basket_arrays with basket excess
             trimmed and in the form required by a given library.
    """

    @property
    def cache_key(self):
        raise AssertionError

    @property
    def typename(self):
        raise AssertionError

    @property
    def numpy_dtype(self):
        raise AssertionError

    def awkward_form(self, file, index_format="i64", header=False, tobject_header=True):
        raise AssertionError

    def basket_array(self, data, byte_offsets, basket, branch, context, cursor_offset):
        raise AssertionError

    def final_array(
        self, basket_arrays, entry_start, entry_stop, entry_offsets, library, branch
    ):
        raise AssertionError

    def __eq__(self, other):
        raise AssertionError

    def __ne__(self, other):
        raise not self == other

    def hook_before_basket_array(self, *args, **kwargs):
        pass

    def hook_after_basket_array(self, *args, **kwargs):
        pass

    def hook_before_final_array(self, *args, **kwargs):
        pass

    def hook_before_library_finalize(self, *args, **kwargs):
        pass

    def hook_after_final_array(self, *args, **kwargs):
        pass
