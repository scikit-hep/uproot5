# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import


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
       * `empty`: An empty array, according to this Interpretation.
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
       * `finalize(fillable_array)`: Return an array in the desired form.
    """

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

    def hook_before_finalize(self, fillable_array):
        pass

    def hook_after_finalize(self, fillable_array, final_array):
        pass
