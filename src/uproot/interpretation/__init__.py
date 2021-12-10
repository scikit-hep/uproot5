# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines procedures for interpreting data in ``TTrees`` as arrays.

All interpretations must be subclasses of
:doc:`uproot.interpretation.Interpretation`.

See :doc:`uproot.interpretation.identify.interpretation_of` for heuristics
that determine the default interpretation of a
:doc:`uproot.behaviors.TBranch.TBranch`.
"""


class Interpretation:
    """
    Abstract class for all interpretations of ``TBranch`` data as arrays.

    The interpretation cycle consists of:

    1. Producing temporary arrays from each uncompressed ``TBasket``.
    2. Combining those temporary arrays for the whole range of entries
       requested between ``entry_start`` and ``entry_stop`` in
       :ref:`uproot.behaviors.TBranch.HasBranches.arrays` or
       :ref:`uproot.behaviors.TBranch.TBranch.array`, or by ``entry_step``
       in :ref:`uproot.behaviors.TBranch.TBranch.iterate`.
    3. Trimming the combined array to the exact entry range requested.
       (``TBasket`` boundaries might not align with the requested entry range.)
    4. Passing the combined, trimmed temporary array to a selected
       :doc:`uproot.interpretation.library.Library` for finalization
       and possibly grouping.
    """

    @property
    def cache_key(self):
        """
        String that uniquely specifies this interpretation, to use as part of
        an array's cache key.
        """
        raise AssertionError

    @property
    def typename(self):
        """
        String that describes this interpretation as a C++ type.

        This type might not exactly correspond to the type in C++, but it would
        have equivalent meaning.
        """
        raise AssertionError

    @property
    def numpy_dtype(self):
        """
        The ``numpy.dtype`` to use to put objects of this type in a NumPy array.
        """
        raise AssertionError

    def awkward_form(
        self,
        file,
        index_format="i64",
        header=False,
        tobject_header=True,
        breadcrumbs=(),
    ):
        """
        Args:
            file (:doc:`uproot.reading.ReadOnlyFile`): File to use to generate
                :doc:`uproot.model.Model` classes from its
                :ref:`uproot.reading.ReadOnlyFile.streamers` and ``file_path``
                for error messages.
            index_format (str): Format to use for indexes of the
                ``awkward.forms.Form``; may be ``"i32"``, ``"u32"``, or
                ``"i64"``.
            header (bool): If True, include headers in the Form's ``"uproot"``
                parameters.
            tobject_header (bool): If True, include headers for ``TObject``
                classes in the Form's ``"uproot"`` parameters.
            breadcrumbs (tuple of class objects): Used to check for recursion.
                Types that contain themselves cannot be Awkward Arrays because the
                depth of instances is unknown.

        The ``awkward.forms.Form`` to use to put objects of type type in an
        Awkward Array.
        """
        raise AssertionError

    def basket_array(
        self, data, byte_offsets, basket, branch, context, cursor_offset, library
    ):
        """
        Args:
            data (array of ``numpy.uint8``): Raw but uncompressed data from the
                ``TBasket``. If the ``TBasket`` has offsets and navigational
                metadata, it is not included in this array.
            byte_offsets (array of ``numpy.int32``): Index where each entry of
                the ``TBasket`` starts and stops. The header is not included
                (i.e. the first offset is ``0``), and the length of this array
                is one greater than the number of entries in the ``TBasket``.
            basket (:doc:`uproot.models.TBasket.Model_TBasket`): The ``TBasket`` object.
            context (dict): Auxiliary data used in deserialization.
            cursor_offset (int): Correction to the integer keys used in
                :ref:`uproot.source.cursor.Cursor.refs` for objects
                deserialized by reference
                (:doc:`uproot.deserialization.read_object_any`).
            library (:doc:`uproot.interpretation.library.Library`): The
                requested library for output.

        Performs the first step of interpretation, from uncompressed ``TBasket``
        data to a temporary array.
        """
        raise AssertionError

    def final_array(
        self, basket_arrays, entry_start, entry_stop, entry_offsets, library, branch
    ):
        """
        Args:
            basket_arrays (dict of int \u2192 array): Mapping from ``TBasket``
                number to the temporary array returned by
                :ref:`uproot.interpretation.Interpretation.basket_array`.
            entry_start (int): First entry to include when trimming any
                excess entries from the first ``TBasket``.
            entry_stop (int): FIrst entry to exclude (one greater than the last
                entry to include) when trimming any excess entries from the
                last ``TBasket``.
            entry_offsets (list of int): The
                :ref:`uproot.behaviors.TBranch.TBranch.entry_offsets` for this
                ``TBranch``.
            library (:doc:`uproot.interpretation.library.Library`): The
                requested library for output.
            branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch``
                that is being interpreted.

        Performs the last steps of interpretation, from a collection of
        temporary arrays, one for each ``TBasket``, to a trimmed, finalized,
        grouped array, produced by the ``library``.
        """
        raise AssertionError

    def __eq__(self, other):
        raise AssertionError

    def __ne__(self, other):
        raise not self == other

    def hook_before_basket_array(self, *args, **kwargs):
        """
        Called in :ref:`uproot.interpretation.Interpretation.basket_array`,
        before any interpretation.

        This is the first hook called in
        :ref:`uproot.interpretation.Interpretation.basket_array`.
        """
        pass

    def hook_after_basket_array(self, *args, **kwargs):
        """
        Called in :ref:`uproot.interpretation.Interpretation.basket_array`,
        after all interpretation.

        This is the last hook called in
        :ref:`uproot.interpretation.Interpretation.basket_array`.
        """
        pass

    def hook_before_final_array(self, *args, **kwargs):
        """
        Called in :ref:`uproot.interpretation.Interpretation.final_array`,
        before any trimming, finalization, or grouping.

        This is the first hook called in
        :ref:`uproot.interpretation.Interpretation.final_array`.
        """
        pass

    def hook_before_library_finalize(self, *args, **kwargs):
        """
        Called in :ref:`uproot.interpretation.Interpretation.final_array`,
        after trimming but before calling the
        :ref:`uproot.interpretation.library.Library.finalize` routine.
        """
        pass

    def hook_after_final_array(self, *args, **kwargs):
        """
        Called in :ref:`uproot.interpretation.Interpretation.final_array`,
        after all trimming, finalization, and grouping.

        This is the last hook called in
        :ref:`uproot.interpretation.Interpretation.final_array`.
        """
        pass
