# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines an :doc:`uproot.interpretation.Interpretation` as
a base class for all user-defined custom interpretations, and provides
a :doc:`uproot.interpretation.Interpretation` for extracting binary data
from ``TBasket`` objects.
"""
from __future__ import annotations

import numpy

import uproot
import uproot.behaviors.TBranch
import uproot.extras
import uproot.interpretation

awkward = uproot.extras.awkward()


class CustomInterpretation(uproot.interpretation.Interpretation):
    def __init__(
        self,
        branch: uproot.behaviors.TBranch.TBranch,
        context: dict,
        simplify: bool,
    ):
        """
        Args:
            branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch`` to
                interpret as an array.
            context (dict): Auxiliary data used in deserialization.
            simplify (bool): If True, call
                :ref:`uproot.interpretation.objects.AsObjects.simplify` on any
                :doc:`uproot.interpretation.objects.AsObjects` to try to get a
                more efficient interpretation.

        Accept arguments from `uproot.interpretation.identify.interpretation_of`.
        """
        self._branch = branch
        self._context = context
        self._simplify = simplify

    def match_branch(
        self,
        branch: uproot.behaviors.TBranch.TBranch,
        context: dict,
        simplify: bool,
    ) -> bool:
        """
        Args:
            branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch`` to
                interpret as an array.
            context (dict): Auxiliary data used in deserialization.
            simplify (bool): If True, call
                :ref:`uproot.interpretation.objects.AsObjects.simplify` on any
                :doc:`uproot.interpretation.objects.AsObjects` to try to get a
                more efficient interpretation.

        Accept arguments from `uproot.interpretation.identify.interpretation_of`,
        determine whether this interpretation can be applied to the given branch.
        """
        raise NotImplementedError

    @property
    def typename(self) -> str:
        """
        The name of the type of the interpretation.
        """
        return self._branch.streamer.typename

    @property
    def cache_key(self) -> str:
        """
        The cache key of the interpretation.
        """
        return id(self)

    def __repr__(self) -> str:
        """
        The string representation of the interpretation.
        """
        return self.__class__.__name__

    def final_array(
        self,
        basket_arrays,
        entry_start,
        entry_stop,
        entry_offsets,
        library,
        branch,
        options,
    ):
        """
        Concatenate the arrays from the baskets and return the final array.
        """

        awkward = uproot.extras.awkward()

        basket_entry_starts = numpy.array(entry_offsets[:-1])
        basket_entry_stops = numpy.array(entry_offsets[1:])

        basket_start_idx = numpy.where(basket_entry_starts <= entry_start)[0].max()
        basket_end_idx = numpy.where(basket_entry_stops >= entry_stop)[0].min()

        arr_to_concat = [
            basket_arrays[i] for i in range(basket_start_idx, basket_end_idx + 1)
        ]
        tot_array = awkward.concatenate(arr_to_concat)

        relative_entry_start = entry_start - basket_entry_starts[basket_start_idx]
        relative_entry_stop = entry_stop - basket_entry_starts[basket_start_idx]

        return tot_array[relative_entry_start:relative_entry_stop]


class AsBinary(uproot.interpretation.Interpretation):
    """
    Return binary data of the ``TBasket``. Pass an instance of this class
    to :ref:`uproot.behaviors.TBranch.TBranch.array` like this:

    .. code-block:: python
        binary_data = branch.array(interpretation=AsBinary())

    """

    @property
    def cache_key(self) -> str:
        return id(self)

    def basket_array(
        self,
        data,
        byte_offsets,
        basket,
        branch,
        context,
        cursor_offset,
        library,
        interp_options,
    ):
        counts = byte_offsets[1:] - byte_offsets[:-1]
        awkward = uproot.extras.awkward()
        return awkward.unflatten(data, counts)

    def final_array(
        self,
        basket_arrays,
        entry_start,
        entry_stop,
        entry_offsets,
        library,
        branch,
        options,
    ):
        basket_entry_starts = numpy.array(entry_offsets[:-1])
        basket_entry_stops = numpy.array(entry_offsets[1:])

        basket_start_idx = numpy.where(basket_entry_starts <= entry_start)[0].max()
        basket_end_idx = numpy.where(basket_entry_stops >= entry_stop)[0].min()

        arr_to_concat = [
            basket_arrays[i] for i in range(basket_start_idx, basket_end_idx + 1)
        ]

        awkward = uproot.extras.awkward()
        tot_array = awkward.concatenate(arr_to_concat)

        relative_entry_start = entry_start - basket_entry_starts[basket_start_idx]
        relative_entry_stop = entry_stop - basket_entry_starts[basket_start_idx]

        return tot_array[relative_entry_start:relative_entry_stop]
