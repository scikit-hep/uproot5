# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines known forth code for types it is known a priori.
"""
from __future__ import annotations


def known_forth_of(branch):
    from uproot.interpretation.known_forth.atlas.element_link import (
        vector_vector_element_link,
    )

    if (
        # len(branch.branches) == 0 # don't understand why this goes nuts
        # and branch.has_member("fClassName")
        branch.has_member("fClassName")
    ):
        typename = branch.member("fClassName").replace(" ", "")
        if typename.startswith("vector<vector<ElementLink<DataVector<xAOD::"):
            return vector_vector_element_link
