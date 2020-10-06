# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import skhep_testdata

import uproot4


def test_version():
    assert uproot4.classname_decode(
        uproot4.classname_encode("xAOD::MissingETAuxAssociationMap_v2")
    ) == ("xAOD::MissingETAuxAssociationMap_v2", None)
    assert uproot4.classname_decode(
        uproot4.classname_encode("xAOD::MissingETAuxAssociationMap_v2", 9)
    ) == ("xAOD::MissingETAuxAssociationMap_v2", 9)
