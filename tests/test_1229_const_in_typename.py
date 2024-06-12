# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

from uproot.containers import AsPointer
from uproot.interpretation.identify import parse_typename
from uproot.models.TH import Model_TH1I


def test_const_in_typename():
    assert parse_typename("TH1I*") == AsPointer(Model_TH1I)
    assert parse_typename("const TH1I*") == AsPointer(Model_TH1I)
