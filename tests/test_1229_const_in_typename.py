# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

from uproot.interpretation.identify import parse_typename


def test_const_in_typename():
    assert parse_typename("TH1I*") == parse_typename("const TH1I*")
