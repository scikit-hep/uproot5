# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys

import numpy
import pytest
import skhep_testdata

import uproot4
import uproot4.model
import uproot4.reading


def test_classname_encoding(tmpdir):
    assert (
        uproot4.model.classname_encode("namespace::some.deep<templated, thing>", 12)
        == "ROOT_namespace_3a3a_some_2e_deep_3c_templated_2c20_thing_3e__v12"
    )
    assert (
        uproot4.model.classname_encode("namespace::some.deep<templated, thing>")
        == "ROOT_namespace_3a3a_some_2e_deep_3c_templated_2c20_thing_3e_"
    )

    assert uproot4.model.classname_decode(
        "ROOT_namespace_3a3a_some_2e_deep_3c_templated_2c20_thing_3e__v12"
    ) == ("namespace::some.deep<templated, thing>", 12)
    assert uproot4.model.classname_decode(
        "ROOT_namespace_3a3a_some_2e_deep_3c_templated_2c20_thing_3e_"
    ) == ("namespace::some.deep<templated, thing>", None)


def test_file_header():
    filename = skhep_testdata.data_path("uproot-Zmumu.root")
    file = uproot4.reading.ReadOnlyFile(filename)
    assert repr(file.compression) == "ZLIB(4)"
