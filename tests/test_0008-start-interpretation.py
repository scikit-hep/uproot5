# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys

import numpy
import pytest
import skhep_testdata

import uproot4
import uproot4.model
import uproot4.reading
import uproot4.source.file
import uproot4.source.memmap
import uproot4.source.http
import uproot4.source.xrootd


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
    assert not file.is_64bit
    assert file.fNbytesInfo == 4447
    assert file.hex_uuid == "944b77d0-98ab-11e7-a769-0100007fbeef"


def test_http_begin_end():
    filename = "https://example.com"
    with uproot4.source.http.HTTPSource(
        filename, timeout=10, num_fallback_workers=0
    ) as source:
        begin, end = source.begin_end_chunks(20, 30)
        assert len(begin.raw_data.tostring()) == 20
        assert len(end.raw_data.tostring()) == 30


def test_http_begin_end_fallback():
    filename = "https://scikit-hep.org/uproot/examples/Zmumu.root"
    with uproot4.source.http.HTTPSource(
        filename, timeout=10, num_fallback_workers=0
    ) as source:
        begin, end = source.begin_end_chunks(20, 30)
        assert len(begin.raw_data.tostring()) == 20
        assert len(end.raw_data.tostring()) == 30
