# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import skhep_testdata

import uproot4


def test_open():
    assert isinstance(
        uproot4.open(skhep_testdata.data_path("uproot-issue63.root")),
        uproot4.reading.ReadOnlyDirectory,
    )
    assert isinstance(
        uproot4.open(
            {skhep_testdata.data_path("uproot-issue63.root"): "WtLoop_nominal"}
        ),
        uproot4.behaviors.TTree.TTree,
    )

    with pytest.raises(ValueError):
        uproot4.open([skhep_testdata.data_path("uproot-issue63.root")])


def test_lazy():
    with pytest.raises(ValueError):
        uproot4.lazy(skhep_testdata.data_path("uproot-issue63.root"))

    with pytest.raises(ValueError):
        uproot4.lazy(
            {skhep_testdata.data_path("uproot-issue63.root"): "blah"},
            allow_missing=True,
        )

    uproot4.lazy({skhep_testdata.data_path("uproot-issue63.root"): "WtLoop_nominal"})
    uproot4.lazy(
        {
            skhep_testdata.data_path("uproot-issue63.root"): "WtLoop_nominal",
            skhep_testdata.data_path("uproot-issue63.root"): "WtLoop_Fake_nominal",
        }
    )

    uproot4.lazy([{skhep_testdata.data_path("uproot-issue63.root"): "WtLoop_nominal"}])
    uproot4.lazy(
        {skhep_testdata.data_path("uproot-issue63.root") + "*": "WtLoop_nominal"}
    )
    uproot4.lazy(
        [{skhep_testdata.data_path("uproot-issue63.root") + "*": "WtLoop_nominal"}]
    )


def test_concatenate():
    with pytest.raises(ValueError):
        uproot4.concatenate(skhep_testdata.data_path("uproot-issue63.root"))

    assert (
        len(
            uproot4.concatenate(
                {skhep_testdata.data_path("uproot-issue63.root"): "blah"},
                allow_missing=True,
            )
        )
        == 0
    )

    files = skhep_testdata.data_path("uproot-sample-6.16.00-uncompressed.root").replace(
        "6.16.00", "*"
    )

    uproot4.concatenate(files, "Ai8")
    uproot4.concatenate({files: "sample"}, "Ai8")
    uproot4.concatenate([files], "Ai8")
    uproot4.concatenate([{files: "sample"}], "Ai8")


def test_iterate():
    with pytest.raises(ValueError):
        for arrays in uproot4.iterate(skhep_testdata.data_path("uproot-issue63.root")):
            pass

    assert (
        len(
            list(
                uproot4.iterate(
                    {skhep_testdata.data_path("uproot-issue63.root"): "blah"},
                    allow_missing=True,
                )
            )
        )
        == 0
    )

    files = skhep_testdata.data_path("uproot-sample-6.16.00-uncompressed.root").replace(
        "6.16.00", "*"
    )

    for arrays in uproot4.iterate(files, "Ai8"):
        pass
    for arrays in uproot4.iterate({files: "sample"}, "Ai8"):
        pass
    for arrays in uproot4.iterate([files], "Ai8"):
        pass
    for arrays in uproot4.iterate([{files: "sample"}], "Ai8"):
        pass
