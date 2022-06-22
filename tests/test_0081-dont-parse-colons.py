# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

pytest.importorskip("awkward")


def test_open():
    assert isinstance(
        uproot.open(skhep_testdata.data_path("uproot-issue63.root")),
        uproot.reading.ReadOnlyDirectory,
    )
    assert isinstance(
        uproot.open(
            {skhep_testdata.data_path("uproot-issue63.root"): "WtLoop_nominal"}
        ),
        uproot.behaviors.TTree.TTree,
    )

    with pytest.raises(ValueError):
        uproot.open([skhep_testdata.data_path("uproot-issue63.root")])


def test_concatenate():
    with pytest.raises(ValueError):
        uproot.concatenate(skhep_testdata.data_path("uproot-issue63.root"))

    assert (
        len(
            uproot.concatenate(
                {skhep_testdata.data_path("uproot-issue63.root"): "blah"},
                allow_missing=True,
            )
        )
        == 0
    )

    files = skhep_testdata.data_path("uproot-sample-6.16.00-uncompressed.root").replace(
        "6.16.00", "*"
    )

    uproot.concatenate(files, "Ai8")
    uproot.concatenate({files: "sample"}, "Ai8")
    uproot.concatenate([files], "Ai8")
    uproot.concatenate([{files: "sample"}], "Ai8")


def test_iterate():
    with pytest.raises(ValueError):
        for arrays in uproot.iterate(skhep_testdata.data_path("uproot-issue63.root")):
            pass

    assert (
        len(
            list(
                uproot.iterate(
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

    for arrays in uproot.iterate(files, "Ai8"):
        pass
    for arrays in uproot.iterate({files: "sample"}, "Ai8"):
        pass
    for arrays in uproot.iterate([files], "Ai8"):
        pass
    for arrays in uproot.iterate([{files: "sample"}], "Ai8"):
        pass


pathlib = pytest.importorskip("pathlib")


def test_open_colon():
    assert isinstance(
        uproot.open(
            skhep_testdata.data_path("uproot-issue63.root") + ":WtLoop_nominal"
        ),
        uproot.behaviors.TTree.TTree,
    )

    with pytest.raises(FileNotFoundError):
        uproot.open(
            pathlib.Path(
                skhep_testdata.data_path("uproot-issue63.root") + ":WtLoop_nominal"
            )
        )

    with pytest.raises(FileNotFoundError):
        uproot.open(
            {skhep_testdata.data_path("uproot-issue63.root") + ":WtLoop_nominal": None}
        )
