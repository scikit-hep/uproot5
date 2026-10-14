# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import pytest
import skhep_testdata

import uproot
import uproot.source.file


def test_unknown_option_raises_for_uproot_source():
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with pytest.raises(TypeError, match="num_wrokers"):
        uproot.open(path, handler=uproot.source.file.MemmapSource, num_wrokers=2)


def test_unknown_option_raises_for_file_like_object():
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with open(path, "rb") as f, pytest.raises(TypeError, match="bogus"):
        uproot.open(f, bogus=1)


def test_valid_options_accepted_for_uproot_source():
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with uproot.open(path, handler=uproot.source.file.MemmapSource, num_workers=2) as f:
        assert f["events"].num_entries == 2304


def test_typo_warns_for_fsspec_source():
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with pytest.warns(UserWarning, match="did you mean.*'num_workers'"):
        uproot.open(path, num_wrokers=2).close()


def test_valid_fsspec_storage_options_not_rejected(recwarn):
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with uproot.open(path, auto_mkdir=False, num_workers=2) as f:
        assert f["events"].num_entries == 2304
    assert not [w for w in recwarn if "did you mean" in str(w.message)]
