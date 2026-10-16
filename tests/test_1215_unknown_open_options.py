# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import warnings

import pytest
import skhep_testdata

import uproot
import uproot._util
import uproot.source.file
import uproot.source.fsspec

ZMUMU = "uproot-Zmumu.root"


def test_unknown_option_raises_for_uproot_source():
    path = skhep_testdata.data_path(ZMUMU)
    with pytest.raises(TypeError, match="num_wrokers"):
        uproot.open(path, handler=uproot.source.file.MemmapSource, num_wrokers=2)


def test_unknown_option_raises_for_file_like_object():
    path = skhep_testdata.data_path(ZMUMU)
    with open(path, "rb") as f, pytest.raises(TypeError, match="bogus"):
        uproot.open(f, bogus=1)


def test_subclass_of_uproot_source_is_strict():
    class MyMemmap(uproot.source.file.MemmapSource):
        pass

    path = skhep_testdata.data_path(ZMUMU)
    with pytest.raises(TypeError, match="bogus"):
        uproot.open(path, handler=MyMemmap, bogus=1)


def test_subclass_of_fsspec_source_warns():
    class MyFSSpec(uproot.source.fsspec.FSSpecSource):
        pass

    path = skhep_testdata.data_path(ZMUMU)
    with pytest.warns(UserWarning, match="did you mean.*'num_workers'"):
        uproot.open(path, handler=MyFSSpec, num_wrokers=2).close()


def test_non_source_handler_is_not_checked():
    path = skhep_testdata.data_path(ZMUMU)

    def handler(file_path, **options):
        return uproot.source.file.MemmapSource(file_path, **options)

    uproot._util._check_unknown_options({"handler": handler, "bogus": 1}, path)


def test_valid_options_accepted_for_uproot_source():
    path = skhep_testdata.data_path(ZMUMU)
    with uproot.open(path, handler=uproot.source.file.MemmapSource, num_workers=2) as f:
        assert f["events"].num_entries == 2304


def test_typo_warns_for_fsspec_source():
    path = skhep_testdata.data_path(ZMUMU)
    with pytest.warns(UserWarning, match="did you mean.*'num_workers'"):
        uproot.open(path, num_wrokers=2).close()


def test_valid_fsspec_storage_options_not_rejected(recwarn):
    path = skhep_testdata.data_path(ZMUMU)
    with uproot.open(path, auto_mkdir=False, num_workers=2) as f:
        assert f["events"].num_entries == 2304
    assert not [w for w in recwarn if "did you mean" in str(w.message)]


@pytest.mark.parametrize(
    "key",
    [
        "max_workers",
        "num_threads",
        "use_threading",
        "max_header_bytes",
        "headers",
        "anon",
        "key",
        "token",
        "client_kwargs",
        "block_size",
    ],
)
def test_plausible_fsspec_options_do_not_warn(key):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        uproot._util._check_unknown_options({key: 1})


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("num_wrokers", "num_workers"),
        ("num_worker", "num_workers"),
        ("numworkers", "num_workers"),
        ("handlr", "handler"),
        ("timout", "timeout"),
        ("begin_chunksize", "begin_chunk_size"),
    ],
)
def test_near_miss_options_warn(key, expected):
    with pytest.warns(UserWarning, match=f"did you mean.*{expected!r}"):
        uproot._util._check_unknown_options({key: 1})


def test_warning_points_at_caller():
    path = skhep_testdata.data_path(ZMUMU)
    with pytest.warns(UserWarning, match="did you mean") as record:
        uproot.open(path, num_wrokers=2).close()
    assert record[0].filename == __file__


def test_timeout_accepted_with_http_source():
    uproot._util._check_unknown_options(
        {"handler": uproot.HTTPSource, "timeout": 5}, "http://127.0.0.1:1/f.root"
    )
    # nothing listens here; the failure must be a connection error, not a TypeError
    with pytest.raises(OSError):  # noqa: PT011
        uproot.open("http://127.0.0.1:1/f.root", handler=uproot.HTTPSource, timeout=1)


def test_timeout_reaches_fsspec():
    path = skhep_testdata.data_path(ZMUMU)
    source = uproot.source.fsspec.FSSpecSource(path, timeout=5)
    assert source._fsspec_options == {"timeout": 5}
    source.close()

    source = uproot.source.fsspec.FSSpecSource(path, timeout=None)
    assert source._fsspec_options == {}
    source.close()

    assert uproot._util.fsspec_storage_options(
        {"timeout": 5, "num_workers": 2, "anon": True}
    ) == {"timeout": 5, "anon": True}
    assert uproot._util.fsspec_storage_options({"timeout": None}) == {}


def test_path_object_options_ignored_for_non_fsspec_handler():
    path = skhep_testdata.data_path(ZMUMU)

    class StubPath:
        storage_options = {"headers": {"X": "y"}}

        def __str__(self):
            return path

    with uproot.open(StubPath(), handler=uproot.source.file.MemmapSource) as f:
        assert f["events"].num_entries == 2304


def test_path_object_options_merged_for_default_handler():
    path = skhep_testdata.data_path(ZMUMU)

    class StubPath:
        storage_options = {"num_wrokers": 1}

        def __str__(self):
            return path

    with pytest.warns(UserWarning, match="did you mean.*'num_workers'"):
        uproot.open(StubPath()).close()


def test_iterate_concatenate():
    path = skhep_testdata.data_path(ZMUMU) + ":events"
    handler = uproot.source.file.MemmapSource

    with pytest.raises(TypeError, match="bogus"):
        next(uproot.iterate(path, handler=handler, bogus=1))
    with pytest.raises(TypeError, match="bogus"):
        uproot.concatenate(path, handler=handler, bogus=1)

    with pytest.warns(UserWarning, match="did you mean") as record:
        next(uproot.iterate(path, num_wrokers=2))
    assert record[0].filename == __file__
    with pytest.warns(UserWarning, match="did you mean") as record:
        uproot.concatenate(path, num_wrokers=2, library="np")
    assert record[0].filename == __file__


@pytest.mark.parametrize("open_files", [True, False])
def test_dask(open_files):
    pytest.importorskip("dask")
    pytest.importorskip("dask_awkward")
    path = skhep_testdata.data_path(ZMUMU) + ":events"
    handler = uproot.source.file.MemmapSource

    with pytest.raises(TypeError, match="bogus"):
        uproot.dask(path, open_files=open_files, handler=handler, bogus=1)

    with pytest.warns(UserWarning, match="did you mean") as record:
        uproot.dask(path, open_files=open_files, num_wrokers=2)
    assert record[0].filename == __file__
