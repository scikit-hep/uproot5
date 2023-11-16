import pytest

import uproot
import pathlib


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (
            "https://github.com/scikit-hep/scikit-hep-testdata/raw/v0.4.33/src/skhep_testdata/data/uproot-issue121.root:Events",
            (
                "https://github.com/scikit-hep/scikit-hep-testdata/raw/v0.4.33/src/skhep_testdata/data/uproot-issue121.root",
                "Events",
            ),
        ),
        (
            "https://github.com/scikit-hep/scikit-hep-testdata/raw/v0.4.33/src/skhep_testdata/data/uproot-issue121.root",
            (
                "https://github.com/scikit-hep/scikit-hep-testdata/raw/v0.4.33/src/skhep_testdata/data/uproot-issue121.root",
                None,
            ),
        ),
        (
            "github://scikit-hep:scikit-hep-testdata@v0.4.33/src/skhep_testdata/data/uproot-issue121.root:Dir/Events",
            (
                "github://scikit-hep:scikit-hep-testdata@v0.4.33/src/skhep_testdata/data/uproot-issue121.root",
                "Dir/Events",
            ),
        ),
        (
            "github://scikit-hep:scikit-hep-testdata@v0.4.33/src/skhep_testdata/data/uproot-issue121.root",
            (
                "github://scikit-hep:scikit-hep-testdata@v0.4.33/src/skhep_testdata/data/uproot-issue121.root",
                None,
            ),
        ),
        (
            "  http://localhost:8080/dir/test.root: Test ",
            (
                "http://localhost:8080/dir/test.root",
                "Test",
            ),
        ),
        (
            pathlib.Path("/tmp/test/dir/file.root:Test"),
            (
                str(pathlib.Path("/tmp/test/dir/file.root")),
                "Test",
            ),
        ),
        (
            r"C:\tmp\test\dir\file.root:Dir/Test",
            (
                r"C:\tmp\test\dir\file.root",
                "Dir/Test",
            ),
        ),
        (
            r"C:\tmp\test\dir\file.root",
            (
                r"C:\tmp\test\dir\file.root",
                None,
            ),
        ),
        (
            "ssh://user@host:22/path/to/file.root:/object//path",
            (
                "ssh://user@host:22/path/to/file.root",
                "object/path",
            ),
        ),
        (
            "ssh://user@host:22/path/to/file.root:/object//path:with:colon:in:path/something/",
            (
                "ssh://user@host:22/path/to/file.root",
                "object/path:with:colon:in:path/something",
            ),
        ),
        (
            "ssh://user@host:50230/path/to/file.root",
            (
                "ssh://user@host:50230/path/to/file.root",
                None,
            ),
        ),
        (
            "s3://bucket/path/to/file.root:/dir////object",
            (
                "s3://bucket/path/to/file.root",
                "dir/object",
            ),
        ),
        (
            "s3://bucket/path/to/file.root:",
            (
                "s3://bucket/path/to/file.root",
                "",
            ),
        ),
        (
            "00376186-543E-E311-8D30-002618943857.root:Events",
            (
                "00376186-543E-E311-8D30-002618943857.root",
                "Events",
            ),
        ),
        (
            "00376186-543E-E311-8D30-002618943857.root",
            (
                "00376186-543E-E311-8D30-002618943857.root",
                None,
            ),
        ),
        # https://github.com/scikit-hep/uproot5/issues/975
        (
            "DAOD_PHYSLITE_2023-09-13T1230.art.rntuple.root:RNT:CollectionTree",
            (
                "DAOD_PHYSLITE_2023-09-13T1230.art.rntuple.root",
                "RNT:CollectionTree",
            ),
        ),
        (
            "zip://uproot-issue121.root:Events/MET_pt::file:///tmp/pytest-of-runner/pytest-0/test_fsspec_zip0/uproot-issue121.root.zip",
            (
                "zip://uproot-issue121.root::file:///tmp/pytest-of-runner/pytest-0/test_fsspec_zip0/uproot-issue121.root.zip",
                "Events/MET_pt",
            ),
        ),
        (
            "simplecache::zip://uproot-issue121.root:Events/MET_pt::file:///tmp/pytest-of-runner/pytest-0/test_fsspec_zip0/uproot-issue121.root.zip",
            (
                "simplecache::zip://uproot-issue121.root::file:///tmp/pytest-of-runner/pytest-0/test_fsspec_zip0/uproot-issue121.root.zip",
                "Events/MET_pt",
            ),
        ),
        (
            r"zip://uproot-issue121.root:Events/MET_pt::file://C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_fsspec_zip0\uproot-issue121.root.zip",
            (
                r"zip://uproot-issue121.root::file://C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_fsspec_zip0\uproot-issue121.root.zip",
                "Events/MET_pt",
            ),
        ),
    ],
)
def test_url_split(input_value, expected_output):
    url, obj = uproot._util.file_object_path_split(input_value)
    url_expected, obj_expected = expected_output
    assert url == url_expected
    assert obj == obj_expected


@pytest.mark.parametrize(
    "input_value",
    [
        "local/file.root.zip://Events",
        "local/file.roo://Events",
        "local/file.roo://Events",
    ],
)
def test_url_split_invalid(input_value):
    path, obj = uproot._util.file_object_path_split(input_value)
    assert obj is None
    assert path == input_value
