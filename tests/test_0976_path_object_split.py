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
            "ssh://user@host:port/path/to/file:object",
            (
                "ssh://user@host:port/path/to/file",
                "object",
            ),
        ),
        (
            "ssh://user@host:port/path/to/file",
            (
                "ssh://user@host:port/path/to/file",
                None,
            ),
        ),
        (
            "s3://bucket/path/to/file:object",
            (
                "s3://bucket/path/to/file",
                "object",
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
        (
            "zip://uproot-issue121.root::file:///tmp/pytest-of-runner/pytest-0/test_fsspec_zip0/uproot-issue121.root.zip:Events/MET_pt",
            (
                "zip://uproot-issue121.root::file:///tmp/pytest-of-runner/pytest-0/test_fsspec_zip0/uproot-issue121.root.zip",
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
        "local/file.root://Events",
    ],
)
def test_url_split_invalid(input_value):
    with pytest.raises(ValueError):
        uproot._util.file_object_path_split(input_value)
