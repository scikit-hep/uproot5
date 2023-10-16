import pytest

import uproot
import pathlib


def test_url_split():
    for input_url, result in [
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
                # make it work on Windows and Linux
                r"C:\tmp\test\dir\file.root",
                "Dir/Test",
            ),
        ),
    ]:
        url, obj = uproot._util.file_object_path_split(input_url)
        assert url == result[0]
        assert obj == result[1]
