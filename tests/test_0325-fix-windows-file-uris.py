# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import numpy as np
import pytest

import uproot._util
import uproot.reading


@pytest.mark.skipif(
    not uproot._util.win, reason="Drive letters only parsed on Windows."
)
def test_windows_drive_letters():
    assert (
        uproot._util.file_path_to_source_class(
            "file:///g:/mydir/file.root", uproot.reading.open.defaults
        )[1]
        == "g:/mydir/file.root"
    )

    assert (
        uproot._util.file_path_to_source_class(
            "file:/g:/mydir/file.root", uproot.reading.open.defaults
        )[1]
        == "g:/mydir/file.root"
    )

    assert (
        uproot._util.file_path_to_source_class(
            "file:g:/mydir/file.root", uproot.reading.open.defaults
        )[1]
        == "g:/mydir/file.root"
    )

    assert (
        uproot._util.file_path_to_source_class(
            "/g:/mydir/file.root", uproot.reading.open.defaults
        )[1]
        == "g:/mydir/file.root"
    )

    assert (
        uproot._util.file_path_to_source_class(
            r"\g:/mydir/file.root", uproot.reading.open.defaults
        )[1]
        == "g:/mydir/file.root"
    )

    assert (
        uproot._util.file_path_to_source_class(
            r"g:\mydir\file.root", uproot.reading.open.defaults
        )[1]
        == r"g:\mydir\file.root"
    )

    assert (
        uproot._util.file_path_to_source_class(
            r"\g:\mydir\file.root", uproot.reading.open.defaults
        )[1]
        == r"g:\mydir\file.root"
    )


def test_escaped_uri_codes():
    # If they're file:// paths, yes we should unquote the % signs.
    assert (
        uproot._util.file_path_to_source_class(
            "file:///my%20file.root", uproot.reading.open.defaults
        )[1]
        == "/my file.root"
    )
    assert (
        uproot._util.file_path_to_source_class(
            "file:///my%E2%80%92file.root", uproot.reading.open.defaults
        )[1]
        == "/my\u2012file.root"
    )

    # Otherwise, no we should not.
    assert (
        uproot._util.file_path_to_source_class(
            "/my%20file.root", uproot.reading.open.defaults
        )[1]
        == "/my%20file.root"
    )
    assert (
        uproot._util.file_path_to_source_class(
            "/my%E2%80%92file.root", uproot.reading.open.defaults
        )[1]
        == "/my%E2%80%92file.root"
    )
