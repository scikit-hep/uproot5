# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Regression tests for PR #1647: model member-name and AwkwardForth-API bugs.

Covers:
- Model_TTree_v18.member_names (was named member_values, silently returning [])
- Model_TAttMarker_v2.member_names typo fMarkserSize -> fMarkerSize
- Model_TTime_v2.read_members Forth path using removed old API
- TTable module-level `format` dict shadowing builtin
"""

from __future__ import annotations

import skhep_testdata

import uproot
import uproot.models.TAtt
import uproot.models.TTime
import uproot.models.TTree


def test_ttree_v18_member_names_is_nonempty():
    """Model_TTree_v18.member_names should return a non-empty list (was silently
    returning [] because the property was misnamed `member_values`)."""
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-5.26.00-uncompressed.root")
    )["sample"] as tree:
        assert tree.class_version == 18
        names = tree.member_names
        assert isinstance(names, list)
        assert len(names) > 0
        # spot-check a few expected names
        assert "fEntries" in names
        assert "fBranches" in names
        assert "fLeaves" in names


def test_tattmarker_v2_member_names_no_typo():
    """Model_TAttMarker_v2.member_names should contain 'fMarkerSize', not 'fMarkserSize'."""
    names = uproot.models.TAtt.Model_TAttMarker_v2.member_names
    assert "fMarkerSize" in names
    assert "fMarkserSize" not in names
    assert names == ["fMarkerColor", "fMarkerStyle", "fMarkerSize"]


def test_ttime_read_members_forth_path_no_old_api():
    """Model_TTime_v2.read_members Forth code must not reference any removed
    old-API method names (get_gen_obj, get_keys, add_to_header, etc.)."""
    import inspect

    src = inspect.getsource(uproot.models.TTime.Model_TTime_v2.read_members)
    removed_api = [
        "get_gen_obj",
        "get_keys",
        "add_to_header",
        "should_add_form",
        "add_form_key",
        "add_to_pre",
        "add_form",
    ]
    for api in removed_api:
        assert api not in src, f"Removed API method {api!r} still referenced in TTime"


def test_ttime_read_members_forth_path_uses_node_api():
    """Model_TTime_v2.read_members should use the current Node-based Forth API."""
    import inspect

    src = inspect.getsource(uproot.models.TTime.Model_TTime_v2.read_members)
    assert "uproot._awkwardforth.Node" in src
    assert "uproot._awkwardforth.get_first_key_number" in src
    assert "forth_obj.add_node" in src
    assert "forth_obj.set_active_node" in src


def test_ttable_format_builtin_not_shadowed():
    """TTable module must not define a module-level `format` that shadows the builtin."""
    import uproot.models.TTable

    assert not hasattr(
        uproot.models.TTable, "format"
    ), "TTable module should not have a 'format' attribute"
