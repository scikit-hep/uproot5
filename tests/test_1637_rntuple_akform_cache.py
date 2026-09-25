# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy
import pytest
import skhep_testdata

import uproot
from uproot.behaviors import RNTuple as rntuple_behavior
from uproot.behaviors.RNTuple import HasFields

ak = pytest.importorskip("awkward")


def same_bytes(one, two):
    """Compare two arrays exactly, NaN included."""
    one_form, one_length, one_buffers = ak.to_buffers(ak.to_packed(one))
    two_form, two_length, two_buffers = ak.to_buffers(ak.to_packed(two))
    if one_form != two_form or one_length != two_length:
        return False
    if set(one_buffers) != set(two_buffers):
        return False
    return all(
        numpy.asarray(one_buffers[key]).tobytes()
        == numpy.asarray(two_buffers[key]).tobytes()
        for key in one_buffers
    )


@pytest.fixture(scope="module")
def ntuple():
    filepath = skhep_testdata.data_path(
        "cmsopendata2015_ttbar_19980_NANOAOD_RNTupleImporter_rntuple_v1-0-0-1.root"
    )
    with uproot.open(filepath) as file:
        yield file["Events"]


def count_full_builds(ntuple, monkeypatch):
    """Count how often the form is built from the field records."""
    calls = []
    original = HasFields._to_akform_full

    def counted(self, **kwargs):
        calls.append(self.path)
        return original(self, **kwargs)

    monkeypatch.setattr(HasFields, "_to_akform_full", counted)
    return calls


def test_form_is_built_once_for_many_fields(ntuple, monkeypatch):
    calls = count_full_builds(ntuple, monkeypatch)

    keys = ntuple.keys()[:200]
    for key in keys:
        ntuple[key].to_akform()

    assert len(calls) == 1


def test_pruned_form_matches_rebuilt_form(ntuple):
    keys = ntuple.keys()

    selections = [
        {},
        {"filter_name": keys[0]},
        {"filter_name": "/^Muon/"},
        {"filter_name": keys[:10]},
        {"filter_typename": "float"},
        {"filter_name": "no_field_is_called_this"},
    ]
    for kwargs in selections:
        assert ntuple.to_akform(**kwargs) == ntuple._to_akform_full(**kwargs)

    # subfields, which is how one column at a time gets read
    for key in keys[:100]:
        field = ntuple[key]
        assert field.to_akform() == field._to_akform_full()

    # the documented form is built the original way, but must still agree
    assert ntuple.to_akform(ak_add_doc=True) == ntuple._to_akform_full(ak_add_doc=True)


def test_arrays_are_unchanged(ntuple):
    for key in ntuple.keys()[:20]:
        field = ntuple[key]
        form, field_path = field.to_akform()
        full_form, full_field_path = field._to_akform_full()
        assert form == full_form
        assert field_path == full_field_path
        assert same_bytes(field.array(), ntuple.arrays(filter_name=key)[key])


@pytest.mark.parametrize(
    ("filename", "key"),
    [
        ("ntpl001_staff_rntuple_v1-0-0-0.root", "Staff"),
        ("test_nested_structs_rntuple_v1-0-0-0.root", "ntuple"),
        # tuples, and records whose subfields all fall outside the selection
        ("test_stl_containers_rntuple_v1-0-0-0.root", "ntuple"),
    ],
)
def test_pruning_handles_the_awkward_shapes(filename, key):
    with uproot.open(skhep_testdata.data_path(filename)) as file:
        obj = file[key]
        for kwargs in ({}, {"filter_name": "no_field_is_called_this"}):
            assert obj.to_akform(**kwargs) == obj._to_akform_full(**kwargs)
        for name in obj.keys():
            field = obj[name]
            assert field.to_akform() == field._to_akform_full()


def test_navigating_to_a_missing_field_gives_nothing():
    form = ak.forms.RecordForm(
        [ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64"))], ["muon"]
    )
    assert rntuple_behavior._navigate_akform(form, ["muon"]) is not None
    assert rntuple_behavior._navigate_akform(form, ["electron"]) is None
    # descending past the end of the form, into a leaf that has no fields
    assert rntuple_behavior._navigate_akform(form, ["muon", "pt"]) is None


def test_field_falls_back_when_the_cached_form_cannot_be_walked(ntuple, monkeypatch):
    """A form the walk cannot follow must not change what a field returns."""
    monkeypatch.setattr(
        rntuple_behavior, "_navigate_akform", lambda form, path_keys: None
    )
    for key in ntuple.keys()[:20]:
        field = ntuple[key]
        assert field.to_akform() == field._to_akform_full()
