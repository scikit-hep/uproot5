# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

files = {skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-0-0.root"): "Staff"}
add_docs = pytest.mark.parametrize(
    "ak_add_doc", [True, {"__doc__": "title", "typename": "typename"}]
)


def _check(form, reference):
    division = form.contents[form.fields.index("Division")]
    assert division.parameters["__array__"] == "string"
    assert "__doc__" in division.parameters
    assert "__doc__" in form.parameters
    assert form.is_equal_to(reference, all_parameters=True)


def _dask_form(ak_add_doc, open_files):
    return uproot.dask(files, open_files=open_files, ak_add_doc=ak_add_doc).layout.form


@add_docs
@pytest.mark.parametrize("open_files", [False, True])
def test_dask_adds_the_rntuple_docs(ak_add_doc, open_files):
    pytest.importorskip("dask_awkward")
    _check(_dask_form(ak_add_doc, open_files), _dask_form(ak_add_doc, False))


@add_docs
def test_graphed_adds_the_rntuple_docs(ak_add_doc):
    pytest.importorskip("dask_awkward")
    pytest.importorskip("graphed")
    array = uproot.graphed(files, ak_add_doc=ak_add_doc)
    _check(array.session.form(array).tt.layout.form, _dask_form(ak_add_doc, False))
