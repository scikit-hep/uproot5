# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import pytest
import skhep_testdata

import uproot

from tests.test_1785_rntuple_ak_add_doc_every_reader import (  # noqa: F401 (fixture)
    _form,
    _lazy,
    documented_rntuple,
)


@pytest.mark.parametrize("reader", ["eager", "dask-delayed", "dask-open", "graphed"])
def test_title_reads_the_rntuple_description(documented_rntuple, reader):
    ((path, name),) = documented_rntuple.items()
    ntuple = uproot.open(path)[name]
    described = ntuple.arrays(ak_add_doc={"__doc__": "description"}).layout.form
    if reader == "eager":
        form = ntuple.arrays(ak_add_doc={"__doc__": "title"}).layout.form
    else:
        form = _form(
            reader, _lazy(reader, documented_rntuple, ak_add_doc={"__doc__": "title"})
        )
    assert form.content("x").parameters["__doc__"] == "an int"
    assert form.is_equal_to(described, all_parameters=True)


def test_an_open_ttree_takes_a_dict_doc_in_dask():
    pytest.importorskip("dask_awkward")
    path = skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    tree = uproot.open(path)["tree"]
    eager = tree.arrays(["I32"], ak_add_doc={"__doc__": "title"}).layout.form
    lazy = uproot.dask(
        tree, filter_name=["I32"], open_files=False, ak_add_doc={"__doc__": "title"}
    )
    assert lazy.layout.form.is_equal_to(eager, all_parameters=True)
