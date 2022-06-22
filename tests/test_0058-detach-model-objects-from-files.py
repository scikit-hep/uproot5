# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import copy
import os
import pickle
import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test_detachment():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        assert getattr(f["hpx"].file, "file_path", None) is not None
        assert getattr(f["hpx"].file, "source", None) is None

        assert getattr(f["ntuple"].file, "file_path", None) is not None
        assert getattr(f["ntuple"].file, "source", None) is not None

    with uproot.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-nosplit.root")
    ) as f:
        array = f["tree/evt"].array(library="np", entry_stop=1)
        assert getattr(array[0].file, "file_path", None) is not None
        assert getattr(array[0].file, "source", None) is None

        assert isinstance(
            f.file.streamer_named("Event").file, uproot.reading.DetachedFile
        )
        assert (
            str(f.file.streamer_named("Event").file_uuid)
            == "9eebcae8-366b-11e7-ab9d-5e789e86beef"
        )


def test_copy():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        original = f["hpx"]
        original_file_path = original.file.file_path

        reconstituted = copy.deepcopy(original)
        reconstituted_file_path = reconstituted.file.file_path

        assert original_file_path == reconstituted_file_path


def test_pickle():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        original = f["hpx"]
        original_file_path = original.file.file_path

        reconstituted = pickle.loads(pickle.dumps(original))
        reconstituted_file_path = reconstituted.file.file_path

        assert original_file_path == reconstituted_file_path


def test_pickle_boost():
    boost_histogram = pytest.importorskip("boost_histogram")
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        original = f["hpx"]
        original_boost = original.to_boost()

        reconstituted = pickle.loads(pickle.dumps(original))
        reconstituted_boost = reconstituted.to_boost()

        pickle.loads(pickle.dumps(original_boost))
        pickle.loads(pickle.dumps(reconstituted_boost))
