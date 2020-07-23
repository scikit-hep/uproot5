# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import pickle

import numpy
import pytest
import skhep_testdata

import uproot4


def test_detachment():
    with uproot4.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        assert getattr(f["hpx"].file, "file_path", None) is not None
        assert getattr(f["hpx"].file, "source", None) is None

        assert getattr(f["ntuple"].file, "file_path", None) is not None
        assert getattr(f["ntuple"].file, "source", None) is not None

    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-nosplit.root")
    ) as f:
        array = f["tree/evt"].array(library="np", entry_stop=1)
        assert getattr(array[0].file, "file_path", None) is not None
        assert getattr(array[0].file, "source", None) is None

        assert isinstance(
            f.file.streamer_named("Event").file, uproot4.reading.DetachedFile
        )
        assert (
            str(f.file.streamer_named("Event").file_uuid)
            == "9eebcae8-366b-11e7-ab9d-5e789e86beef"
        )


# def test_pickle():
#     with uproot4.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
#         original = f["hpx"]
#         print(original.file.file_path)

#         reconstituted = pickle.loads(pickle.dumps(original))
#         print(reconstituted.file.file_path)

#     raise Exception
