# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as fin:
        chunk, cursor = fin.key("hpx").get_uncompressed_chunk_cursor()
        cursor.debug(chunk, limit_bytes=20 * 8)

        hpx = fin["hpx"]

        # with uproot.recreate("try-histogram.root") as fout:
        #     fout["hpx"] = hpx

        # raise Exception


# def test_recreate_update(tmp_path):
#     filename = os.path.join(tmp_path, "whatever.root")

#     f1 = ROOT.TFile(filename, "recreate")
#     f1.mkdir("subdir")
#     f1.cd("subdir")
#     x = ROOT.TObjString("wowie")
#     x.Write()
#     f1.Close()
