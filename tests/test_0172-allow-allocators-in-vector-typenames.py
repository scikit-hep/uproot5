# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue-172.root")) as f:
        t = f["events"]
        t.show()
        assert (
            t["rec_part_px_VecOps"].typename == "std::vector<float>"
        )  # without the allocator
        t["rec_part_px_VecOps"].array(library="np")
