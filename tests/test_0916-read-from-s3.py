# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest

import uproot


@pytest.mark.network
def test_s3_fail():
    with pytest.raises(Exception):
        with uproot.source.http.S3Source(
            "s3://pivarski-princeton/does-not-exist", timeout=0.1
        ) as source:
            tobytes(source.chunk(0, 100).raw_data)


@pytest.mark.network
def test_read_s3():
    with uproot.open(
        "s3://pivarski-princeton/pythia_ppZee_run17emb.picoDst.root:PicoDst"
    ) as f:
        data = f["Event/Event.mEventId"].array(library="np")
        assert len(data) == 8004
