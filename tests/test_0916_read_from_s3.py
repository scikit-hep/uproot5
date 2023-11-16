# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest

import sys
import uproot

pytest.importorskip("minio")

if sys.version_info < (3, 11):
    pytest.skip(
        "S3Source tests cause pytest to break in the CI (https://github.com/scikit-hep/uproot5/pull/1012)",
        allow_module_level=True,
    )


@pytest.mark.network
def test_s3_fail():
    with pytest.raises(FileNotFoundError):
        with uproot.source.s3.S3Source(
            "s3://pivarski-princeton/does-not-exist", timeout=0.1
        ) as source:
            uproot._util.tobytes(source.chunk(0, 100).raw_data)


@pytest.mark.network
def test_read_s3():
    with uproot.open(
        "s3://pivarski-princeton/pythia_ppZee_run17emb.picoDst.root:PicoDst",
        handler=uproot.source.s3.S3Source,
    ) as f:
        data = f["Event/Event.mEventId"].array(library="np")
        assert len(data) == 8004
