# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest

import uproot

pytest.importorskip("s3fs")


def test_s3_fail(s3_server):
    bucket_url, storage_options = s3_server
    with pytest.raises(FileNotFoundError):
        with uproot.source.fsspec.FSSpecSource(
            f"{bucket_url}/does-not-exist", **storage_options
        ) as source:
            uproot._util.tobytes(source.chunk(0, 100).raw_data)


def test_read_s3(s3_server):
    bucket_url, storage_options = s3_server
    with uproot.open(
        f"{bucket_url}/uproot-HZZ-1.root:events",
        **storage_options,
    ) as f:
        data = f["Muon_Px"].array(library="np")
        assert len(data) == 2421
