# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot


@pytest.fixture(scope="module")
def datafile(tmpdir_factory):
    yield skhep_testdata.data_path("uproot-issue-407.root")


@pytest.fixture(params=["foo", "foo_padded"])
def _object(request, datafile):
    with uproot.open(datafile) as f:
        yield f[request.param]


@pytest.fixture
def tree(datafile):
    with uproot.open(datafile) as f:
        yield f["tree"]


def test_streamer(_object):
    assert _object.members["d"].members["fDatime"] == 1749155840


def test_strided_interpretation(tree):
    assert list(tree.iterate(how=tuple))[0][0].fDatime[0] == 1749155840
