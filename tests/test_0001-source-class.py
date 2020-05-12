# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import os

import uproot4
import uproot4.source.source

def test(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    for num_workers in [1, 2]:
        source = uproot4.source.source.FileSource(filename, num_workers=num_workers)
        assert not source.ready

        with source as tmp:
            assert source.ready
            chunks = tmp.chunks([(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)])
            assert [chunk.raw_data for chunk in chunks] == [
                b"******", b"    ", b"...", b"+++++++", b"!!!!!", b"@@@@@"]

        assert not source.ready
