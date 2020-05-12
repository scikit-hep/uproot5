# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import os
try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

import numpy

import uproot4
import uproot4.futures
import uproot4.source.source


def test_source(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    for num_workers in [1, 2]:
        source = uproot4.source.source.FileSource(filename, num_workers=num_workers)
        assert not source.ready

        with source as tmp:
            assert source.ready
            chunks = tmp.chunks(
                [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)]
            )
            assert [chunk.raw_data for chunk in chunks] == [
                b"******",
                b"    ",
                b"...",
                b"+++++++",
                b"!!!!!",
                b"@@@@@",
            ]

        assert not source.ready

def test_debug():
    data = numpy.concatenate([numpy.array([123, 123, 123], "u1"), numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 101, 202, 303], ">f4").view("u1"), numpy.array([123, 123], "u1")])
    future = uproot4.futures.TrivialFuture(data)

    chunk = uproot4.source.source.Chunk(None, 0, len(data), future)
    cursor = uproot4.source.source.Cursor(0)

    output = StringIO()

    cursor.debug(chunk, offset=3, dtype=">f4", stream=output)

    assert output.getvalue() == """--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
123 123 123  63 140 204 205  64  12 204 205  64  83  51  51  64 140 204 205  64
  {   {   {   ? --- --- ---   @ --- --- ---   @   S   3   3   @ --- --- ---   @
                        1.1             2.2             3.3             4.4
    --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
    176   0   0  64 211  51  51  64 246 102 102  65  12 204 205  65  30 102 102  66
    --- --- ---   @ ---   3   3   @ ---   f   f   A --- --- ---   A ---   f   f   B
            5.5             6.6             7.7             8.8             9.9
    --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
    202   0   0  67  74   0   0  67 151 128   0 123 123
    --- --- ---   C   J --- ---   C --- --- ---   {   {
          101.0           202.0           303.0
"""
