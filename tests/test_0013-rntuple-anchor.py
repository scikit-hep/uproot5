# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import json
import queue
import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test():
    filename = skhep_testdata.data_path("uproot-ntpl001_staff.root")
    with uproot.open(filename) as f:
        obj = f["Staff"]
        assert obj.member("fVersion") == 0
        assert obj.member("fSize") == 48
        assert obj.member("fSeekHeader") == 854
        assert obj.member("fNBytesHeader") == 537
        assert obj.member("fLenHeader") == 2495
        assert obj.member("fSeekFooter") == 72369
        assert obj.member("fNBytesFooter") == 285
        assert obj.member("fLenFooter") == 804
        assert obj.member("fReserved") == 0

        header_start = obj.member("fSeekHeader")
        header_stop = header_start + obj.member("fNBytesHeader")
        header_chunk = f.file.source.chunk(header_start, header_stop)

        # print("HEADER")
        # cursor = uproot.Cursor(header_start)
        # cursor.debug(header_chunk, 80)
        # print("\n")

        notifications = queue.Queue()
        footer_start = obj.member("fSeekFooter")
        footer_stop = footer_start + obj.member("fNBytesFooter")
        header_chunk, footer_chunk = f.file.source.chunks(
            [(header_start, header_stop), (footer_start, footer_stop)],
            notifications,
        )

        # print("FOOTER")
        # cursor = uproot.Cursor(footer_start)
        # cursor.debug(footer_chunk, 80)
        # print("\n")


# HEADER
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  76  52   1  16   2   0 191   9   0 198  14 105   8  80  63  75 128 117   0   0
#   L   4 --- --- --- --- --- --- --- --- ---   i ---   P   ?   K ---   u --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#   0   0 187   9   0   1   0 144   5   0   0   0  83 116  97 102 102  13   0 255
# --- --- --- --- --- --- --- --- --- --- --- ---   S   t   a   f   f --- --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#   6  16   0   0   0 117 110 100 101 102 105 110 101 100  32  97 117 116 104 111
# --- --- --- --- ---   u   n   d   e   f   i   n   e   d       a   u   t   h   o
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
# 114   0   1   0   4  47  24   0   1   0   3  31  12  12   0   0   4   8   0 110
#   r --- --- --- ---   / --- --- --- --- --- --- --- --- --- --- --- --- ---   n


# FOOTER
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  76  52   1  20   1   0  36   3   0  86 138 213  67  60 183  39 139  27   0   1
#   L   4 --- --- --- ---   $ --- ---   V --- ---   C   < ---   ' --- --- --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#   0  23   1  12   0  23  12  12   0  42  72   0   1   0  47  24   0   1   0   7
# --- --- --- --- --- --- --- --- ---   *   H --- --- ---   / --- --- --- --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  34  26  13   8   0  34 145   5   8   0  34 213   9  86   0  27  13  84   0   0
#   " --- --- --- ---   " --- --- --- ---   " --- ---   V --- --- ---   T --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#   1   0 102  52  26   0   0 148   1 124   0   0  16   0  34 102  15  17   0  34
# --- ---   f   4 --- --- --- --- ---   | --- --- --- ---   "   f --- --- ---   "
