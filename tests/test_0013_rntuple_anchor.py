# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

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
        assert obj.member("fVersionEpoch") == 0
        assert obj.member("fVersionMajor") == 2
        assert obj.member("fVersionMinor") == 0
        assert obj.member("fVersionPatch") == 0
        assert obj.member("fSeekHeader") == 266
        assert obj.member("fNBytesHeader") == 391
        assert obj.member("fLenHeader") == 996
        assert obj.member("fSeekFooter") == 36420
        assert obj.member("fNBytesFooter") == 89
        assert obj.member("fLenFooter") == 172
        assert obj.member("fChecksum") == 12065027575882477574

        header_start = obj.member("fSeekHeader")
        header_stop = header_start + obj.member("fNBytesHeader")
        header_chunk = f.file.source.chunk(header_start, header_stop)

        # print("HEADER")
        # cursor = uproot.Cursor(header_start)
        # cursor.debug(header_chunk, limit_bytes=80)
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
        # cursor.debug(footer_chunk, limit_bytes=80)
        # print("\n")


# HEADER
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  76  52   1 126   1   0 228   3   0 226  97  52  14 191  32 119  70  87   1   0
#   L   4 ---   ~ --- --- --- --- --- ---   a   4 --- ---       w   F   W --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
# 228   3   0   1   0 144   5   0   0   0  83 116  97 102 102  13   0 241   5  13
# --- --- --- --- --- --- --- --- --- ---   S   t   a   f   f --- --- --- --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#   0   0   0  82  79  79  84  32 118  54  46  51  49  47  48  49 122 253 255   1
# --- --- ---   R   O   O   T       v   6   .   3   1   /   0   1   z --- --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#   0 111  11   0   0   0  60   0   1   0   3 244  13   8   0   0   0  67  97 116
# ---   o --- --- --- ---   < --- --- --- --- --- --- --- --- --- ---   C   a   t


# FOOTER
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  76  52   1  80   0   0 172   0   0 143 248  61  98 249  16  31  87  72   2   0
#   L   4 ---   P --- --- --- --- --- --- ---   =   b --- --- ---   W   H --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
# 172   0   1   0 147 119 211 249  23 217  49  71 211  56  16   0  34 244 255   1
# --- --- --- --- ---   w --- --- --- ---   1   G ---   8 --- ---   " --- --- ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#   0  15  12   0  33  19 196  12   0 106   1   0   0   0  48   0   1   0  34  26
# --- --- --- ---   ! --- --- --- ---   j --- --- --- ---   0 --- --- ---   " ---
# --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
#  13   8   0   0  28   0  34  92   2  12   0 110 234   0   0   0  56 141 120   0
# --- --- --- --- --- ---   "   \ --- --- ---   n --- --- --- ---   8 ---   x ---
