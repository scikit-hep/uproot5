# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import queue
import pytest
import uproot
import skhep_testdata


def test_new_support_RNTuple_files():
    with uproot.open(
        "https://xrootd-local.unl.edu:1094//store/user/AGC/nanoaod-rntuple/zstd/TT_TuneCUETP8M1_13TeV-powheg-pythia8/cmsopendata2015_ttbar_19980_PU25nsData2015v1_76X_mcRun2_asymptotic_v12_ext3-v1_00000_0000.root"
    ) as f:
        obj = f["Events"]
        header_start = obj.member("fSeekHeader")
        header_stop = header_start + obj.member("fNBytesHeader")

        notifications = queue.Queue()

        footer_start = obj.member("fSeekFooter")
        footer_stop = footer_start + obj.member("fNBytesFooter")
        header_chunk, footer_chunk = f.file.source.chunks(
            [(header_start, header_stop), (footer_start, footer_stop)],
            notifications,
        )
        # assert footer_stop - footer_start == 273

        # print("FOOTER")
        # cursor = uproot.Cursor(footer_start)
        # cursor.debug(footer_chunk, 80)
        # print("\n")
        # array = obj.arrays(["nTau"])
