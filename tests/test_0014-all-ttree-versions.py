# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4


def test():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.23.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 16

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.24.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 16

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.25.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 17

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.26.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.27.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.28.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.29.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.30.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 19

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 19

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.10.05-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 19

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.14.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.16.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.18.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20
