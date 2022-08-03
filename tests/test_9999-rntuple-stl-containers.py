# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import json
import queue
import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test_rntuple_stl_containers():
    filename = skhep_testdata.data_path("test_ntuple_stl_containers.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert R.keys() == ['string', 'vector_int32', 'vector_vector_int32', 
                'vector_string', 'vector_vector_string', 'variant_int32_float', 
                'vector_variant_int32_float', 'tuple_int32_string', 'vector_tuple_int32_string']
        R.arrays()
