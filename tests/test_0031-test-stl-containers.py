# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4
from uproot4.stl_containers import AsString
from uproot4.stl_containers import AsVector
from uproot4.stl_containers import AsSet
from uproot4.stl_containers import AsMap


# def test_typename():
#     with uproot4.open(
#         "stl_containers.root"
#     )["tree"] as tree:
#         print("\n".join("{}:\t{}\t{}".format(branch.name, branch.typename, branch.interpretation) for branch in tree.values()))

#     raise Exception


# def test_simple():
#     with uproot4.open(
#         "stl_containers.root"
#     )["tree"] as tree:
#         print(tree["vector_int32"].array())

#     raise Exception
