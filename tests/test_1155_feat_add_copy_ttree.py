import uproot
from skhep_testdata import data_path
import uproot.writing.writable
# import ROOT
import numpy as np

import awkward as ak

def test_vector():
    with uproot.update("/Users/zobil/Documents/samples/uproot-vectorVectorDouble-work.root") as write:
        write.add_branches("tree1", {"branch": int}, source='t')

    with uproot.open("/Users/zobil/Documents/samples/uproot-vectorVectorDouble.root") as read:
        print(read["t"]["x"].arrays())

    with uproot.open("/Users/zobil/Documents/samples/uproot-vectorVectorDouble-work.root") as read:
        print(read["tree1"])
        # print(read["tree1"].all_members)
        # print(read["tree1"]["x"].all_members)
        # print(read["tree1"]["x"].member("fLeaves")[0])


def simple_test():
    with uproot.recreate("arrays.root") as file:
        file['tree'] = {"b1": [1,2,3], "b2": [2,3,4]}

    with uproot.recreate("arrays_check.root") as file:
        file['tree'] = {"b1": [1,2,3], "b2": [2,3,4]}
        
    with uproot.open("arrays.root", minimal_ttree_metadata=False) as read:
        print(read['tree']['b1'].all_members)

    with uproot.update("arrays.root") as write:
        write.add("tree", {"b3": [5,6,7]}, source='tree')

    with uproot.open("arrays.root") as new:
        print(new['tree'].keys())
        print(new['tree'].member("fBranches"))

        # for key in 

simple_test()