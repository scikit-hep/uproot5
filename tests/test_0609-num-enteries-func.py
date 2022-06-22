# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import skhep_testdata

import uproot


def test_num_entries_single():
    file1 = skhep_testdata.data_path("uproot-Zmumu.root")
    ttree1 = file1 + ":events"
    assert list(uproot.num_entries(ttree1)) == [(file1, "events", 2304)]


def test_num_entries_multiple():
    file1 = skhep_testdata.data_path("uproot-Zmumu.root")
    ttree1 = file1 + ":events"
    file2 = skhep_testdata.data_path("uproot-HZZ.root")
    ttree2 = file2 + ":events"
    file3 = skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root")
    ttree3 = file3 + ":sample"

    assert list(uproot.num_entries([ttree1, ttree2, ttree3])) == [
        (file1, "events", 2304),
        (file2, "events", 2421),
        (file3, "sample", 30),
    ]


def test_num_entries_as_iterator():
    file1 = skhep_testdata.data_path("uproot-Zmumu.root")
    ttree1 = file1 + ":events"
    file2 = skhep_testdata.data_path("uproot-HZZ.root")
    ttree2 = file2 + ":events"
    file3 = skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root")
    ttree3 = file3 + ":sample"

    vals = [
        (file1, "events", 2304),
        (file2, "events", 2421),
        (file3, "sample", 30),
    ]
    for i, num in enumerate(uproot.num_entries([ttree1, ttree2, ttree3])):
        assert num == vals[i]


def test_dict_input():
    file1 = skhep_testdata.data_path("uproot-Zmumu.root")
    file2 = skhep_testdata.data_path("uproot-HZZ.root")
    file3 = skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root")

    vals = [
        (file1, "events", 2304),
        (file2, "events", 2421),
        (file3, "sample", 30),
    ]
    for i, num in enumerate(
        uproot.num_entries({file1: "events", file2: "events", file3: "sample"})
    ):
        assert num == vals[i]
