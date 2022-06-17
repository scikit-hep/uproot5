import numpy
import skhep_testdata

import uproot


def test_num_entries_single():
    file1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    assert list(uproot.num_entries(file1)) == [2304]


def test_num_entries_multiple():
    file1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    file2 = skhep_testdata.data_path("uproot-HZZ.root") + ":events"
    file3 = (
        skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root") + ":sample"
    )
    assert list(uproot.num_entries([file1, file2, file3])) == [2304, 2421, 30]


def test_num_entries_as_iterator():
    file1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    file2 = skhep_testdata.data_path("uproot-HZZ.root") + ":events"
    file3 = (
        skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root") + ":sample"
    )

    i = 0
    vals = [2304, 2421, 30]
    for num in uproot.num_entries([file1, file2, file3]):
        assert num == vals[i]
        i += 1


def test_dict_input():
    file_name1 = skhep_testdata.data_path("uproot-Zmumu.root")
    file_name2 = skhep_testdata.data_path("uproot-HZZ.root")
    file_name3 = skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root")

    i = 0
    vals = [2304, 2421, 30]
    for num in uproot.num_entries(
        {file_name1: "events", file_name2: "events", file_name3: "sample"}
    ):
        assert num == vals[i]
        i += 1
