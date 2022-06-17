import numpy
import skhep_testdata

import uproot


def test_num_entries_single():
    file1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    assert list(uproot.num_entries(file1)) == [
        ("/home/kmk/.local/skhepdata/uproot-Zmumu.root", "events", 2304)
    ]


def test_num_entries_multiple():
    file1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    file2 = skhep_testdata.data_path("uproot-HZZ.root") + ":events"
    file3 = (
        skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root") + ":sample"
    )

    assert list(uproot.num_entries([file1, file2, file3])) == [
        ("/home/kmk/.local/skhepdata/uproot-Zmumu.root", "events", 2304),
        ("/home/kmk/.local/skhepdata/uproot-HZZ.root", "events", 2421),
        (
            "/home/kmk/.local/skhepdata/uproot-sample-6.08.04-uncompressed.root",
            "sample",
            30,
        ),
    ]


def test_num_entries_as_iterator():
    file1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    file2 = skhep_testdata.data_path("uproot-HZZ.root") + ":events"
    file3 = (
        skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root") + ":sample"
    )

    vals = [
        ("/home/kmk/.local/skhepdata/uproot-Zmumu.root", "events", 2304),
        ("/home/kmk/.local/skhepdata/uproot-HZZ.root", "events", 2421),
        (
            "/home/kmk/.local/skhepdata/uproot-sample-6.08.04-uncompressed.root",
            "sample",
            30,
        ),
    ]
    for i, num in enumerate(uproot.num_entries([file1, file2, file3])):
        assert num == vals[i]


def test_dict_input():
    file_name1 = skhep_testdata.data_path("uproot-Zmumu.root")
    file_name2 = skhep_testdata.data_path("uproot-HZZ.root")
    file_name3 = skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root")

    vals = [
        ("/home/kmk/.local/skhepdata/uproot-Zmumu.root", "events", 2304),
        ("/home/kmk/.local/skhepdata/uproot-HZZ.root", "events", 2421),
        (
            "/home/kmk/.local/skhepdata/uproot-sample-6.08.04-uncompressed.root",
            "sample",
            30,
        ),
    ]
    for i, num in enumerate(
        uproot.num_entries(
            {file_name1: "events", file_name2: "events", file_name3: "sample"}
        )
    ):
        assert num == vals[i]
