# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot

ROOT = pytest.importorskip("ROOT")


def test_ZLIB():
    for _ in range(2):
        with uproot.open(skhep_testdata.data_path("uproot-Zmumu-zlib.root"))[
            "events"
        ] as events:
            assert events["px1"].array(entry_stop=5).tolist() == [
                -41.1952876442,
                35.1180497674,
                35.1180497674,
                34.1444372454,
                22.7835819537,
            ]


def test_LZMA():
    pytest.importorskip("lzma")

    for _ in range(2):
        with uproot.open(skhep_testdata.data_path("uproot-Zmumu-lzma.root"))[
            "events"
        ] as events:
            assert events["px1"].array(entry_stop=5).tolist() == [
                -41.1952876442,
                35.1180497674,
                35.1180497674,
                34.1444372454,
                22.7835819537,
            ]


def test_LZ4():
    pytest.importorskip("lz4")

    for _ in range(2):
        with uproot.open(skhep_testdata.data_path("uproot-Zmumu-lz4.root"))[
            "events"
        ] as events:
            assert events["px1"].array(entry_stop=5).tolist() == [
                -41.1952876442,
                35.1180497674,
                35.1180497674,
                34.1444372454,
                22.7835819537,
            ]


def test_ZSTD():
    pytest.importorskip("zstandard")

    for _ in range(2):
        with uproot.open(skhep_testdata.data_path("uproot-Zmumu-zstd.root"))[
            "events"
        ] as events:
            assert events["px1"].array(entry_stop=5).tolist() == [
                -41.1952876442,
                35.1180497674,
                35.1180497674,
                34.1444372454,
                22.7835819537,
            ]


def test_histogram_ZLIB(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    SIZE = 2**21
    histogram = (np.random.randint(0, 10, SIZE), np.linspace(0, 1, SIZE + 1))
    last = histogram[0][-1]

    with uproot.recreate(newfile, compression=uproot.ZLIB(1)) as fout:
        fout["out"] = histogram

    with uproot.open(newfile) as fin:
        content, edges = fin["out"].to_numpy()
        assert len(content) == SIZE
        assert len(edges) == SIZE + 1
        assert content[-1] == last

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetNbinsX() == SIZE
    assert h3.GetBinContent(SIZE) == last
    f3.Close()


def test_histogram_LZMA(tmp_path):
    pytest.importorskip("lzma")

    newfile = os.path.join(tmp_path, "newfile.root")

    SIZE = 2**20
    histogram = (np.random.randint(0, 10, SIZE), np.linspace(0, 1, SIZE + 1))
    last = histogram[0][-1]

    with uproot.recreate(newfile, compression=uproot.LZMA(1)) as fout:
        fout["out"] = histogram

    with uproot.open(newfile) as fin:
        content, edges = fin["out"].to_numpy()
        assert len(content) == SIZE
        assert len(edges) == SIZE + 1
        assert content[-1] == last

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetNbinsX() == SIZE
    assert h3.GetBinContent(SIZE) == last
    f3.Close()


def test_histogram_LZ4(tmp_path):
    pytest.importorskip("lz4")

    newfile = os.path.join(tmp_path, "newfile.root")

    SIZE = 2**21
    histogram = (np.random.randint(0, 10, SIZE), np.linspace(0, 1, SIZE + 1))
    last = histogram[0][-1]

    with uproot.recreate(newfile, compression=uproot.LZ4(1)) as fout:
        fout["out"] = histogram

    with uproot.open(newfile) as fin:
        content, edges = fin["out"].to_numpy()
        assert len(content) == SIZE
        assert len(edges) == SIZE + 1
        assert content[-1] == last

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetNbinsX() == SIZE
    assert h3.GetBinContent(SIZE) == last
    f3.Close()


def test_histogram_ZSTD(tmp_path):
    pytest.importorskip("zstandard")

    newfile = os.path.join(tmp_path, "newfile.root")

    SIZE = 2**21
    histogram = (np.random.randint(0, 10, SIZE), np.linspace(0, 1, SIZE + 1))
    last = histogram[0][-1]

    with uproot.recreate(newfile, compression=uproot.ZSTD(1)) as fout:
        fout["out"] = histogram

    with uproot.open(newfile) as fin:
        content, edges = fin["out"].to_numpy()
        assert len(content) == SIZE
        assert len(edges) == SIZE + 1
        assert content[-1] == last

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetNbinsX() == SIZE
    assert h3.GetBinContent(SIZE) == last
    f3.Close()


def test_flattree_ZLIB(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(100)
    branch2 = 1.1 * np.arange(100)

    with uproot.recreate(newfile, compression=uproot.ZLIB(5)) as fout:
        fout["tree"] = {"branch1": branch1, "branch2": branch2}
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array(library="np").tolist() == branch1.tolist() * 2
        assert fin["tree/branch2"].array(library="np").tolist() == branch2.tolist() * 2

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [x.branch1 for x in t3] == branch1.tolist() * 2
    assert [x.branch2 for x in t3] == branch2.tolist() * 2
    f3.Close()


def test_flattree_LZMA(tmp_path):
    pytest.importorskip("lzma")

    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(100)
    branch2 = 1.1 * np.arange(100)

    with uproot.recreate(newfile, compression=uproot.LZMA(5)) as fout:
        fout["tree"] = {"branch1": branch1, "branch2": branch2}
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array(library="np").tolist() == branch1.tolist() * 2
        assert fin["tree/branch2"].array(library="np").tolist() == branch2.tolist() * 2

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [x.branch1 for x in t3] == branch1.tolist() * 2
    assert [x.branch2 for x in t3] == branch2.tolist() * 2
    f3.Close()


def test_flattree_LZ4(tmp_path):
    pytest.importorskip("lz4")

    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(100)
    branch2 = 1.1 * np.arange(100)

    with uproot.recreate(newfile, compression=uproot.LZ4(5)) as fout:
        fout["tree"] = {"branch1": branch1, "branch2": branch2}
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array(library="np").tolist() == branch1.tolist() * 2
        assert fin["tree/branch2"].array(library="np").tolist() == branch2.tolist() * 2

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [x.branch1 for x in t3] == branch1.tolist() * 2
    assert [x.branch2 for x in t3] == branch2.tolist() * 2
    f3.Close()


def test_flattree_ZSTD(tmp_path):
    pytest.importorskip("zstandard")

    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(100)
    branch2 = 1.1 * np.arange(100)

    with uproot.recreate(newfile, compression=uproot.ZSTD(5)) as fout:
        fout["tree"] = {"branch1": branch1, "branch2": branch2}
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array(library="np").tolist() == branch1.tolist() * 2
        assert fin["tree/branch2"].array(library="np").tolist() == branch2.tolist() * 2

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [x.branch1 for x in t3] == branch1.tolist() * 2
    assert [x.branch2 for x in t3] == branch2.tolist() * 2
    f3.Close()


def test_jaggedtree_ZLIB(tmp_path):
    ak = pytest.importorskip("awkward")

    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = ak.Array([[1, 2, 3], [], [4, 5]] * 10)
    branch2 = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]] * 10)

    with uproot.recreate(newfile, compression=uproot.ZLIB(5)) as fout:
        fout["tree"] = {"branch1": branch1, "branch2": branch2}
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array().tolist() == branch1.tolist() * 2
        assert fin["tree/branch2"].array().tolist() == branch2.tolist() * 2

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [list(x.branch1) for x in t3] == branch1.tolist() * 2
    assert [list(x.branch2) for x in t3] == branch2.tolist() * 2
    f3.Close()


def test_jaggedtree_LZMA(tmp_path):
    pytest.importorskip("lzma")
    ak = pytest.importorskip("awkward")

    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = ak.Array([[1, 2, 3], [], [4, 5]] * 10)
    branch2 = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]] * 10)

    with uproot.recreate(newfile, compression=uproot.LZMA(5)) as fout:
        fout["tree"] = {"branch1": branch1, "branch2": branch2}
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array().tolist() == branch1.tolist() * 2
        assert fin["tree/branch2"].array().tolist() == branch2.tolist() * 2

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [list(x.branch1) for x in t3] == branch1.tolist() * 2
    assert [list(x.branch2) for x in t3] == branch2.tolist() * 2
    f3.Close()


def test_jaggedtree_LZ4(tmp_path):
    pytest.importorskip("lz4")
    ak = pytest.importorskip("awkward")

    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = ak.Array([[1, 2, 3], [], [4, 5]] * 10)
    branch2 = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]] * 10)

    with uproot.recreate(newfile, compression=uproot.LZ4(5)) as fout:
        fout["tree"] = {"branch1": branch1, "branch2": branch2}
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array().tolist() == branch1.tolist() * 2
        assert fin["tree/branch2"].array().tolist() == branch2.tolist() * 2

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [list(x.branch1) for x in t3] == branch1.tolist() * 2
    assert [list(x.branch2) for x in t3] == branch2.tolist() * 2
    f3.Close()


def test_jaggedtree_ZSTD(tmp_path):
    pytest.importorskip("zstandard")
    ak = pytest.importorskip("awkward")

    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = ak.Array([[1, 2, 3], [], [4, 5]] * 10)
    branch2 = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]] * 10)

    with uproot.recreate(newfile, compression=uproot.ZSTD(5)) as fout:
        fout["tree"] = {"branch1": branch1, "branch2": branch2}
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array().tolist() == branch1.tolist() * 2
        assert fin["tree/branch2"].array().tolist() == branch2.tolist() * 2

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [list(x.branch1) for x in t3] == branch1.tolist() * 2
    assert [list(x.branch2) for x in t3] == branch2.tolist() * 2
    f3.Close()


def test_multicompression_1(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(100)
    branch2 = 1.1 * np.arange(100)

    with uproot.recreate(newfile) as fout:
        fout.mktree("tree", {"branch1": branch1.dtype, "branch2": branch2.dtype})
        fout["tree"]["branch1"].compression = uproot.ZLIB(5)
        fout["tree"]["branch2"].compression = None
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array(library="np").tolist() == branch1.tolist()
        assert fin["tree/branch2"].array(library="np").tolist() == branch2.tolist()
        assert fin["tree/branch1"].compressed_bytes < 874
        assert fin["tree/branch2"].compressed_bytes == 874
        assert fin["tree/branch1"].uncompressed_bytes == 874
        assert fin["tree/branch2"].uncompressed_bytes == 874

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [x.branch1 for x in t3] == branch1.tolist()
    assert [x.branch2 for x in t3] == branch2.tolist()
    f3.Close()


def test_multicompression_2(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(100)
    branch2 = 1.1 * np.arange(100)

    with uproot.recreate(newfile) as fout:
        fout.mktree("tree", {"branch1": branch1.dtype, "branch2": branch2.dtype})
        fout["tree"].compression = {"branch1": uproot.ZLIB(5), "branch2": None}
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array(library="np").tolist() == branch1.tolist()
        assert fin["tree/branch2"].array(library="np").tolist() == branch2.tolist()
        assert fin["tree/branch1"].compressed_bytes < 874
        assert fin["tree/branch2"].compressed_bytes == 874
        assert fin["tree/branch1"].uncompressed_bytes == 874
        assert fin["tree/branch2"].uncompressed_bytes == 874

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [x.branch1 for x in t3] == branch1.tolist()
    assert [x.branch2 for x in t3] == branch2.tolist()
    f3.Close()


def test_multicompression_3(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(100)
    branch2 = 1.1 * np.arange(100)

    with uproot.recreate(newfile) as fout:
        fout.mktree("tree", {"branch1": branch1.dtype, "branch2": branch2.dtype})
        fout["tree"].compression = {"branch1": uproot.ZLIB(5), "branch2": None}
        fout["tree"].compression = uproot.ZLIB(5)
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array(library="np").tolist() == branch1.tolist()
        assert fin["tree/branch2"].array(library="np").tolist() == branch2.tolist()
        assert fin["tree/branch1"].compressed_bytes < 874
        assert fin["tree/branch2"].compressed_bytes < 874
        assert fin["tree/branch1"].uncompressed_bytes == 874
        assert fin["tree/branch2"].uncompressed_bytes == 874

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [x.branch1 for x in t3] == branch1.tolist()
    assert [x.branch2 for x in t3] == branch2.tolist()
    f3.Close()


def test_multicompression_4(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(100)
    branch2 = 1.1 * np.arange(100)

    with uproot.recreate(newfile, compression=uproot.ZLIB(5)) as fout:
        fout.mktree("tree", {"branch1": branch1.dtype, "branch2": branch2.dtype})
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array(library="np").tolist() == branch1.tolist()
        assert fin["tree/branch2"].array(library="np").tolist() == branch2.tolist()
        assert fin["tree/branch1"].compressed_bytes < 874
        assert fin["tree/branch2"].compressed_bytes < 874
        assert fin["tree/branch1"].uncompressed_bytes == 874
        assert fin["tree/branch2"].uncompressed_bytes == 874

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [x.branch1 for x in t3] == branch1.tolist()
    assert [x.branch2 for x in t3] == branch2.tolist()
    f3.Close()


def test_multicompression_5(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(100)
    branch2 = 1.1 * np.arange(100)

    with uproot.recreate(newfile, compression=uproot.ZLIB(5)) as fout:
        fout.compression = None
        fout.mktree("tree", {"branch1": branch1.dtype, "branch2": branch2.dtype})
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array(library="np").tolist() == branch1.tolist()
        assert fin["tree/branch2"].array(library="np").tolist() == branch2.tolist()
        assert fin["tree/branch1"].compression is None
        assert fin["tree/branch2"].compression is None
        assert fin["tree/branch1"].compressed_bytes == 874
        assert fin["tree/branch2"].compressed_bytes == 874
        assert fin["tree/branch1"].uncompressed_bytes == 874
        assert fin["tree/branch2"].uncompressed_bytes == 874

    f3 = ROOT.TFile(newfile)
    t3 = f3.Get("tree")
    assert [x.branch1 for x in t3] == branch1.tolist()
    assert [x.branch2 for x in t3] == branch2.tolist()
    f3.Close()
