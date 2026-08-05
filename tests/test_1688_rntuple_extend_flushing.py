# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Regression tests for issue #1688: RNTuple extension flushing and serialization.

``add_rblob`` flushed the sink for every blob it wrote, so a single ``extend``
flushed once per column plus once for the page list and once for the footer.
Every extension also re-serialized every cluster group record written so far.
"""

from __future__ import annotations

import numpy as np
import pytest

import uproot
import uproot.sink.file
import uproot.writing._cascadentuple


@pytest.fixture
def count_flushes(monkeypatch):
    counter = {"flushes": 0}
    original = uproot.sink.file.FileSink.flush

    def counting_flush(self):
        counter["flushes"] += 1
        return original(self)

    monkeypatch.setattr(uproot.sink.file.FileSink, "flush", counting_flush)
    return counter


@pytest.mark.parametrize("num_columns", [1, 8, 32])
def test_extend_flushes_once_regardless_of_column_count(
    tmp_path, count_flushes, num_columns
):
    path = str(tmp_path / "file.root")
    chunk = {f"c{i}": np.arange(10, dtype=np.int64) for i in range(num_columns)}

    with uproot.recreate(path) as f:
        ntuple = f.mkrntuple("nt", {k: np.dtype("int64") for k in chunk})
        count_flushes["flushes"] = 0
        ntuple.extend(chunk)
        assert count_flushes["flushes"] == 1

        count_flushes["flushes"] = 0
        ntuple.extend(chunk)
        ntuple.extend(chunk)
        assert count_flushes["flushes"] == 2


def test_repeated_extension_round_trips(tmp_path):
    path = str(tmp_path / "file.root")
    columns = {f"c{i}": np.dtype("int64") for i in range(3)}

    with uproot.recreate(path) as f:
        ntuple = f.mkrntuple("nt", columns)
        for step in range(20):
            ntuple.extend(
                {name: np.arange(10 * step, 10 * (step + 1)) for name in columns}
            )

    with uproot.open(path) as f:
        arrays = f["nt"].arrays()
        assert len(arrays) == 200
        for name in columns:
            assert arrays[name].tolist() == list(range(200))


def test_cluster_group_record_serialization_is_stable_and_cached():
    envlink = uproot.writing._cascadentuple.NTuple_EnvLink(
        48, uproot.writing._cascadentuple.NTuple_Locator(48, 1024)
    )
    record = uproot.writing._cascadentuple.NTuple_ClusterGroupRecord(0, 10, 1, envlink)

    first = record.serialize()
    second = record.serialize()

    assert first == second
    # cached, so the same object is handed back rather than rebuilt
    assert first is second


def test_data_is_on_disk_after_each_extend(tmp_path):
    """The sink is flushed at the end of every extend, not only at close."""
    path = str(tmp_path / "file.root")

    with uproot.recreate(path) as f:
        ntuple = f.mkrntuple("nt", {"x": np.dtype("int64")})
        ntuple.extend({"x": np.arange(10)})

        with uproot.open(path) as check:
            assert check["nt"].num_entries == 10

        ntuple.extend({"x": np.arange(10, 20)})

        with uproot.open(path) as check:
            assert check["nt"].num_entries == 20
            assert check["nt"].arrays()["x"].tolist() == list(range(20))
